/**
 * CT Clinical Enhancer — Klinik Ön Yüz (tek dosya)
 *
 * Mimari kurallar (CLAUDE.md):
 * - Kaydırıcı harmanlaması backend'i ÇAĞIRMAZ. /api/v1/enhance-slice yanıtındaki
 *   Base64 float32 tamponları (i_orig, delta, lap_sharpen, tv_map, delta_contrast)
 *   bir kez çözülür; denklem 2.1 Canvas üzerinde 60 FPS uygulanır:
 *     I_out = clip(I_orig + α·φ_γ(ΔI) + β·lap − λ_TV·tv + κ_HU·ΔI_contrast, −1024, 3071)
 *     φ_γ(ΔI) = ΔI (ΔI≥0) | γ·ΔI (ΔI<0)
 * - Backend kapalıyken aynı denklemlerle sentetik demo verisi üretilir (çevrimdışı mod).
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

// ====================================================================== //
// Sabitler
// ====================================================================== //

// Varsayılan backend; `?api=http://host:port/api/v1` ile geçersiz kılınabilir
// (ulaşılamazsa uygulama çevrimdışı demo moduna düşer).
const API = new URLSearchParams(window.location.search).get("api")
  ?? "http://localhost:8000/api/v1";
const WS_URL = `${API.replace(/^http/, "ws")}/ws/jobs`;
const HU_MIN = -1024;
const HU_MAX = 3071;

// ---------------------------------------------------------------------- //
// Dosya yükleme yardımcıları: tarayıcı güvenlik gereği mutlak yol VERMEZ;
// sürüklenen/seçilen dosyaların içeriği multipart olarak backend'in
// /ingest-upload uç noktasına gönderilir (göreli yollar korunur).
// ---------------------------------------------------------------------- //
const UPLOADABLE_RE = /\.(dcm|dicom|ima|nii|nii\.gz|hdf5|h5|hdf|zip|rar)$/i;

function shouldUpload(relPath: string): boolean {
  const base = relPath.split("/").pop() ?? "";
  if (!base || base.startsWith(".")) return false; // .DS_Store vb.
  return UPLOADABLE_RE.test(base) || !base.includes("."); // uzantısız DICOM'lar dahil
}

interface PickedFile { file: File; rel: string }

/** Sürüklenen klasör ağacını özyinelemeli dolaşır (FileSystemEntry API).
 *  DİKKAT: webkitGetAsEntry drop olayının İÇİNDE eşzamanlı çağrılmalıdır;
 *  bu fonksiyona hazır entry listesi verilir. */
async function walkDroppedEntries(entries: unknown[]): Promise<PickedFile[]> {
  const out: PickedFile[] = [];
  async function walk(entry: any, prefix: string): Promise<void> {
    if (!entry) return;
    if (entry.isFile) {
      const file: File = await new Promise((res, rej) => entry.file(res, rej));
      out.push({ file, rel: prefix + entry.name });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      // readEntries bir seferde en çok ~100 öğe verir; boş gelene dek oku
      for (;;) {
        const batch: any[] = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (batch.length === 0) break;
        for (const child of batch) await walk(child, prefix + entry.name + "/");
      }
    }
  }
  for (const e of entries) await walk(e, "");
  return out;
}

function labelOf(picked: PickedFile[]): string {
  const first = picked[0];
  if (first.rel.includes("/")) return first.rel.split("/")[0];
  return first.file.name.replace(/\.[^.]*$/, "") || "vaka";
}

type ParamKey = "alpha" | "gamma" | "lambda_tv" | "beta" | "kappa_hu";

interface BlendParams {
  alpha: number;
  gamma: number;
  lambda_tv: number;
  beta: number;
  kappa_hu: number;
}

const DEFAULT_PARAMS: BlendParams = {
  alpha: 1.0,
  gamma: 0.25,
  lambda_tv: 0.0,
  beta: 0.0,
  kappa_hu: 0.0,
};

const PARAM_DEFS: {
  key: ParamKey;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
}[] = [
  { key: "alpha", label: "α · Gürültü Giderme", hint: "Artık harmanlama oranı", min: 0, max: 1.5, step: 0.01 },
  { key: "gamma", label: "γ · PReLU Negatif Eğim", hint: "Negatif artık ölçeği", min: 0.01, max: 1.0, step: 0.01 },
  { key: "lambda_tv", label: "λ_TV · TV Bastırma", hint: "Granüler bölge pürüzsüzleştirme", min: 0, max: 0.5, step: 0.005 },
  { key: "beta", label: "β · Kenar Keskinleştirme", hint: "−Laplacian(Î_DL) katkısı", min: 0, max: 1.0, step: 0.01 },
  { key: "kappa_hu", label: "κ_HU · Vasküler Kontrast", hint: "ContrastGAN lümen kazancı", min: 0, max: 2.0, step: 0.01 },
];

const FALLBACK_MODELS = [
  "rdn", "noise2sim", "contrast_gan", "srcnn", "real_esrgan", "swinir", "hat", "dat",
];

const MODEL_LABELS: Record<string, string> = {
  rdn: "RDN (gürültü giderme)",
  noise2sim: "Noise2Sim (öz-denetimli)",
  contrast_gan: "ContrastGAN (kontrast)",
  srcnn: "SRCNN (süper çözünürlük)",
  real_esrgan: "Real-ESRGAN",
  swinir: "SwinIR",
  hat: "HAT",
  dat: "DAT",
  auto_best: "Otomatik En İyi",
};

const WINDOW_PRESETS = [
  { name: "Yumuşak Doku", wl: 50, ww: 400 },
  { name: "Akciğer", wl: -600, ww: 1500 },
  { name: "Kemik", wl: 500, ww: 2000 },
  { name: "Anjiyo", wl: 150, ww: 600 },
];

// ====================================================================== //
// Tipler
// ====================================================================== //

interface SliceBuffers {
  w: number;
  h: number;
  iOrig: Float32Array;
  delta: Float32Array;
  lap: Float32Array;
  tv: Float32Array;
  dc: Float32Array;
  // HU sadakati skalerleri (backend residual_blender ile aynı): doku
  // maskesinde E[ΔI⁺] ve E[ΔI⁻]; harmanlamada α·(m⁺+γ·m⁻) geri çıkarılır.
  phiBiasPos: number;
  phiBiasNeg: number;
}

interface SliceMeta {
  modelId: string;
  fallback: boolean;
  runtimeS: number;
  sigmaBefore: number;
  sigmaAfter: number;
  demo: boolean;
  autoBestCandidates: { model_id: string; score?: number; sigma_after?: number }[] | null;
}

interface PhaseMetric {
  phase_name: string;
  num_slices: number;
  sigma_hu: number;
  mu_peak_hu: number;
  niqe: number;
  pi: number;
  ssim_inter: number;
  n_k: number;
  c_k: number;
  m_k: number;
  q_k: number;
  s_faz: number;
  recommended: boolean;
}

interface CaseInfo {
  caseId: string;
  label: string;
  demo: boolean;
  phases: { name: string; numSlices: number }[];
}

interface Preset {
  id: number;
  name: string;
  model_type: string;
  alpha: number;
  gamma: number;
  lambda_tv: number;
  beta: number;
  kappa_hu: number;
  window_center: number;
  window_width: number;
}

interface BatchRow {
  path: string;
  status: string;
  error?: string;
  case_id?: string;
  recommended_phase?: string;
  selected_phase?: string;
  phase_score?: number;
  sigma_hu?: number;
  mu_peak_hu?: number;
  niqe?: number;
  ssim_inter?: number;
  metadata?: Record<string, number | string>;
  phases?: PhaseMetric[];
  suggested_params?: Record<string, number | string>;
  selected?: boolean;
}

// ====================================================================== //
// Yardımcılar: Base64 float32 çözme + harmanlama + pencereleme
// ====================================================================== //

function b64ToF32(b64: string): Float32Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer, 0, bytes.length >> 2);
}

/** Denklem 2.1 — sunucudaki residual_blender.blend ile birebir aynı. */
function blendInto(out: Float32Array, buf: SliceBuffers, p: BlendParams): void {
  const { iOrig, delta, lap, tv, dc } = buf;
  const a = p.alpha, g = p.gamma, b = p.beta, l = p.lambda_tv, k = p.kappa_hu;
  // φ_γ'nin doku ortalamasına verdiği sabit kayma (backend ile birebir aynı)
  const bias = a * (buf.phiBiasPos + g * buf.phiBiasNeg);
  const n = iOrig.length;
  for (let i = 0; i < n; i++) {
    const d = delta[i];
    let v = iOrig[i] + a * (d >= 0 ? d : g * d) - bias;
    if (b > 0) v += b * lap[i];
    if (l > 0) v -= l * tv[i];
    if (k > 0) v += k * dc[i];
    out[i] = v < HU_MIN ? HU_MIN : v > HU_MAX ? HU_MAX : v;
  }
}

/** Doku maskesinde E[ΔI⁺] ve E[ΔI⁻] (demo üretici için; backend ile aynı). */
function phiBiasOf(iOrig: Float32Array, delta: Float32Array): [number, number] {
  let pos = 0, neg = 0, n = 0;
  for (let i = 0; i < iOrig.length; i++) {
    if (iOrig[i] > -300 && iOrig[i] < 300) {
      n++;
      if (delta[i] > 0) pos += delta[i];
      else neg += delta[i];
    }
  }
  return n > 64 ? [pos / n, neg / n] : [0, 0];
}

/** HU → 8-bit gri (pencere/seviye). */
function drawGray(
  ctx: CanvasRenderingContext2D, data: Float32Array, w: number, h: number,
  wl: number, ww: number, imageData: ImageData,
): void {
  const lo = wl - ww / 2;
  const scale = 255 / Math.max(ww, 1);
  const px = imageData.data;
  for (let i = 0, j = 0; i < data.length; i++, j += 4) {
    let v = (data[i] - lo) * scale;
    v = v < 0 ? 0 : v > 255 ? 255 : v;
    px[j] = px[j + 1] = px[j + 2] = v;
    px[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

/** Termal LUT (inferno benzeri): t∈[0,1] → [r,g,b]. */
const THERMAL_LUT: Uint8Array = (() => {
  const stops = [
    [0, [8, 8, 20]], [0.25, [87, 16, 110]], [0.5, [188, 55, 84]],
    [0.75, [249, 142, 9]], [1, [252, 255, 164]],
  ] as [number, number[]][];
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let s = 0;
    while (s < stops.length - 2 && t > stops[s + 1][0]) s++;
    const [t0, c0] = stops[s];
    const [t1, c1] = stops[s + 1];
    const f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(c0[c] + f * (c1[c] - c0[c]));
  }
  return lut;
})();

/** |ΔI| mutlak HU farkını termal haritayla çizer (0..diffMax HU). */
function drawThermal(
  ctx: CanvasRenderingContext2D, iOut: Float32Array, iOrig: Float32Array,
  w: number, h: number, diffMax: number, imageData: ImageData,
): void {
  const px = imageData.data;
  const scale = 255 / Math.max(diffMax, 1);
  for (let i = 0, j = 0; i < iOut.length; i++, j += 4) {
    let v = Math.abs(iOut[i] - iOrig[i]) * scale;
    v = v > 255 ? 255 : v;
    const idx = (v | 0) * 3;
    px[j] = THERMAL_LUT[idx];
    px[j + 1] = THERMAL_LUT[idx + 1];
    px[j + 2] = THERMAL_LUT[idx + 2];
    px[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

/** SNR/CNR/σ_HU kestirimi: yumuşak doku yüksek-geçiren gürültü + vasküler ROI. */
function computeStats(iOut: Float32Array, dc: Float32Array, w: number, h: number) {
  let sumSoft = 0, nSoft = 0, sumVes = 0, nVes = 0, sumHp2 = 0, nHp = 0;
  for (let y = 1; y < h - 1; y += 2) {
    for (let x = 1; x < w - 1; x += 2) {
      const i = y * w + x;
      const v = iOut[i];
      if (v > -100 && v < 200) {
        sumSoft += v; nSoft++;
        const l = iOut[i - 1], r = iOut[i + 1], u = iOut[i - w], d = iOut[i + w];
        // Doku/hava ve doku/damar kenarları kestirimi şişirmesin: 4 komşu da ROI içinde olmalı
        if (l > -100 && l < 200 && r > -100 && r < 200 &&
            u > -100 && u < 200 && d > -100 && d < 200) {
          const hp = v - (l + r + u + d) * 0.25;
          sumHp2 += hp * hp; nHp++;
        }
      }
      if (v > 200 || dc[i] > 1) { sumVes += v; nVes++; }
    }
  }
  const meanSoft = nSoft ? sumSoft / nSoft : 0;
  // 4-komşu yüksek geçiren filtre varyansı ≈ 1.25·σ² → σ = sqrt(var/1.25)
  const sigma = nHp ? Math.sqrt(sumHp2 / nHp / 1.25) : 0;
  const meanVes = nVes ? sumVes / nVes : meanSoft;
  return {
    sigma,
    snr: sigma > 0.01 ? Math.abs(meanSoft) / sigma : 0,
    cnr: sigma > 0.01 ? Math.abs(meanVes - meanSoft) / sigma : 0,
  };
}

// ====================================================================== //
// Çevrimdışı demo modu: sentetik kardiyak fantom + aynı denklem bileşenleri
// ====================================================================== //

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const DEMO_SIZE = 256;

const DEMO_PHASES: PhaseMetric[] = [
  mkDemoPhase("%0 (sistol)", 34.2, 318, 5.9, 0.78, 0.74, 0.58, 0.62, false),
  mkDemoPhase("%20", 30.8, 331, 5.1, 0.84, 0.79, 0.66, 0.70, false),
  mkDemoPhase("%40 (diyastol-erken)", 26.5, 349, 4.3, 0.90, 0.85, 0.79, 0.78, false),
  mkDemoPhase("%75 (diyastol-geç)", 23.9, 362, 3.6, 0.95, 0.90, 0.93, 0.88, true),
  mkDemoPhase("%90", 28.4, 344, 4.7, 0.88, 0.82, 0.74, 0.75, false),
];

function mkDemoPhase(
  name: string, sigma: number, mu: number, niqe: number,
  n: number, c: number, m: number, q: number, rec: boolean,
): PhaseMetric {
  return {
    phase_name: name, num_slices: 24, sigma_hu: sigma, mu_peak_hu: mu,
    niqe, pi: (10 - 5 + niqe) / 2, ssim_inter: m,
    n_k: n, c_k: c, m_k: m, q_k: q,
    s_faz: Math.round(100 * (0.3 * n + 0.25 * c + 0.25 * m + 0.2 * q) * 10) / 10,
    recommended: rec,
  };
}

/** Sentetik kardiyak BT kesiti: HU uzayında fantom + faz bağımlı gürültü. */
function genDemoSlice(phaseIdx: number, sliceIdx: number): SliceBuffers {
  const s = DEMO_SIZE;
  const iOrig = new Float32Array(s * s);
  const rnd = mulberry32(1337 + phaseIdx * 101 + sliceIdx * 7);
  const sigma = DEMO_PHASES[phaseIdx]?.sigma_hu ?? 28;
  const cx = s / 2, cy = s / 2;
  const zF = Math.sin((sliceIdx / 23) * Math.PI); // kesit boyunca anatomi ölçeği
  const motion = 1 - (DEMO_PHASES[phaseIdx]?.m_k ?? 0.8); // hareket artefakt şiddeti

  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      let v = -1000; // hava
      const dxB = (x - cx) / (s * 0.46), dyB = (y - cy) / (s * 0.38);
      if (dxB * dxB + dyB * dyB < 1) {
        v = 35; // yumuşak doku gövde
        // akciğerler
        const dxL = (x - cx * 0.55) / (s * 0.14), dyL = (y - cy) / (s * 0.22);
        const dxR = (x - cx * 1.45) / (s * 0.14), dyR = (y - cy) / (s * 0.22);
        if (dxL * dxL + dyL * dyL < 1 || dxR * dxR + dyR * dyR < 1) v = -820;
        // kalp gölgesi
        const hx = x - cx * 0.95 + motion * 6 * Math.sin(y * 0.2);
        const hy = y - cy * 1.1;
        const rh = s * (0.13 + 0.05 * zF);
        if (hx * hx + hy * hy < rh * rh) {
          v = 45;
          // kontrastlı LV lümeni
          const lx = hx + rh * 0.25, ly = hy + rh * 0.1;
          const rl = rh * 0.55;
          if (lx * lx + ly * ly < rl * rl) v = 330 + 25 * zF;
          // koroner kesitleri (küçük kontrast noktaları)
          const c1x = hx - rh * 0.75, c1y = hy - rh * 0.55;
          const c2x = hx + rh * 0.85, c2y = hy - rh * 0.3;
          if (c1x * c1x + c1y * c1y < 9 || c2x * c2x + c2y * c2y < 9) v = 390;
        }
        // inen aort
        const ax = x - cx * 1.08, ay = y - cy * 1.52;
        if (ax * ax + ay * ay < 52) v = 310;
        // omurga
        const sx = x - cx, sy = y - cy * 1.62;
        const ds = sx * sx + sy * sy;
        if (ds < 230) v = ds < 90 ? 210 : 720;
      }
      // Box-Muller gürültü + faza bağlı σ
      const n1 = Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(2 * Math.PI * rnd());
      iOrig[i] = v + n1 * sigma;
    }
  }

  // Î_DL ≈ kenar koruyucusuz 2 geçişli 3×3 kutu bulanıklığı (σ 30 → ~8)
  const iDl = boxBlur3(boxBlur3(iOrig, s), s);
  const delta = new Float32Array(s * s);
  for (let i = 0; i < delta.length; i++) delta[i] = iDl[i] - iOrig[i];

  // ∇_lap(Î_DL) = −Laplacian (keskinleştirme yönü)
  const lap = new Float32Array(s * s);
  for (let y = 1; y < s - 1; y++) {
    for (let x = 1; x < s - 1; x++) {
      const i = y * s + x;
      lap[i] = -(4 * iDl[i] - iDl[i - 1] - iDl[i + 1] - iDl[i - s] - iDl[i + s]);
    }
  }

  // TV(ΔI) = |∂x ΔI| + |∂y ΔI|
  const tv = new Float32Array(s * s);
  for (let y = 1; y < s - 1; y++) {
    for (let x = 1; x < s - 1; x++) {
      const i = y * s + x;
      tv[i] = Math.abs(delta[i + 1] - delta[i - 1]) * 0.5 +
              Math.abs(delta[i + s] - delta[i - s]) * 0.5;
    }
  }

  // ΔI_contrast: vasküler eşik maskesi üzerinde yumuşatılmış kazanç
  const dcRaw = new Float32Array(s * s);
  for (let i = 0; i < dcRaw.length; i++) dcRaw[i] = iDl[i] > 180 ? 90 : 0;
  const dc = boxBlur3(dcRaw, s);

  const [phiBiasPos, phiBiasNeg] = phiBiasOf(iOrig, delta);
  return { w: s, h: s, iOrig, delta, lap, tv, dc, phiBiasPos, phiBiasNeg };
}

function boxBlur3(src: Float32Array, s: number): Float32Array {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      const l = x > 0 ? src[i - 1] : src[i];
      const r = x < s - 1 ? src[i + 1] : src[i];
      tmp[i] = (l + src[i] + r) / 3;
    }
  }
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const i = y * s + x;
      const u = y > 0 ? tmp[i - s] : tmp[i];
      const d = y < s - 1 ? tmp[i + s] : tmp[i];
      out[i] = (u + tmp[i] + d) / 3;
    }
  }
  return out;
}

function demoBatchRow(path: string, idx: number): BatchRow {
  const rnd = mulberry32(900 + idx * 31);
  const best = DEMO_PHASES.reduce((a, b) => (b.s_faz > a.s_faz ? b : a));
  const jitter = (v: number, f: number) => Math.round((v + (rnd() - 0.5) * f) * 100) / 100;
  return {
    path,
    status: "ok",
    case_id: `demo-${idx + 1}`,
    recommended_phase: best.phase_name,
    selected_phase: best.phase_name,
    phase_score: jitter(best.s_faz, 8),
    sigma_hu: jitter(best.sigma_hu, 6),
    mu_peak_hu: jitter(best.mu_peak_hu, 30),
    niqe: jitter(best.niqe, 1.2),
    ssim_inter: jitter(best.ssim_inter, 0.06),
    metadata: { kvp: 120, mas: 300 + Math.round(rnd() * 200), slice_thickness: 0.625 },
    phases: DEMO_PHASES,
    suggested_params: {
      model_id: idx % 2 ? "noise2sim" : "rdn",
      alpha: jitter(1.0, 0.3), gamma: jitter(0.25, 0.1),
      lambda_tv: jitter(0.05, 0.05), beta: jitter(0.15, 0.1), kappa_hu: jitter(0.4, 0.3),
    },
    selected: true,
  };
}

// ====================================================================== //
// Stil (dark-slate medikal tema) — tek dosya gereği gömülü CSS
// ====================================================================== //

const CSS = `
:root {
  --bg: #0b1120; --panel: #111a2e; --panel2: #16213a; --border: #233150;
  --text: #dbe4f3; --dim: #8296b8; --accent: #38bdf8; --accent2: #22d3ee;
  --good: #34d399; --warn: #fbbf24; --bad: #f87171; --violet: #a78bfa;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text);
  font: 13px/1.45 "SF Pro Text", "Segoe UI", system-ui, sans-serif; }
#root { min-height: 100vh; }
.app { display: flex; flex-direction: column; min-height: 100vh; }
.hdr { display: flex; align-items: center; gap: 14px; padding: 10px 18px;
  background: linear-gradient(180deg, #101a30, #0d1526); border-bottom: 1px solid var(--border); }
.hdr h1 { font-size: 16px; margin: 0; font-weight: 600; letter-spacing: 0.3px; }
.hdr h1 span { color: var(--accent); }
.pill { padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.pill.on { background: rgba(52,211,153,.15); color: var(--good); border: 1px solid rgba(52,211,153,.4); }
.pill.off { background: rgba(251,191,36,.12); color: var(--warn); border: 1px solid rgba(251,191,36,.4); }
.tabs { display: flex; gap: 4px; margin-left: auto; }
.tabs button { background: transparent; border: 1px solid var(--border); color: var(--dim);
  padding: 6px 14px; border-radius: 8px; cursor: pointer; font-size: 12px; }
.tabs button.act { background: var(--panel2); color: var(--accent); border-color: var(--accent); }
.main { display: flex; flex: 1; gap: 10px; padding: 10px; align-items: stretch; min-height: 0; }
.col { background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
  padding: 12px; overflow-y: auto; }
.left { width: 265px; flex: none; display: flex; flex-direction: column; gap: 12px; }
.center { flex: 1; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
.right { width: 300px; flex: none; display: flex; flex-direction: column; gap: 14px; }
.sec-t { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--dim);
  margin: 0 0 8px; font-weight: 700; }
.drop { border: 2px dashed var(--border); border-radius: 10px; padding: 16px 10px; text-align: center;
  color: var(--dim); cursor: pointer; transition: all .15s; font-size: 12px; }
.drop:hover, .drop.over { border-color: var(--accent); color: var(--accent); background: rgba(56,189,248,.06); }
.drop .big { font-size: 22px; display: block; margin-bottom: 4px; }
.pathrow { display: flex; gap: 6px; margin-top: 8px; }
input[type=text] { flex: 1; background: var(--bg); border: 1px solid var(--border); color: var(--text);
  border-radius: 7px; padding: 6px 9px; font-size: 12px; min-width: 0; }
input[type=text]:focus { outline: none; border-color: var(--accent); }
.btn { background: var(--panel2); color: var(--text); border: 1px solid var(--border);
  border-radius: 7px; padding: 6px 12px; cursor: pointer; font-size: 12px; white-space: nowrap; }
.btn:hover { border-color: var(--accent); color: var(--accent); }
.btn:disabled { opacity: .45; cursor: default; }
.btn.primary { background: linear-gradient(135deg, #0ea5e9, #0369a1); border: none; color: #fff; font-weight: 600; }
.btn.primary:hover { filter: brightness(1.15); color: #fff; }
.btn.violet { background: linear-gradient(135deg, #8b5cf6, #6d28d9); border: none; color: #fff; font-weight: 600; }
.btn.violet:hover { filter: brightness(1.15); }
.btn.danger:hover { border-color: var(--bad); color: var(--bad); }
.phase { border: 1px solid var(--border); border-radius: 9px; padding: 8px 10px; margin-bottom: 7px;
  cursor: pointer; position: relative; transition: border-color .12s; }
.phase:hover { border-color: var(--accent); }
.phase.sel { border-color: var(--accent); background: rgba(56,189,248,.08); }
.phase .nm { font-weight: 600; font-size: 12.5px; }
.phase .met { color: var(--dim); font-size: 11px; margin-top: 3px; display: flex; gap: 8px; flex-wrap: wrap; }
.phase .score { position: absolute; right: 9px; top: 8px; font-weight: 700; color: var(--accent2); font-size: 13px; }
.badge-best { display: inline-block; background: linear-gradient(135deg,#059669,#10b981); color: #fff;
  font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 999px; margin-left: 6px; vertical-align: 1px; }
.sbar { height: 4px; border-radius: 2px; background: var(--border); margin-top: 6px; overflow: hidden; }
.sbar > div { height: 100%; background: linear-gradient(90deg, var(--accent), var(--good)); }
.canvases { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
.cv-wrap { background: #000; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; position: relative; }
.cv-wrap .cap { position: absolute; top: 0; left: 0; right: 0; padding: 4px 9px; font-size: 11px; font-weight: 600;
  background: linear-gradient(180deg, rgba(5,10,20,.85), transparent); display: flex; gap: 6px; align-items: center; z-index: 2; }
.cv-wrap canvas { display: block; width: 100%; image-rendering: pixelated; cursor: crosshair; }
.xh { position: absolute; background: rgba(56,189,248,.55); pointer-events: none; z-index: 1; display: none; }
.xh.h { left: 0; right: 0; height: 1px; }
.xh.v { top: 0; bottom: 0; width: 1px; }
.fb { background: rgba(251,191,36,.18); color: var(--warn); font-size: 9.5px; padding: 1px 6px;
  border-radius: 999px; border: 1px solid rgba(251,191,36,.4); font-weight: 700; }
.cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 8px 11px; }
.card .k { font-size: 10px; color: var(--dim); text-transform: uppercase; letter-spacing: .6px; }
.card .v { font-size: 17px; font-weight: 700; margin-top: 2px; color: var(--accent2); font-variant-numeric: tabular-nums; }
.card .v.g { color: var(--good); } .card .v.n { color: var(--text); }
.readout { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 7px 12px;
  font-size: 12px; color: var(--dim); display: flex; gap: 16px; font-variant-numeric: tabular-nums; min-height: 33px; align-items: center; }
.readout b { color: var(--text); }
.slider-row { margin-bottom: 11px; }
.slider-row .lr { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2px; }
.slider-row label { font-size: 12px; font-weight: 600; }
.slider-row .hint { font-size: 10px; color: var(--dim); }
.slider-row .val { font-variant-numeric: tabular-nums; color: var(--accent2); font-weight: 700; font-size: 12px; }
input[type=range] { width: 100%; accent-color: var(--accent); height: 20px; cursor: pointer; }
select { background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 7px;
  padding: 6px 8px; font-size: 12px; width: 100%; }
.wbtns { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 5px; }
.wbtns button { font-size: 10px; padding: 3px 8px; background: var(--bg); border: 1px solid var(--border);
  color: var(--dim); border-radius: 999px; cursor: pointer; }
.wbtns button:hover { color: var(--accent); border-color: var(--accent); }
.preset { display: flex; align-items: center; gap: 7px; border: 1px solid var(--border); border-radius: 8px;
  padding: 6px 9px; margin-bottom: 6px; cursor: pointer; font-size: 12px; }
.preset:hover { border-color: var(--accent); }
.preset .pn { flex: 1; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.preset .pm { color: var(--dim); font-size: 10.5px; }
.preset .del { background: none; border: none; color: var(--dim); cursor: pointer; font-size: 14px; padding: 0 2px; }
.preset .del:hover { color: var(--bad); }
.msg { padding: 8px 12px; border-radius: 9px; font-size: 12px; margin-top: 8px; }
.msg.info { background: rgba(56,189,248,.1); color: var(--accent); border: 1px solid rgba(56,189,248,.3); }
.msg.err { background: rgba(248,113,113,.1); color: var(--bad); border: 1px solid rgba(248,113,113,.3); }
.msg.ok { background: rgba(52,211,153,.1); color: var(--good); border: 1px solid rgba(52,211,153,.3); }
.batch { flex: 1; padding: 14px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }
textarea { background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 8px;
  padding: 8px 10px; font-size: 12px; font-family: ui-monospace, monospace; resize: vertical; width: 100%; }
textarea:focus { outline: none; border-color: var(--accent); }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; color: var(--dim); font-size: 10.5px; text-transform: uppercase; letter-spacing: .5px;
  padding: 7px 9px; border-bottom: 1px solid var(--border); white-space: nowrap; }
td { padding: 7px 9px; border-bottom: 1px solid rgba(35,49,80,.5); font-variant-numeric: tabular-nums; }
tr:hover td { background: rgba(56,189,248,.04); }
td select { width: auto; padding: 3px 6px; font-size: 11px; }
.prog { height: 7px; background: var(--border); border-radius: 4px; overflow: hidden; flex: 1; }
.prog > div { height: 100%; background: linear-gradient(90deg, var(--violet), var(--accent)); transition: width .3s; }
.chip { display: inline-block; padding: 1px 7px; border-radius: 999px; font-size: 10.5px; font-weight: 600; }
.chip.ok { background: rgba(52,211,153,.14); color: var(--good); }
.chip.err { background: rgba(248,113,113,.14); color: var(--bad); }
.foot { font-size: 11px; color: var(--dim); text-align: center; padding: 4px 0 8px; }
.spin { display: inline-block; width: 12px; height: 12px; border: 2px solid var(--border);
  border-top-color: var(--accent); border-radius: 50%; animation: sp .7s linear infinite; vertical-align: -2px; }
@keyframes sp { to { transform: rotate(360deg); } }
`;

// ====================================================================== //
// Ana bileşen
// ====================================================================== //

export default function App() {
  // -- bağlantı & genel durum
  const [online, setOnline] = useState<boolean | null>(null);
  const [models, setModels] = useState<string[]>(FALLBACK_MODELS);
  const [tab, setTab] = useState<"viewer" | "batch">("viewer");
  const [msg, setMsg] = useState<{ kind: "info" | "err" | "ok"; text: string } | null>(null);

  // -- vaka & faz
  const [caseInfo, setCaseInfo] = useState<CaseInfo | null>(null);
  const [phaseMetrics, setPhaseMetrics] = useState<PhaseMetric[]>([]);
  const [selPhase, setSelPhase] = useState<string>("");
  const [sliceIdx, setSliceIdx] = useState(0);
  const [numSlices, setNumSlices] = useState(1);
  const [pathInput, setPathInput] = useState("");
  const [loadingSlice, setLoadingSlice] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  // -- model & parametreler
  const [modelId, setModelId] = useState("rdn");
  const [params, setParams] = useState<BlendParams>({ ...DEFAULT_PARAMS });
  const [wl, setWl] = useState(50);
  const [ww, setWw] = useState(400);
  const [sliceMeta, setSliceMeta] = useState<SliceMeta | null>(null);
  const [stats, setStats] = useState({ snr: 0, cnr: 0, sigma: 0, meanDiff: 0 });

  // -- ön ayarlar
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetName, setPresetName] = useState("");

  // -- toplu işlem
  const [batchPaths, setBatchPaths] = useState("");
  const [batchRows, setBatchRows] = useState<BatchRow[]>([]);
  const [batchBusy, setBatchBusy] = useState(false);
  const [commitBusy, setCommitBusy] = useState(false);
  const [commitResult, setCommitResult] = useState<string | null>(null);
  const [learnProgress, setLearnProgress] = useState<number | null>(null);
  const [exportState, setExportState] = useState<{
    busy: boolean; progress: number; stage: string; outputDir: string;
  }>({ busy: false, progress: 0, stage: "", outputDir: "" });

  // -- 60 FPS render hattı (state yerine ref: rAF döngüsü React'i beklemez)
  const buffersRef = useRef<SliceBuffers | null>(null);
  const outRef = useRef<Float32Array | null>(null);
  const paramsRef = useRef<BlendParams>({ ...DEFAULT_PARAMS });
  const windowRef = useRef({ wl: 50, ww: 400 });
  const rafRef = useRef(0);
  const lastStatsT = useRef(-1e9); // ilk karede istatistikler mutlaka hesaplansın
  const sliceCache = useRef(new Map<string, { buf: SliceBuffers; meta: SliceMeta }>());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const hoverRef = useRef<HTMLDivElement>(null);

  const cvOrig = useRef<HTMLCanvasElement>(null);
  const cvEnh = useRef<HTMLCanvasElement>(null);
  const cvDiff = useRef<HTMLCanvasElement>(null);
  const imgDataRef = useRef<{ orig: ImageData; enh: ImageData; diff: ImageData } | null>(null);
  const xhRefs = useRef<(HTMLDivElement | null)[]>([]);

  const flash = useCallback((kind: "info" | "err" | "ok", text: string) => {
    setMsg({ kind, text });
    window.setTimeout(() => setMsg((m) => (m?.text === text ? null : m)), 6000);
  }, []);

  // ------------------------------------------------------------------ //
  // Render çekirdeği: harmanla + çiz (tamamı istemci tarafı, <10 ms)
  // ------------------------------------------------------------------ //
  const renderFrame = useCallback(() => {
    rafRef.current = 0;
    const buf = buffersRef.current;
    if (!buf) return;
    const { w, h } = buf;
    const o = cvOrig.current, e = cvEnh.current, d = cvDiff.current;
    if (!o || !e || !d) return;
    if (o.width !== w) { [o, e, d].forEach((c) => { c.width = w; c.height = h; }); imgDataRef.current = null; }
    const octx = o.getContext("2d")!, ectx = e.getContext("2d")!, dctx = d.getContext("2d")!;
    if (!imgDataRef.current) {
      imgDataRef.current = {
        orig: octx.createImageData(w, h),
        enh: ectx.createImageData(w, h),
        diff: dctx.createImageData(w, h),
      };
    }
    if (!outRef.current || outRef.current.length !== w * h) outRef.current = new Float32Array(w * h);
    const out = outRef.current;
    const p = paramsRef.current;
    const { wl: WL, ww: WW } = windowRef.current;

    blendInto(out, buf, p);
    drawGray(octx, buf.iOrig, w, h, WL, WW, imgDataRef.current.orig);
    drawGray(ectx, out, w, h, WL, WW, imgDataRef.current.enh);
    drawThermal(dctx, out, buf.iOrig, w, h, 120, imgDataRef.current.diff);

    // SNR/CNR kartları ~6 Hz güncellenir; harmanlama 60 FPS kalır
    const now = performance.now();
    if (now - lastStatsT.current > 160) {
      lastStatsT.current = now;
      const st = computeStats(out, buf.dc, w, h);
      let sd = 0;
      for (let i = 0; i < out.length; i += 4) sd += Math.abs(out[i] - buf.iOrig[i]);
      setStats({ ...st, meanDiff: sd / Math.ceil(out.length / 4) });
    }
  }, []);

  const scheduleRender = useCallback(() => {
    if (!rafRef.current) rafRef.current = requestAnimationFrame(renderFrame);
  }, [renderFrame]);

  useEffect(() => { paramsRef.current = params; scheduleRender(); }, [params, scheduleRender]);
  useEffect(() => { windowRef.current = { wl, ww }; scheduleRender(); }, [wl, ww, scheduleRender]);

  // ------------------------------------------------------------------ //
  // Backend bağlantısı & açılış
  // ------------------------------------------------------------------ //
  useEffect(() => {
    const ctrl = new AbortController();
    const t = window.setTimeout(() => ctrl.abort(), 2500);
    fetch(`${API}/health`, { signal: ctrl.signal })
      .then((r) => r.json())
      .then((j) => {
        setOnline(true);
        if (Array.isArray(j.models) && j.models.length) setModels(j.models);
        fetchPresets(true);
      })
      .catch(() => {
        setOnline(false);
        fetchPresets(false);
        loadDemoCase("Sentetik Kardiyak Fantom");
      })
      .finally(() => window.clearTimeout(t));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // WebSocket: arka plan öğrenme ilerlemesi
  useEffect(() => {
    if (!online) return;
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(WS_URL);
      ws.onmessage = (ev) => {
        try {
          const m = JSON.parse(ev.data);
          if (m.type === "noise2sim_adaptation") {
            setLearnProgress(m.progress ?? null);
            if (m.progress >= 1) {
              flash("ok", `Noise2Sim adaptasyonu tamamlandı${m.checkpoint ? ` · ${m.checkpoint}` : ""}`);
              window.setTimeout(() => setLearnProgress(null), 4000);
            }
          } else if (m.type === "export_progress") {
            const done = (m.progress ?? 0) >= 1 || m.status === "done";
            setExportState({
              busy: !done, progress: m.progress ?? 0,
              stage: m.stage ?? "", outputDir: m.output_dir ?? "",
            });
            if (done) {
              flash("ok", `Dışa aktarma tamamlandı: ${m.output_dir}`);
            }
          }
        } catch { /* yoksay */ }
      };
    } catch { /* WS yoksa sessiz geç */ }
    return () => ws?.close();
  }, [online, flash]);

  // ------------------------------------------------------------------ //
  // Vaka yükleme
  // ------------------------------------------------------------------ //
  const loadDemoCase = useCallback((label: string) => {
    const best = DEMO_PHASES.find((p) => p.recommended)!;
    setCaseInfo({
      caseId: "demo-case",
      label,
      demo: true,
      phases: DEMO_PHASES.map((p) => ({ name: p.phase_name, numSlices: p.num_slices })),
    });
    setPhaseMetrics(DEMO_PHASES);
    setSelPhase(best.phase_name);
    setNumSlices(best.num_slices);
    setSliceIdx(Math.floor(best.num_slices / 2));
    sliceCache.current.clear();
  }, []);

  /** ingest / ingest-upload yanıtını uygular ve faz önerisini çeker. */
  const applyIngest = useCallback(async (j: {
    case_id: string;
    phases: Record<string, { num_slices: number }>;
  }, label: string) => {
    const phases = Object.entries(j.phases)
      .map(([name, p]) => ({ name, numSlices: p.num_slices }));
    setCaseInfo({ caseId: j.case_id, label, demo: false, phases });
    sliceCache.current.clear();

    const r2 = await fetch(`${API}/recommend-phase`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ case_id: j.case_id }),
    });
    if (!r2.ok) throw new Error((await r2.json()).detail || r2.statusText);
    const j2 = await r2.json();
    const metrics: PhaseMetric[] = j2.phases || [];
    setPhaseMetrics(metrics);
    const rec = j2.recommended_phase || metrics[0]?.phase_name || phases[0]?.name || "";
    setSelPhase(rec);
    const np = phases.find((p) => p.name === rec)?.numSlices ?? 1;
    setNumSlices(np);
    setSliceIdx(Math.floor(np / 2));
    flash("ok", `Vaka yüklendi: ${j.case_id} · ${phases.length} seri/faz`);
  }, [flash]);

  const ingestPath = useCallback(async (path: string) => {
    if (!path.trim()) return;
    if (!online) {
      loadDemoCase(path.split("/").pop() || path);
      flash("info", "Çevrimdışı mod: yol sentetik demo vakası olarak yüklendi.");
      return;
    }
    setLoadingSlice(true);
    try {
      const r = await fetch(`${API}/ingest`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: path.trim() }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      await applyIngest(await r.json(), path.split("/").pop() || path);
    } catch (err) {
      flash("err", `Yükleme hatası: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoadingSlice(false);
    }
  }, [online, loadDemoCase, flash, applyIngest]);

  /** Sürüklenen/seçilen dosyaların içeriğini backend'e yükler. */
  const ingestUpload = useCallback(async (picked: PickedFile[]) => {
    const files = picked.filter((p) => shouldUpload(p.rel));
    if (files.length === 0) {
      flash("err", "Yüklenebilir görüntü dosyası yok (.dcm / .nii / .nii.gz / .hdf5 / .zip / .rar).");
      return;
    }
    const label = labelOf(files);
    if (!online) {
      loadDemoCase(label);
      flash("info", `Çevrimdışı demo: "${label}" sentetik vaka olarak açıldı.`);
      return;
    }
    setLoadingSlice(true);
    try {
      // Büyük PACS vakaları (binlerce dosya) sunucunun form sınırına takılmasın
      // diye paketler halinde, aynı upload_id ile gönderilir; son paket
      // finalize=true taşır ve ingest'i tetikler.
      const CHUNK = 400;
      const uploadId = Math.random().toString(36).slice(2, 10);
      let lastResponse: unknown = null;
      for (let i = 0; i < files.length; i += CHUNK) {
        const batch = files.slice(i, i + CHUNK);
        const isLast = i + CHUNK >= files.length;
        flash("info",
          `${Math.min(i + batch.length, files.length)}/${files.length} dosya yükleniyor…`);
        const fd = new FormData();
        fd.append("case_name", label);
        fd.append("upload_id", uploadId);
        fd.append("finalize", isLast ? "true" : "false");
        for (const { file, rel } of batch) fd.append("files", file, rel);
        const r = await fetch(`${API}/ingest-upload`, { method: "POST", body: fd });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        lastResponse = await r.json();
      }
      flash("info", "Dosyalar yüklendi · fazlar analiz ediliyor (büyük vakada ~1 dk)…");
      await applyIngest(lastResponse as Parameters<typeof applyIngest>[0], label);
    } catch (err) {
      flash("err", `Yükleme hatası: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoadingSlice(false);
    }
  }, [online, loadDemoCase, flash, applyIngest]);

  /** Klasör seçici (webkitdirectory) dosyaları. */
  const handlePickedFiles = useCallback((list: FileList | null) => {
    if (!list || list.length === 0) return;
    const picked: PickedFile[] = Array.from(list).map((f) => ({
      file: f,
      rel: (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name,
    }));
    void ingestUpload(picked);
  }, [ingestUpload]);

  /** Drop olayından eşzamanlı alınan FileSystemEntry'lerle klasör ağacı yükleme. */
  const handleDroppedEntries = useCallback(async (entries: unknown[], fallback: File[]) => {
    try {
      let picked = await walkDroppedEntries(entries);
      if (picked.length === 0) {
        picked = fallback.map((f) => ({ file: f, rel: f.name }));
      }
      if (picked.length === 0) return;
      await ingestUpload(picked);
    } catch (err) {
      flash("err", `Sürükle-bırak okunamadı: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [ingestUpload, flash]);

  // ------------------------------------------------------------------ //
  // Kesit getirme (gerçek backend tamponları veya demo üretimi)
  // ------------------------------------------------------------------ //
  const fetchSlice = useCallback(async () => {
    if (!caseInfo || !selPhase) return;
    const key = `${caseInfo.caseId}|${selPhase}|${sliceIdx}|${modelId}`;
    const cached = sliceCache.current.get(key);
    if (cached) {
      buffersRef.current = cached.buf;
      setSliceMeta(cached.meta);
      scheduleRender();
      return;
    }
    if (caseInfo.demo) {
      const pIdx = Math.max(0, DEMO_PHASES.findIndex((p) => p.phase_name === selPhase));
      const buf = genDemoSlice(pIdx, sliceIdx);
      const meta: SliceMeta = {
        modelId, fallback: true, runtimeS: 0.004,
        sigmaBefore: DEMO_PHASES[pIdx].sigma_hu, sigmaAfter: 8.4,
        demo: true, autoBestCandidates: null,
      };
      sliceCache.current.set(key, { buf, meta });
      buffersRef.current = buf;
      setSliceMeta(meta);
      scheduleRender();
      return;
    }
    setLoadingSlice(true);
    try {
      const r = await fetch(`${API}/enhance-slice`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          case_id: caseInfo.caseId, phase: selPhase, slice_index: sliceIdx,
          model_id: modelId, params: paramsRef.current, include_contrast_delta: true,
        }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const j = await r.json();
      const [h, w] = j.shape as [number, number];
      const buf: SliceBuffers = {
        w, h,
        iOrig: b64ToF32(j.buffers.i_orig),
        delta: b64ToF32(j.buffers.delta),
        lap: b64ToF32(j.buffers.lap_sharpen),
        tv: b64ToF32(j.buffers.tv_map),
        dc: b64ToF32(j.buffers.delta_contrast),
        phiBiasPos: j.bias?.phi_pos ?? 0,
        phiBiasNeg: j.bias?.phi_neg ?? 0,
      };
      const meta: SliceMeta = {
        modelId: j.model_id, fallback: !!j.fallback, runtimeS: j.runtime_s,
        sigmaBefore: j.sigma_before, sigmaAfter: j.sigma_after,
        demo: false, autoBestCandidates: j.auto_best_candidates ?? null,
      };
      sliceCache.current.set(key, { buf, meta });
      buffersRef.current = buf;
      setSliceMeta(meta);
      scheduleRender();
      if (meta.fallback) flash("info", `${meta.modelId}: model ağırlığı yok → deterministik yedek çekirdek kullanıldı.`);
    } catch (err) {
      flash("err", `Kesit hatası: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoadingSlice(false);
    }
  }, [caseInfo, selPhase, sliceIdx, modelId, scheduleRender, flash]);

  useEffect(() => { fetchSlice(); }, [fetchSlice]);

  // ------------------------------------------------------------------ //
  // Fare altında HU/ΔHU okuma + senkron crosshair
  // ------------------------------------------------------------------ //
  const onCanvasMove = useCallback((ev: React.MouseEvent<HTMLCanvasElement>) => {
    const buf = buffersRef.current;
    const out = outRef.current;
    if (!buf || !out) return;
    const rect = ev.currentTarget.getBoundingClientRect();
    const fx = (ev.clientX - rect.left) / rect.width;
    const fy = (ev.clientY - rect.top) / rect.height;
    const x = Math.min(buf.w - 1, Math.max(0, Math.floor(fx * buf.w)));
    const y = Math.min(buf.h - 1, Math.max(0, Math.floor(fy * buf.h)));
    const i = y * buf.w + x;
    const hu0 = buf.iOrig[i], hu1 = out[i];
    if (hoverRef.current) {
      hoverRef.current.innerHTML =
        `<b>(${x}, ${y})</b><span>Orijinal: <b>${hu0.toFixed(1)} HU</b></span>` +
        `<span>İyileştirilmiş: <b>${hu1.toFixed(1)} HU</b></span>` +
        `<span>ΔHU: <b style="color:${Math.abs(hu1 - hu0) > 1.5 ? "#fbbf24" : "#34d399"}">${(hu1 - hu0).toFixed(2)}</b></span>`;
    }
    // üç canvas üzerinde eşzamanlı crosshair
    for (let c = 0; c < 3; c++) {
      const hEl = xhRefs.current[c * 2], vEl = xhRefs.current[c * 2 + 1];
      if (hEl) { hEl.style.display = "block"; hEl.style.top = `${fy * 100}%`; }
      if (vEl) { vEl.style.display = "block"; vEl.style.left = `${fx * 100}%`; }
    }
  }, []);

  const onCanvasLeave = useCallback(() => {
    xhRefs.current.forEach((el) => { if (el) el.style.display = "none"; });
    if (hoverRef.current) hoverRef.current.innerHTML =
      "<span>İmleci görüntü üzerinde gezdirin: piksel HU / ΔHU değerleri burada görünür.</span>";
  }, []);

  // ------------------------------------------------------------------ //
  // Otomatik en iyi & öğrenilmiş parametreler
  // ------------------------------------------------------------------ //
  const runAutoBest = useCallback(() => {
    if (!caseInfo) return;
    if (caseInfo.demo) {
      setModelId("rdn");
      setParams({ alpha: 1.2, gamma: 0.15, lambda_tv: 0.08, beta: 0.25, kappa_hu: 0.6 });
      flash("ok", "Demo: otomatik en iyi kombinasyon uygulandı (rdn + agresif harman).");
      return;
    }
    setModelId("auto_best");
  }, [caseInfo, flash]);

  const applyLearnedParams = useCallback(async () => {
    if (!caseInfo) return;
    if (caseInfo.demo || !online) {
      const best = phaseMetrics.find((p) => p.recommended) ?? phaseMetrics[0];
      const s = best ? best.sigma_hu : 28;
      setParams({
        alpha: Math.min(1.5, 0.7 + s / 60),
        gamma: 0.2, lambda_tv: Math.min(0.5, s / 250),
        beta: 0.15, kappa_hu: 0.5,
      });
      flash("info", "Demo: σ_HU'ya göre sezgisel öğrenilmiş parametre kestirimi uygulandı.");
      return;
    }
    try {
      const r = await fetch(`${API}/predict-learned-params`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_id: caseInfo.caseId }),
      });
      if (r.status === 409) {
        flash("info", "Meta-optimizer henüz eğitilmedi (en az 5 onaylı vaka gerekli).");
        return;
      }
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const j = await r.json();
      const pred = j.prediction || {};
      setParams((prev) => ({
        alpha: Number(pred.alpha ?? prev.alpha),
        gamma: Number(pred.gamma ?? prev.gamma),
        lambda_tv: Number(pred.lambda_tv ?? prev.lambda_tv),
        beta: Number(pred.beta ?? prev.beta),
        kappa_hu: Number(pred.kappa_hu ?? prev.kappa_hu),
      }));
      if (pred.model_id && models.includes(pred.model_id)) setModelId(pred.model_id);
      flash("ok", `Öğrenilmiş kombinasyon uygulandı${pred.model_id ? ` · model: ${pred.model_id}` : ""}`);
    } catch (err) {
      flash("err", `Tahmin hatası: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [caseInfo, online, phaseMetrics, models, flash]);

  // ------------------------------------------------------------------ //
  // Ön ayarlar
  // ------------------------------------------------------------------ //
  const fetchPresets = useCallback(async (isOnline: boolean) => {
    if (isOnline) {
      try {
        const r = await fetch(`${API}/presets`);
        const j = await r.json();
        setPresets(j.presets || []);
        return;
      } catch { /* yereldekine düş */ }
    }
    try {
      setPresets(JSON.parse(localStorage.getItem("ct_presets") || "[]"));
    } catch { setPresets([]); }
  }, []);

  // ------------------------------------------------------------------ //
  // Dışa aktarma: seçili faz + o anki ayarlarla tüm kesitleri klasöre yaz
  // ------------------------------------------------------------------ //
  const startExport = useCallback(async () => {
    if (!caseInfo || !selPhase) return;
    if (caseInfo.demo || !online) {
      flash("err", "Dışa aktarma için backend bağlantısı gerekir (demo modunda kullanılamaz).");
      return;
    }
    // auto_best seçiliyse o kesit için çözümlenen somut model kullanılır
    const resolved = modelId === "auto_best" ? (sliceMeta?.modelId || "rdn") : modelId;
    setExportState({ busy: true, progress: 0, stage: "başlatılıyor…", outputDir: "" });
    try {
      const r = await fetch(`${API}/export-phase`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          case_id: caseInfo.caseId, phase: selPhase,
          model_id: resolved, params,
        }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      flash("info", `Dışa aktarma başladı: ${selPhase} · ${resolved} — ilerleme aşağıda.`);
    } catch (err) {
      setExportState({ busy: false, progress: 0, stage: "", outputDir: "" });
      flash("err", `Dışa aktarma hatası: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [caseInfo, selPhase, online, modelId, sliceMeta, params, flash]);

  const savePreset = useCallback(async () => {
    const name = presetName.trim();
    if (!name) { flash("err", "Ön ayar adı girin."); return; }
    const body = {
      name, model_type: modelId === "auto_best" ? (sliceMeta?.modelId || "rdn") : modelId,
      ...params, window_center: wl, window_width: ww,
    };
    if (online) {
      try {
        const r = await fetch(`${API}/presets`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error(r.statusText);
        await fetchPresets(true);
        setPresetName("");
        flash("ok", `Ön ayar kaydedildi: ${name}`);
      } catch (err) {
        flash("err", `Ön ayar hatası: ${err instanceof Error ? err.message : String(err)}`);
      }
    } else {
      const next = presets.filter((p) => p.name !== name);
      next.unshift({ id: -Date.now(), ...body } as Preset);
      setPresets(next);
      localStorage.setItem("ct_presets", JSON.stringify(next));
      setPresetName("");
      flash("ok", `Ön ayar yerel olarak kaydedildi: ${name}`);
    }
  }, [presetName, modelId, sliceMeta, params, wl, ww, online, presets, fetchPresets, flash]);

  const deletePreset = useCallback(async (p: Preset) => {
    if (online && p.id > 0) {
      try { await fetch(`${API}/presets/${p.id}`, { method: "DELETE" }); } catch { /* yoksay */ }
      fetchPresets(true);
    } else {
      const next = presets.filter((q) => q.id !== p.id);
      setPresets(next);
      localStorage.setItem("ct_presets", JSON.stringify(next));
    }
  }, [online, presets, fetchPresets]);

  const applyPreset = useCallback((p: Preset) => {
    setParams({ alpha: p.alpha, gamma: p.gamma, lambda_tv: p.lambda_tv, beta: p.beta, kappa_hu: p.kappa_hu });
    setWl(p.window_center); setWw(p.window_width);
    if (models.includes(p.model_type)) setModelId(p.model_type);
    flash("info", `Ön ayar uygulandı: ${p.name}`);
  }, [models, flash]);

  // ------------------------------------------------------------------ //
  // Toplu işlem & sürekli öğrenme
  // ------------------------------------------------------------------ //
  const runBatchAnalyze = useCallback(async () => {
    const paths = batchPaths.split("\n").map((s) => s.trim()).filter(Boolean);
    if (paths.length === 0) { flash("err", "Analiz için en az bir yol girin (satır başına bir vaka)."); return; }
    setBatchBusy(true);
    setCommitResult(null);
    try {
      if (!online) {
        setBatchRows(paths.map((p, i) => demoBatchRow(p, i)));
        flash("info", "Çevrimdışı demo: sentetik toplu analiz sonuçları üretildi.");
        return;
      }
      const r = await fetch(`${API}/batch-analyze`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const j = await r.json();
      setBatchRows((j.rows as BatchRow[]).map((row) => ({
        ...row,
        selected: row.status === "ok",
        selected_phase: row.recommended_phase,
      })));
    } catch (err) {
      flash("err", `Toplu analiz hatası: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setBatchBusy(false);
    }
  }, [batchPaths, online, flash]);

  const commitBatch = useCallback(async () => {
    const chosen = batchRows.filter((r) => r.selected && r.status === "ok");
    if (chosen.length === 0) { flash("err", "Önce onaylanacak vakaları seçin."); return; }
    setCommitBusy(true);
    try {
      if (!online) {
        setLearnProgress(0);
        const timer = window.setInterval(() => {
          setLearnProgress((p) => {
            const np = Math.min(1, (p ?? 0) + 0.2);
            if (np >= 1) {
              window.clearInterval(timer);
              setCommitResult(
                `Demo: ${chosen.length} vaka kaydedildi · meta-optimizer güncellendi (sentetik).`);
              window.setTimeout(() => setLearnProgress(null), 2500);
            }
            return np;
          });
        }, 350);
        return;
      }
      const cases = chosen.map((r) => {
        const ph = (r.phases || []).find((p) => p.phase_name === r.selected_phase);
        const sp = r.suggested_params || {};
        const md = r.metadata || {};
        return {
          case_id: r.case_id,
          folder_path: r.path,
          selected_phase: r.selected_phase || r.recommended_phase || "",
          phase_score: ph?.s_faz ?? r.phase_score ?? 0,
          kvp: Number(md.kvp ?? 0), mas: Number(md.mas ?? 0),
          slice_thickness: Number(md.slice_thickness ?? 0),
          kernel_code: Number(md.kernel_code ?? 0),
          sigma_hu_before: ph?.sigma_hu ?? r.sigma_hu ?? 0,
          sigma_hu_after: (ph?.sigma_hu ?? r.sigma_hu ?? 0) * 0.28,
          gmm_peak_hu: ph?.mu_peak_hu ?? r.mu_peak_hu ?? 0,
          niqe_before: ph?.niqe ?? r.niqe ?? 0,
          niqe_after: (ph?.niqe ?? r.niqe ?? 0) * 0.7,
          ssim_inter: ph?.ssim_inter ?? r.ssim_inter ?? 0,
          approved_model: String(sp.model_id ?? "rdn"),
          alpha: Number(sp.alpha ?? params.alpha),
          gamma: Number(sp.gamma ?? params.gamma),
          lambda_tv: Number(sp.lambda_tv ?? params.lambda_tv),
          beta: Number(sp.beta ?? params.beta),
          kappa_hu: Number(sp.kappa_hu ?? params.kappa_hu),
        };
      });
      const r = await fetch(`${API}/batch-commit-and-learn`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cases, run_noise2sim_adaptation: true }),
      });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const j = await r.json();
      setCommitResult(
        `${j.committed} vaka kaydedildi · toplam ${j.total_cases} · ` +
        `meta-optimizer ${j.meta_optimizer_retrained ? "yeniden eğitildi ✓" : "eğitim eşiğini bekliyor"} · ` +
        `Noise2Sim adaptasyonu ${j.noise2sim_adaptation_queued ? "kuyruğa alındı" : "atlandı"}`);
      if (j.noise2sim_adaptation_queued) setLearnProgress(0);
    } catch (err) {
      flash("err", `Kaydetme hatası: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCommitBusy(false);
    }
  }, [batchRows, online, params, flash]);

  const openBatchRowInViewer = useCallback((row: BatchRow) => {
    if (row.status !== "ok" || !row.case_id) return;
    if (!online) {
      loadDemoCase(row.path.split("/").pop() || row.path);
    } else {
      const phases = (row.phases || []).map((p) => ({ name: p.phase_name, numSlices: p.num_slices }));
      setCaseInfo({ caseId: row.case_id, label: row.path.split("/").pop() || row.case_id, demo: false, phases });
      setPhaseMetrics(row.phases || []);
      const rec = row.recommended_phase || phases[0]?.name || "";
      setSelPhase(rec);
      const np = phases.find((p) => p.name === rec)?.numSlices ?? 1;
      setNumSlices(np);
      setSliceIdx(Math.floor(np / 2));
      sliceCache.current.clear();
    }
    setTab("viewer");
  }, [online, loadDemoCase]);

  // ------------------------------------------------------------------ //
  // Türetilmiş değerler
  // ------------------------------------------------------------------ //
  const recommendedPhase = useMemo(
    () => phaseMetrics.find((p) => p.recommended)?.phase_name ?? null,
    [phaseMetrics]);
  const maxSFaz = useMemo(
    () => Math.max(1, ...phaseMetrics.map((p) => p.s_faz)), [phaseMetrics]);
  const selectedBatchCount = batchRows.filter((r) => r.selected && r.status === "ok").length;

  const selectPhase = useCallback((name: string) => {
    setSelPhase(name);
    const np = caseInfo?.phases.find((p) => p.name === name)?.numSlices ?? 1;
    setNumSlices(np);
    setSliceIdx((i) => Math.min(i, np - 1));
  }, [caseInfo]);

  // ------------------------------------------------------------------ //
  // Görünüm
  // ------------------------------------------------------------------ //
  return (
    <div className="app">
      <style>{CSS}</style>

      <header className="hdr">
        <h1>CT <span>Clinical Enhancer</span></h1>
        {online === null ? (
          <span className="pill off"><span className="spin" /> bağlanıyor…</span>
        ) : online ? (
          <span className="pill on">● Backend çevrimiçi</span>
        ) : (
          <span className="pill off">◌ Çevrimdışı · Sentetik Demo Modu</span>
        )}
        {sliceMeta?.fallback && !sliceMeta.demo && <span className="fb">DETERMİNİSTİK YEDEK</span>}
        <nav className="tabs">
          <button className={tab === "viewer" ? "act" : ""} onClick={() => setTab("viewer")}>
            Vaka Görüntüleyici
          </button>
          <button className={tab === "batch" ? "act" : ""} onClick={() => setTab("batch")}>
            Toplu İşlem &amp; Öğrenme
          </button>
        </nav>
      </header>

      {tab === "viewer" ? (
        <div className="main">
          {/* ------------------- SOL PANEL ------------------- */}
          <aside className="col left">
            <section>
              <h2 className="sec-t">Vaka Yükle</h2>
              <div
                className={`drop${dragOver ? " over" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault(); setDragOver(false);
                  // webkitGetAsEntry drop olayı içinde EŞZAMANLI alınmalıdır
                  const entries = Array.from(e.dataTransfer.items)
                    .map((it) => (it as DataTransferItem & {
                      webkitGetAsEntry?: () => unknown;
                    }).webkitGetAsEntry?.())
                    .filter(Boolean) as unknown[];
                  const fallback = Array.from(e.dataTransfer.files);
                  void handleDroppedEntries(entries, fallback);
                }}
              >
                <span className="big">⊕</span>
                DICOM klasörü / .nii / .nii.gz / .hdf5 / .zip / .rar<br />
                sürükleyip bırakın ya da tıklayın — dosyalar sunucuya yüklenir
              </div>
              <input
                ref={fileInputRef} type="file" multiple style={{ display: "none" }}
                // @ts-expect-error: standart dışı klasör seçimi
                webkitdirectory=""
                onChange={(e) => { handlePickedFiles(e.target.files); e.target.value = ""; }}
              />
              <div className="pathrow">
                <input
                  type="text" placeholder="Sunucu yolu (örn. /veri/vaka01.zip)"
                  value={pathInput}
                  onChange={(e) => setPathInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && ingestPath(pathInput)}
                />
                <button className="btn" onClick={() => ingestPath(pathInput)} disabled={loadingSlice}>
                  Yükle
                </button>
              </div>
              {!online && (
                <button className="btn" style={{ width: "100%", marginTop: 8 }}
                  onClick={() => loadDemoCase("Sentetik Kardiyak Fantom")}>
                  ▶ Demo vakayı aç
                </button>
              )}
            </section>

            <section style={{ flex: 1 }}>
              <h2 className="sec-t">
                Fazlar {caseInfo ? `· ${caseInfo.label}` : ""}
              </h2>
              {phaseMetrics.length === 0 && (
                <div style={{ color: "var(--dim)", fontSize: 12 }}>
                  Henüz vaka yok. Bir vaka yükleyin{!online ? " veya demo vakayı açın" : ""}.
                </div>
              )}
              {phaseMetrics.map((p) => (
                <div
                  key={p.phase_name}
                  className={`phase${selPhase === p.phase_name ? " sel" : ""}`}
                  onClick={() => selectPhase(p.phase_name)}
                >
                  <div className="nm">
                    {p.phase_name}
                    {p.phase_name === recommendedPhase && (
                      <span className="badge-best">★ ÖNERİLEN EN İYİ FAZ</span>
                    )}
                  </div>
                  <span className="score">{p.s_faz.toFixed(1)}</span>
                  <div className="met">
                    <span>σ_HU {p.sigma_hu.toFixed(1)}</span>
                    <span>NIQE {p.niqe.toFixed(2)}</span>
                    <span>SSIM {p.ssim_inter.toFixed(3)}</span>
                  </div>
                  <div className="sbar"><div style={{ width: `${(p.s_faz / maxSFaz) * 100}%` }} /></div>
                </div>
              ))}
              {caseInfo && caseInfo.phases.length > 0 && phaseMetrics.length === 0 &&
                caseInfo.phases.map((p) => (
                  <div key={p.name}
                    className={`phase${selPhase === p.name ? " sel" : ""}`}
                    onClick={() => selectPhase(p.name)}>
                    <div className="nm">{p.name}</div>
                    <div className="met"><span>{p.numSlices} kesit</span></div>
                  </div>
                ))}
            </section>
          </aside>

          {/* ------------------- ORTA PANEL ------------------- */}
          <div className="center">
            <div className="canvases">
              {[
                { label: "Orijinal", ref: cvOrig, extra: null as React.ReactNode },
                {
                  label: "İyileştirilmiş", ref: cvEnh,
                  extra: sliceMeta && (
                    <span style={{ color: "var(--dim)", fontWeight: 400 }}>
                      {MODEL_LABELS[sliceMeta.modelId] || sliceMeta.modelId}
                      {sliceMeta.fallback ? " · yedek" : ""} · {(sliceMeta.runtimeS * 1000).toFixed(0)} ms
                    </span>
                  ),
                },
                { label: "|ΔI| Termal Fark Haritası", ref: cvDiff, extra: null as React.ReactNode },
              ].map((cv, idx) => (
                <div className="cv-wrap" key={cv.label}>
                  <div className="cap">{cv.label}{cv.extra}</div>
                  <div ref={(el) => { xhRefs.current[idx * 2] = el; }} className="xh h" />
                  <div ref={(el) => { xhRefs.current[idx * 2 + 1] = el; }} className="xh v" />
                  <canvas
                    ref={cv.ref} width={DEMO_SIZE} height={DEMO_SIZE}
                    onMouseMove={onCanvasMove} onMouseLeave={onCanvasLeave}
                  />
                </div>
              ))}
            </div>

            <div className="cards">
              <div className="card">
                <div className="k">SNR</div>
                <div className="v">{stats.snr.toFixed(2)}</div>
              </div>
              <div className="card">
                <div className="k">CNR</div>
                <div className="v">{stats.cnr.toFixed(2)}</div>
              </div>
              <div className="card">
                <div className="k">σ_HU (çıkış)</div>
                <div className={`v ${stats.sigma <= 10 ? "g" : "n"}`}>{stats.sigma.toFixed(1)}</div>
              </div>
              <div className="card">
                <div className="k">σ_HU model önce→sonra</div>
                <div className="v n">
                  {sliceMeta ? `${sliceMeta.sigmaBefore.toFixed(1)} → ${sliceMeta.sigmaAfter.toFixed(1)}` : "—"}
                </div>
              </div>
              <div className="card">
                <div className="k">Ort. |ΔHU|</div>
                <div className="v n">{stats.meanDiff.toFixed(1)}</div>
              </div>
            </div>

            <div className="readout" ref={hoverRef}>
              <span>İmleci görüntü üzerinde gezdirin: piksel HU / ΔHU değerleri burada görünür.</span>
            </div>

            <div className="readout" style={{ gap: 10 }}>
              <span style={{ whiteSpace: "nowrap" }}>Kesit <b>{sliceIdx + 1}</b> / {numSlices}</span>
              <input
                type="range" min={0} max={Math.max(0, numSlices - 1)} step={1} value={sliceIdx}
                onChange={(e) => setSliceIdx(Number(e.target.value))}
                style={{ flex: 1 }}
              />
              {loadingSlice && <span className="spin" />}
            </div>

            {sliceMeta?.autoBestCandidates && (
              <div className="msg info">
                Otomatik seçim adayları:{" "}
                {sliceMeta.autoBestCandidates.map((c) =>
                  `${c.model_id}${c.sigma_after != null ? ` (σ ${Number(c.sigma_after).toFixed(1)})` : ""}`
                ).join(" · ")}
                {" → seçilen: "}<b>{sliceMeta.modelId}</b>
              </div>
            )}
            {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
          </div>

          {/* ------------------- SAĞ PANEL ------------------- */}
          <aside className="col right">
            <section>
              <h2 className="sec-t">Model</h2>
              <select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                {models.map((m) => (
                  <option key={m} value={m}>{MODEL_LABELS[m] || m}</option>
                ))}
                <option value="auto_best">{MODEL_LABELS.auto_best}</option>
              </select>
              <div style={{ display: "flex", flexDirection: "column", gap: 7, marginTop: 10 }}>
                <button className="btn primary" onClick={runAutoBest} disabled={!caseInfo || loadingSlice}>
                  ✦ Otomatik En İyi Görüntüyü Çıkar
                </button>
                <button className="btn violet" onClick={applyLearnedParams} disabled={!caseInfo}>
                  ◎ Öğrenilmiş Parametre Kombinasyonunu Uygula
                </button>
              </div>
            </section>

            <section>
              <h2 className="sec-t">Harmanlama Parametreleri · 60 FPS</h2>
              {PARAM_DEFS.map((def) => (
                <div className="slider-row" key={def.key}>
                  <div className="lr">
                    <label>{def.label}</label>
                    <span className="val">{params[def.key].toFixed(def.step < 0.01 ? 3 : 2)}</span>
                  </div>
                  <input
                    type="range" min={def.min} max={def.max} step={def.step}
                    value={params[def.key]}
                    onChange={(e) =>
                      setParams((p) => ({ ...p, [def.key]: Number(e.target.value) }))}
                  />
                  <div className="hint">{def.hint}</div>
                </div>
              ))}

              <div className="slider-row">
                <div className="lr">
                  <label>Pencere / Seviye</label>
                  <span className="val">WL {wl} · WW {ww}</span>
                </div>
                <input type="range" min={-1000} max={1000} step={5} value={wl}
                  onChange={(e) => setWl(Number(e.target.value))} />
                <input type="range" min={10} max={3000} step={10} value={ww}
                  onChange={(e) => setWw(Number(e.target.value))} />
                <div className="wbtns">
                  {WINDOW_PRESETS.map((p) => (
                    <button key={p.name} onClick={() => { setWl(p.wl); setWw(p.ww); }}>
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>

              <button className="btn" style={{ width: "100%" }}
                onClick={() => setParams({ ...DEFAULT_PARAMS })}>
                Parametreleri sıfırla
              </button>
            </section>

            <section>
              <h2 className="sec-t">Dışa Aktar</h2>
              <button className="btn" style={{ width: "100%" }}
                disabled={!caseInfo || caseInfo.demo || !online || exportState.busy || !selPhase}
                onClick={() => void startExport()}>
                💾 İyileştirilmiş Fazı Klasöre Aktar
              </button>
              <div style={{ color: "var(--dim)", fontSize: 11, marginTop: 6 }}>
                Seçili faz ({selPhase || "—"}) tüm kesitleriyle, <b>o anki model ve
                kaydırıcı ayarlarıyla</b> işlenir. Her tıklama ayrı bir klasör üretir;
                klasör adı ayarları içerir (örn. IYILESTIRILMIS_…_a1.00_g0.25_…).
                Yol ile yüklenen vakalarda orijinal klasörün içine, sürükle-bırakla
                yüklenenlerde proje içindeki <b>disa_aktarimlar/</b> klasörüne yazılır
                (tarayıcı, sürüklenen klasörün gerçek yolunu vermez).
              </div>
              {exportState.busy && (
                <div style={{ marginTop: 8, fontSize: 12 }}>
                  <div style={{ color: "var(--accent)" }}>
                    {Math.round(exportState.progress * 100)}% · {exportState.stage}
                  </div>
                  <div style={{ background: "var(--border)", borderRadius: 4, height: 6, marginTop: 4 }}>
                    <div style={{
                      width: `${Math.round(exportState.progress * 100)}%`,
                      background: "var(--accent)", height: 6, borderRadius: 4,
                      transition: "width .3s",
                    }} />
                  </div>
                </div>
              )}
              {!exportState.busy && exportState.outputDir && (
                <div style={{ color: "var(--ok, #4ade80)", fontSize: 11.5, marginTop: 6,
                              wordBreak: "break-all" }}>
                  ✓ Yazıldı: {exportState.outputDir}
                </div>
              )}
            </section>

            <section style={{ flex: 1 }}>
              <h2 className="sec-t">Ön Ayarlar</h2>
              <div className="pathrow" style={{ marginTop: 0, marginBottom: 9 }}>
                <input type="text" placeholder="Ön ayar adı…" value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && savePreset()} />
                <button className="btn" onClick={savePreset}>Kaydet</button>
              </div>
              {presets.length === 0 && (
                <div style={{ color: "var(--dim)", fontSize: 11.5 }}>Kayıtlı ön ayar yok.</div>
              )}
              {presets.map((p) => (
                <div className="preset" key={p.id} onClick={() => applyPreset(p)}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="pn">{p.name}</div>
                    <div className="pm">
                      {p.model_type} · α{p.alpha.toFixed(2)} γ{p.gamma.toFixed(2)} λ{p.lambda_tv.toFixed(2)}{" "}
                      β{p.beta.toFixed(2)} κ{p.kappa_hu.toFixed(2)}
                    </div>
                  </div>
                  <button className="del" title="Sil"
                    onClick={(e) => { e.stopPropagation(); deletePreset(p); }}>
                    ✕
                  </button>
                </div>
              ))}
            </section>
          </aside>
        </div>
      ) : (
        /* ------------------- TOPLU İŞLEM SEKMESİ ------------------- */
        <div className="batch">
          <section className="col">
            <h2 className="sec-t">Toplu Vaka Analizi</h2>
            <div style={{ color: "var(--dim)", fontSize: 12, marginBottom: 7 }}>
              Satır başına bir vaka yolu girin (DICOM klasörü, .nii, .zip …). Analiz; her vaka için
              önerilen fazı, S_faz skorunu ve öğrenilmiş parametre önerisini üretir.
            </div>
            <textarea
              rows={4} value={batchPaths}
              placeholder={"/veri/vaka01\n/veri/vaka02.zip\n/veri/vaka03.nii.gz"}
              onChange={(e) => setBatchPaths(e.target.value)}
            />
            <div style={{ display: "flex", gap: 8, marginTop: 9, alignItems: "center" }}>
              <button className="btn primary" onClick={runBatchAnalyze} disabled={batchBusy}>
                {batchBusy ? <>Analiz ediliyor… <span className="spin" /></> : "Vakaları Analiz Et"}
              </button>
              {!online && (
                <span style={{ color: "var(--warn)", fontSize: 11.5 }}>
                  Çevrimdışı mod: sonuçlar sentetik olarak üretilir.
                </span>
              )}
            </div>
          </section>

          {batchRows.length > 0 && (
            <section className="col">
              <h2 className="sec-t">Analiz Sonuçları · {batchRows.length} vaka</h2>
              <div style={{ overflowX: "auto" }}>
                <table>
                  <thead>
                    <tr>
                      <th>
                        <input type="checkbox"
                          checked={batchRows.every((r) => r.selected || r.status !== "ok")}
                          onChange={(e) =>
                            setBatchRows((rows) => rows.map((r) =>
                              r.status === "ok" ? { ...r, selected: e.target.checked } : r))}
                        />
                      </th>
                      <th>Vaka</th><th>Durum</th><th>Seçili Faz</th><th>S_faz</th>
                      <th>σ_HU</th><th>NIQE</th><th>SSIM</th><th>Önerilen Model</th><th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {batchRows.map((r, i) => (
                      <tr key={r.path + i}>
                        <td>
                          <input type="checkbox" checked={!!r.selected} disabled={r.status !== "ok"}
                            onChange={(e) =>
                              setBatchRows((rows) => rows.map((q, qi) =>
                                qi === i ? { ...q, selected: e.target.checked } : q))}
                          />
                        </td>
                        <td title={r.path} style={{ maxWidth: 220, overflow: "hidden",
                          textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {r.path.split("/").pop() || r.path}
                        </td>
                        <td>
                          {r.status === "ok"
                            ? <span className="chip ok">tamam</span>
                            : <span className="chip err" title={r.error}>hata</span>}
                        </td>
                        <td>
                          {r.status === "ok" ? (
                            <select value={r.selected_phase || ""}
                              onChange={(e) =>
                                setBatchRows((rows) => rows.map((q, qi) =>
                                  qi === i ? { ...q, selected_phase: e.target.value } : q))}>
                              {(r.phases || []).map((p) => (
                                <option key={p.phase_name} value={p.phase_name}>
                                  {p.phase_name}{p.phase_name === r.recommended_phase ? " ★" : ""}
                                </option>
                              ))}
                            </select>
                          ) : (r.error || "—")}
                        </td>
                        <td>{r.phase_score?.toFixed(1) ?? "—"}</td>
                        <td>{r.sigma_hu?.toFixed(1) ?? "—"}</td>
                        <td>{r.niqe?.toFixed(2) ?? "—"}</td>
                        <td>{r.ssim_inter?.toFixed(3) ?? "—"}</td>
                        <td>{String(r.suggested_params?.model_id ?? "—")}</td>
                        <td>
                          {r.status === "ok" && (
                            <button className="btn" style={{ fontSize: 11, padding: "3px 9px" }}
                              onClick={() => openBatchRowInViewer(r)}>
                              Görüntüle
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 13 }}>
                <button className="btn violet" onClick={commitBatch}
                  disabled={commitBusy || selectedBatchCount === 0}>
                  {commitBusy
                    ? <>Kaydediliyor… <span className="spin" /></>
                    : `Seçimleri Uygula ve Arka Plan Öğrenmesini Güncelle (${selectedBatchCount})`}
                </button>
                {learnProgress !== null && (
                  <>
                    <span style={{ fontSize: 11.5, color: "var(--violet)", whiteSpace: "nowrap" }}>
                      Noise2Sim adaptasyonu %{Math.round(learnProgress * 100)}
                    </span>
                    <div className="prog"><div style={{ width: `${learnProgress * 100}%` }} /></div>
                  </>
                )}
              </div>
              {commitResult && <div className="msg ok">{commitResult}</div>}
              {msg && tab === "batch" && <div className={`msg ${msg.kind}`}>{msg.text}</div>}
            </section>
          )}
        </div>
      )}

      <footer className="foot">
        I_out = clip(I_orig + α·φ_γ(ΔI) + β·∇_lap(Î_DL) − λ_TV·TV(ΔI) + κ_HU·ΔI_contrast, −1024, 3071)
        · harmanlama tamamen istemci tarafında (60 FPS), model yeniden çalıştırılmaz
      </footer>
    </div>
  );
}
