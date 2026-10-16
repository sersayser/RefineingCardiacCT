# CT Clinical Enhancer — Proje Anayasası

BT (Bilgisayarlı Tomografi) kesitlerinde gürültü giderme, hareket artefaktı düzeltme,
kontrast iyileştirme, otomatik faz seçimi ve sürekli öğrenme platformu.
7 doğrulanmış açık kaynak klinik repoya dayanır (`third_party/` altında klonludur).

## Mühendislik Kuralları (asla ihlal etme)

1. **Çift katmanlı çıkarım:** Kesit bazlı modeller (RDN ~0.255 sn/kesit, ContrastGAN
   <0.5 sn/kesit, Noise2Sim ~0.3 sn/kesit) senkron API'de; NAB-GS 3B süper çözünürlük
   (~900 sn/hacim) daima FastAPI BackgroundTasks + WebSocket asenkron kuyruğunda çalışır.
2. **HU sadakati:** Tüm tensörler float32 Hounsfield Unit uzayındadır. DICOM okurken
   `RescaleSlope * pixel + RescaleIntercept` uygulanır. Çıkışlar `[-1024, 3071]` HU'ya
   kırpılır (NAB-GS hacimsel işlemde `[-512, 3071]`). Gürültü giderme sonrası homojen
   ROI ortalaması ±1.5 HU'dan fazla sapamaz; yalnız σ_HU düşer.
3. **Artık alan (residual) mimarisi:** Modeller her parametre değişiminde yeniden
   ÇALIŞTIRILMAZ. `ΔI = Î_DL − I_orig` bir kez hesaplanıp önbelleğe alınır; α, γ,
   λ_TV, β, κ_HU kaydırıcıları `residual_blender.py` içindeki vektörize denklemle
   (<10 ms) harmanlanır. Ön yüz aynı denklemi Canvas üzerinde 60 FPS uygular.
4. **Deterministik yedek (fallback):** Model ağırlığı diskte yoksa hiçbir sarmalayıcı
   çökmez; bilateral / non-local means + vasküler eşikleme tabanlı yedek çekirdek aynı
   ΔI istatistiklerini (σ_HU ≈ 30 → 8.4) üretir ve çıktı `fallback=True` ile işaretlenir.
5. **Uydurma kütüphane yok:** Yalnızca `requirements.txt` içindeki paketler ve
   `third_party/` repolarındaki gerçek modül yolları kullanılır. Ağır/opsiyonel paketler
   (pyiqa, torch ağırlıkları, rarfile) import hatasında zarifçe devre dışı kalır.

## Dizin Haritası

- `backend/config.py` — HU sınırları, ağırlık yolları, parametre aralıkları
- `backend/core/ingestion.py` — `.dcm/.nii/.nii.gz/.hdf5/.zip/.rar` okuyucu (CaseIngestor)
- `backend/core/phase_selector.py` — GMM + σ_HU + pyiqa(PI) + SSIM/MVF faz skoru (S_faz)
- `backend/core/residual_blender.py` — φ_γ PReLU artık harmanlama motoru (denklem 2.1)
- `backend/engines/` — BaseCTEnhancer + 6 sarmalayıcı (rdn, noise2sim, sr_benchmark,
  contrast_gan, temporal_mar, nab_gs_3d)
- `backend/learning/models_db.py` — SQLAlchemy: presets, processed_cases, batch_jobs
- `backend/learning/meta_optimizer.py` — MultiOutputRegressor(HistGradientBoostingRegressor)
- `backend/api/main.py` — FastAPI REST + WebSocket
- `database/clinical_learning.sqlite` — kalıcı DB (git'e girmez)
- `third_party/` — 7 referans repo (salt okunur; asla düzenleme)

## Komutlar

- Testler: `.venv/bin/python -m pytest tests/ -x -q`
- API sunucusu: `.venv/bin/uvicorn backend.api.main:app --reload --port 8000`

## Formüller (tek doğruluk kaynağı: backend/core/residual_blender.py)

- `φ_γ(ΔI) = ΔI (ΔI≥0) | γ·ΔI (ΔI<0)`
- `I_out = clip(I_orig + α·φ_γ(ΔI) + β·∇_lap(Î_DL) − λ_TV·TV(ΔI) + κ_HU·ΔI_contrast, −1024, 3071)`
- `PI = ((10 − NRQM) + NIQE) / 2`
- `S_faz = 100·(0.30·N_k + 0.25·C_k + 0.25·M_k + 0.20·Q_k)`
