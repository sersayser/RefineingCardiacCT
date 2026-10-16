# CT Clinical Enhancer

BT kesitlerinde gürültü giderme, hareket artefaktı düzeltme, kontrast iyileştirme,
otomatik faz seçimi ve sürekli öğrenme platformu. 7 doğrulanmış açık kaynak klinik
repoya dayanır (mimari kurallar için [CLAUDE.md](CLAUDE.md)).

## Kurulum

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Referans repolar `third_party/` altına klonlanmıştır (salt okunur).

## Çalıştırma

```bash
.venv/bin/uvicorn backend.api.main:app --reload --port 8000
```

- API dokümantasyonu: http://localhost:8000/docs
- İş ilerleme yayını (NAB-GS 3B, Noise2Sim adaptasyonu): `ws://localhost:8000/api/v1/ws/jobs`

## Testler

```bash
.venv/bin/python -m pytest tests/ -q
```

34 test; klinik doğrulama protokolünün 4 kriterini kapsar (HU sadakati ±1.5 HU,
<10 ms harmanlama, çok fazlı arşivde doğru faz önerisi, kapalı döngü öğrenme).

## Model ağırlıkları

`checkpoints/` altına konur (yollar `backend/config.py` içinde). Ağırlık yokken
tüm motorlar deterministik yedek çekirdekle çalışır ve yanıtlar `fallback: true`
ile işaretlenir:

| Model | Ağırlık dosyası | Yedek çekirdek |
| --- | --- | --- |
| RDN | `rdn_cardiac_ct.pth` (state_dict) | bilateral, sigma 30→8.4 hedefli |
| Noise2Sim unet2 | `noise2sim_clinic_adapted.pth` | bilateral + hızlı NLM |
| ContrastGAN | `contrast_gan.pth` (TorchScript) | vasküler eşikleme ΔI haritası |
| TT-U-Net 4D | `tt_unet_4d.pth` (TorchScript) | optik akış telafili zamansal ortalama |
| NAB-GS 3B | `nab_gs_3d.pth` (TorchScript) | 3. derece spline 4x + keskinleştirme |
| SR havuzu (SRCNN/Real-ESRGAN/SwinIR/HAT/DAT) | `<model>.pth` (spandrel ile) | varyanta özgü denoise+sharpen profili |

## Ön yüz

`frontend/src/App.tsx` tek dosyalı klinik arayüz içindir (plan dokümanı bölüm 5).
Arka uç `http://localhost:8000/api/v1/...` sözleşmesini sunar; `enhance-slice`
yanıtındaki Base64 float32 tamponları (`i_orig`, `delta`, `lap_sharpen`,
`tv_map`, `delta_contrast`) ile 60 FPS kaydırıcı harmanlaması tamamen istemci
tarafında yapılır.
