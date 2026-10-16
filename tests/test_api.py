"""FastAPI uçtan uca testleri + kapalı döngü sürekli öğrenme (protokol 4)."""

import base64
import os

import numpy as np
import pytest

# DB/uploads/exports yolları conftest.py içinde (app import edilmeden önce)
# geçici klasörlere yönlendirilir.
from fastapi.testclient import TestClient  # noqa: E402

from backend.api import main as api_main  # noqa: E402

client = TestClient(api_main.app)


def _decode_f32(b64: str, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").reshape(shape)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "rdn" in body["models"]


def test_predict_before_training_returns_409():
    r = client.post("/api/v1/predict-learned-params",
                    json={"features": [120, 300, 0.6, 30, 30.0, 350.0, 6.0, 0.9]})
    assert r.status_code == 409


def test_ingest_recommend_enhance_flow(multiphase_case_dir):
    # 1) Yutma
    r = client.post("/api/v1/ingest", json={"path": multiphase_case_dir})
    assert r.status_code == 200, r.text
    body = r.json()
    case_id = body["case_id"]
    assert body["is_multiphase"]
    assert len(body["phases"]) == 4
    assert body["metadata"]["kvp"] == 120.0

    # 2) Faz önerisi
    r = client.post("/api/v1/recommend-phase", json={"case_id": case_id})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["recommended_phase"] == "phase_75"
    assert len(rec["phases"]) == 4

    # 3) Kesit iyileştirme + harmanlama önbelleği
    r = client.post("/api/v1/enhance-slice", json={
        "case_id": case_id,
        "phase": "phase_30",
        "slice_index": 2,
        "model_id": "rdn",
        "params": {"alpha": 1.0, "gamma": 0.25},
    })
    assert r.status_code == 200, r.text
    enh = r.json()
    assert enh["model_id"] == "rdn"
    assert enh["fallback"] is True
    h, w = enh["shape"]
    assert (h, w) == (128, 128)

    buffers = enh["buffers"]
    for key in ("i_orig", "delta", "lap_sharpen", "tv_map", "delta_contrast"):
        arr = _decode_f32(buffers[key], (h, w))
        assert arr.dtype == np.float32
    i_orig = _decode_f32(buffers["i_orig"], (h, w))
    delta = _decode_f32(buffers["delta"], (h, w))
    # Gürültü giderici artık, sıfırdan farklı olmalı
    assert float(np.mean(np.abs(delta))) > 0.1
    assert enh["sigma_after"] < enh["sigma_before"]
    assert float(i_orig.min()) >= -1024.0


def test_enhance_slice_auto_best(multiphase_case_dir):
    client.post("/api/v1/ingest", json={"path": multiphase_case_dir})
    r = client.post("/api/v1/enhance-slice", json={
        "case_id": "case_demo",
        "phase": "phase_75",
        "slice_index": 1,
        "model_id": "auto_best",
        "include_contrast_delta": False,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["auto_best_candidates"] is not None
    assert len(body["auto_best_candidates"]) == 5
    assert body["model_id"] in {"rdn", "noise2sim", "swinir", "hat", "real_esrgan"}


def test_ingest_upload_multipart(multiphase_case_dir):
    """Tarayıcı sürükle-bırak akışı: dosya içeriği multipart ile yüklenir."""
    files = []
    for phase in ("phase_30", "phase_75"):
        pdir = os.path.join(multiphase_case_dir, phase)
        for fn in sorted(os.listdir(pdir)):
            with open(os.path.join(pdir, fn), "rb") as fh:
                files.append(("files", (f"vaka_x/{phase}/{fn}", fh.read(),
                                        "application/dicom")))

    r = client.post("/api/v1/ingest-upload", files=files,
                    data={"case_name": "vaka_x"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["case_id"] == "vaka_x"
    assert body["uploaded_files"] == 10  # 2 faz x 5 kesit
    assert set(body["phases"]) == {"phase_30", "phase_75"}
    assert body["phases"]["phase_75"]["num_slices"] == 5

    # Yüklenen vaka bellekte: faz önerisi doğrudan çalışmalı
    r2 = client.post("/api/v1/recommend-phase", json={"case_id": "vaka_x"})
    assert r2.status_code == 200
    assert r2.json()["recommended_phase"] in {"phase_30", "phase_75"}


def test_ingest_upload_chunked(multiphase_case_dir):
    """Parçalı yükleme: aynı upload_id ile 2 paket; ingest son pakette çalışır."""
    def batch_for(phase: str):
        pdir = os.path.join(multiphase_case_dir, phase)
        out = []
        for fn in sorted(os.listdir(pdir)):
            with open(os.path.join(pdir, fn), "rb") as fh:
                out.append(("files", (f"vaka_p/{phase}/{fn}", fh.read(),
                                      "application/dicom")))
        return out

    r1 = client.post("/api/v1/ingest-upload", files=batch_for("phase_30"),
                     data={"case_name": "vaka_p", "upload_id": "parca01",
                           "finalize": "false"})
    assert r1.status_code == 200, r1.text
    assert r1.json() == {"upload_id": "parca01", "saved": 5, "finalized": False}

    r2 = client.post("/api/v1/ingest-upload", files=batch_for("phase_75"),
                     data={"case_name": "vaka_p", "upload_id": "parca01",
                           "finalize": "true"})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["finalized"] is True
    # İki paketin dosyaları aynı klasörde birleşti: her iki faz da görünmeli
    assert set(body["phases"]) == {"phase_30", "phase_75"}


def test_ingest_upload_rejects_traversal_only():
    r = client.post("/api/v1/ingest-upload",
                    files=[("files", ("../../etc/kotu.dcm", b"x", "application/dicom"))],
                    data={"case_name": "kotu"})
    # Yol temizlenir; geçersiz DICOM içeriği 422 ile reddedilir, dışarı yazılmaz
    assert r.status_code == 422


def test_presets_crud():
    r = client.post("/api/v1/presets", json={
        "name": "Koroner Standart", "model_type": "rdn",
        "alpha": 1.15, "gamma": 0.25, "window_center": 300, "window_width": 800,
    })
    assert r.status_code == 200, r.text
    preset_id = r.json()["id"]

    r = client.get("/api/v1/presets")
    names = [p["name"] for p in r.json()["presets"]]
    assert "Koroner Standart" in names

    r = client.delete(f"/api/v1/presets/{preset_id}")
    assert r.status_code == 200
    r = client.get("/api/v1/presets")
    assert "Koroner Standart" not in [p["name"] for p in r.json()["presets"]]


def test_closed_loop_learning():
    """Protokol 4: yüksek gürültülü 10 vakada alpha=1.15, gamma=0.25 onayı
    sonrası benzer profilli yeni vaka için öğrenilmiş değerler dönmeli."""
    rng = np.random.default_rng(42)
    cases = []
    for i in range(10):
        cases.append({
            "case_id": f"noisy_case_{i}",
            "selected_phase": "phase_75",
            "kvp": 120.0, "mas": float(280 + rng.integers(0, 40)),
            "slice_thickness": 0.6, "kernel_code": 30,
            "sigma_hu_before": float(28.0 + rng.uniform(0, 5)),  # > 28 HU
            "sigma_hu_after": 8.4,
            "gmm_peak_hu": float(330 + rng.uniform(0, 60)),
            "niqe_before": float(5.5 + rng.uniform(0, 1.5)),
            "ssim_inter": float(0.85 + rng.uniform(0, 0.1)),
            "approved_model": "rdn",
            "alpha": 1.15, "gamma": 0.25,
            "lambda_tv": 0.1, "beta": 0.2, "kappa_hu": 0.5,
        })

    r = client.post("/api/v1/batch-commit-and-learn", json={
        "cases": cases,
        "run_noise2sim_adaptation": False,  # testte ağır eğitim kuyruğu kapalı
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["committed"] == 10
    assert body["meta_optimizer_retrained"] is True

    # Benzer gürültü profiline sahip yeni vaka
    r = client.post("/api/v1/predict-learned-params", json={
        "features": [120.0, 300.0, 0.6, 30.0, 30.5, 350.0, 6.0, 0.9],
    })
    assert r.status_code == 200, r.text
    pred = r.json()["prediction"]
    assert pred["model_id"] == "rdn"
    assert abs(pred["alpha"] - 1.15) < 0.15
    assert abs(pred["gamma"] - 0.25) < 0.10


def test_export_phase_writes_dicom_into_case_folder(multiphase_case_dir):
    """Dışa aktarma: seçili faz, o anki ayarlarla orijinal klasör içine yazılır."""
    import glob

    import pydicom

    client.post("/api/v1/ingest", json={"path": multiphase_case_dir})
    # Saf gürültü giderme ayarları: HU sadakati iddiası ancak böyle anlamlı
    # (beta kenar keskinleştirme, tanımı gereği kenarlarda HU kaydırır)
    r = client.post("/api/v1/export-phase", json={
        "case_id": "case_demo", "phase": "phase_75", "model_id": "rdn",
        "params": {"alpha": 1.0, "gamma": 0.25},
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # TestClient arka plan görevini yanıt sonrası senkron çalıştırır
    r = client.get(f"/api/v1/export/status/{job_id}")
    status = r.json()
    assert status["status"] == "done", status
    out_dir = status["output_dir"]
    # Çıktı klasörü orijinal vaka klasörünün İÇİNDE olmalı; adı ayarları taşır
    assert out_dir.startswith(multiphase_case_dir)
    assert os.path.basename(out_dir).startswith(
        "IYILESTIRILMIS_phase_75_rdn_a1.00_g0.25_b0.00_tv0.00_k0.00_")

    dcm_files = sorted(glob.glob(os.path.join(out_dir, "*.dcm")))
    assert len(dcm_files) == 5  # fazın tüm kesitleri
    assert os.path.isfile(os.path.join(out_dir, "parametreler.json"))

    ds = pydicom.dcmread(dcm_files[2])
    assert "IYILESTIRILMIS" in str(ds.SeriesDescription)
    hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    assert hu.shape == (128, 128)
    # İyileştirilmiş kesit: doku ortalaması korunur, gürültü düşer
    from backend.core.ingestion import CaseIngestor
    from backend.core.phase_selector import measure_homogeneous_sigma

    orig = CaseIngestor().ingest(multiphase_case_dir).phases["phase_75"].get_slice(2)
    mask = (orig > -300) & (orig < 300)
    assert abs(float(np.mean(hu[mask]) - np.mean(orig[mask]))) < 3.0
    assert measure_homogeneous_sigma(hu) < measure_homogeneous_sigma(orig)


def test_export_twice_with_different_params_makes_separate_folders(multiphase_case_dir):
    """Kullanıcı senaryosu: iki farklı ayarla iki dışa aktarım -> iki ayrı
    klasör, her birinde o anda gönderilen parametreler."""
    import json

    client.post("/api/v1/ingest", json={"path": multiphase_case_dir})
    sent = [
        {"alpha": 1.2, "gamma": 0.10, "beta": 0.3, "lambda_tv": 0.05, "kappa_hu": 0.8},
        {"alpha": 0.5, "gamma": 0.90, "beta": 0.0, "lambda_tv": 0.20, "kappa_hu": 0.0},
    ]
    out_dirs = []
    for params in sent:
        r = client.post("/api/v1/export-phase", json={
            "case_id": "case_demo", "phase": "phase_30",
            "model_id": "rdn", "params": params,
        })
        assert r.status_code == 200, r.text
        status = client.get(f"/api/v1/export/status/{r.json()['job_id']}").json()
        assert status["status"] == "done", status
        out_dirs.append(status["output_dir"])

    assert out_dirs[0] != out_dirs[1]
    for out_dir, params in zip(out_dirs, sent):
        with open(os.path.join(out_dir, "parametreler.json"), encoding="utf-8") as fh:
            recorded = json.load(fh)["parametreler"]
        assert recorded == pytest.approx(params)
    # Ayarlar klasör adından okunabilir
    assert "a1.20_g0.10_b0.30_tv0.05_k0.80" in os.path.basename(out_dirs[0])
    assert "a0.50_g0.90_b0.00_tv0.20_k0.00" in os.path.basename(out_dirs[1])


def test_export_uploaded_case_goes_to_exports_dir(multiphase_case_dir):
    """Sürükle-bırakla yüklenen vaka: çıktı gizli uploads kopyasına değil
    görünür disa_aktarimlar/<vaka>/ klasörüne yazılır."""
    pdir = os.path.join(multiphase_case_dir, "phase_75")
    files = []
    for fn in sorted(os.listdir(pdir)):
        with open(os.path.join(pdir, fn), "rb") as fh:
            files.append(("files", (f"vaka_y/{fn}", fh.read(), "application/dicom")))
    r = client.post("/api/v1/ingest-upload", files=files, data={"case_name": "vaka_y"})
    assert r.status_code == 200, r.text

    r = client.post("/api/v1/export-phase", json={
        "case_id": "vaka_y", "phase": next(iter(r.json()["phases"])),
        "model_id": "rdn", "params": {"alpha": 1.0},
    })
    assert r.status_code == 200, r.text
    status = client.get(f"/api/v1/export/status/{r.json()['job_id']}").json()
    assert status["status"] == "done", status
    from backend import config as cfg

    assert status["output_dir"].startswith(os.path.join(cfg.EXPORTS_DIR, "vaka_y"))
    assert not status["output_dir"].startswith(cfg.UPLOADS_DIR)


def test_export_phase_rejects_auto_best(multiphase_case_dir):
    client.post("/api/v1/ingest", json={"path": multiphase_case_dir})
    r = client.post("/api/v1/export-phase", json={
        "case_id": "case_demo", "phase": "phase_75", "model_id": "auto_best",
    })
    assert r.status_code == 422


def test_sr3d_async_job(multiphase_case_dir):
    client.post("/api/v1/ingest", json={"path": multiphase_case_dir})
    r = client.post("/api/v1/sr3d/submit", json={
        "case_id": "case_demo", "phase": "phase_75", "scale": 2,
    })
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    # TestClient arka plan görevini yanıt sonrası senkron çalıştırır
    r = client.get(f"/api/v1/sr3d/status/{job_id}")
    assert r.status_code == 200
    status = r.json()
    assert status["status"] == "done"
    assert status["fallback"] is True
    assert status["progress"] == 1.0
