"""Sentetik BT fantomları ve DICOM yazıcı yardımcıları (test altyapısı)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# Testler gerçek veritabanını / uploads / disa_aktarimlar klasörlerini
# kirletmesin: backend.config env değişkenlerini import sırasında okur ve
# conftest her test modülünden önce yüklenir.
_TEST_ROOT = tempfile.mkdtemp(prefix="ct_test_")
os.environ.setdefault("CT_ENHANCER_DB", os.path.join(_TEST_ROOT, "test.sqlite"))
os.environ.setdefault("CT_ENHANCER_UPLOADS", os.path.join(_TEST_ROOT, "uploads"))
os.environ.setdefault("CT_ENHANCER_EXPORTS", os.path.join(_TEST_ROOT, "disa_aktarimlar"))


def make_ct_slice(
    shape: tuple[int, int] = (128, 128),
    noise_sigma: float = 30.0,
    vessel_hu: float = 0.0,
    tissue_hu: float = 40.0,
    seed: int = 0,
    shift: int = 0,
) -> np.ndarray:
    """Sentetik kardiyak BT kesiti: hava zemin, doku dairesi, opsiyonel lümen.

    vessel_hu > 0 ise merkezde kontrastlı damar lümeni (daire) eklenir.
    shift, damarı x ekseninde kaydırır (hareket/faz simülasyonu).
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full(shape, -1000.0, dtype=np.float32)

    body = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (min(h, w) * 0.42) ** 2
    img[body] = tissue_hu

    if vessel_hu > 0:
        cx = w / 2 + shift
        vessel = (yy - h / 2) ** 2 + (xx - cx) ** 2 < (min(h, w) * 0.08) ** 2
        img[vessel] = vessel_hu

    img[body] += rng.normal(0.0, noise_sigma, size=shape).astype(np.float32)[body]
    return np.clip(img, -1024.0, 3071.0).astype(np.float32)


def make_volume(
    n_slices: int = 6,
    shape: tuple[int, int] = (128, 128),
    noise_sigma: float = 30.0,
    vessel_hu: float = 0.0,
    shift: int = 0,
    seed: int = 0,
) -> np.ndarray:
    return np.stack(
        [make_ct_slice(shape, noise_sigma, vessel_hu, seed=seed + z, shift=shift)
         for z in range(n_slices)],
        axis=0,
    )


def write_dicom_series(
    directory: str,
    volume: np.ndarray,
    kvp: float = 120.0,
    mas: float = 300.0,
    slice_thickness: float = 0.6,
    kernel: str = "B30f",
    series_uid: str | None = None,
    series_number: int | None = None,
    series_description: str = "",
    cardiac_pcts: list[float] | None = None,
    filename_prefix: str = "slice",
) -> None:
    """float32 HU hacmini RescaleSlope/Intercept'li DICOM serisine yazar.

    cardiac_pcts: kesit başına NominalPercentageOfCardiacPhase değeri
    (çok fazlı retrospektif seri simülasyonu için).
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    os.makedirs(directory, exist_ok=True)
    intercept, slope = -1024.0, 1.0
    series_uid = series_uid or generate_uid()

    for z in range(volume.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = series_uid
        ds.Modality = "CT"
        if series_number is not None:
            ds.SeriesNumber = series_number
        if series_description:
            ds.SeriesDescription = series_description
        if cardiac_pcts is not None:
            ds.NominalPercentageOfCardiacPhase = cardiac_pcts[z]
        ds.KVP = kvp
        ds.XRayTubeCurrent = mas
        ds.SliceThickness = slice_thickness
        ds.ConvolutionKernel = kernel
        ds.InstanceNumber = z + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(z) * slice_thickness]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [0.5, 0.5]
        ds.RescaleIntercept = intercept
        ds.RescaleSlope = slope
        ds.Rows, ds.Columns = volume.shape[1], volume.shape[2]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0

        pixels = np.clip((volume[z] - intercept) / slope, 0, 4095).astype(np.uint16)
        ds.PixelData = pixels.tobytes()
        ds.save_as(os.path.join(directory, f"{filename_prefix}_{z:03d}.dcm"),
                   enforce_file_format=True)


@pytest.fixture
def multiphase_case_dir(tmp_path):
    """4 fazlı sentetik vaka klasörü; %75 fazı en iyi (temiz + kontrastlı).

    - %30: hareketli (shift 6), gürültülü, kontrast zayıf
    - %55: orta gürültü, orta kontrast
    - %75: düşük gürültü, mu_peak ~450 HU (optimal), hareketsiz
    - %85: düşük kontrast, orta gürültü
    """
    root = tmp_path / "case_demo"
    specs = {
        "phase_30": dict(noise_sigma=35.0, vessel_hu=180.0, shift=6),
        "phase_55": dict(noise_sigma=25.0, vessel_hu=260.0, shift=2),
        "phase_75": dict(noise_sigma=10.0, vessel_hu=450.0, shift=0),
        "phase_85": dict(noise_sigma=20.0, vessel_hu=150.0, shift=1),
    }
    for name, spec in specs.items():
        vol = make_volume(n_slices=5, **spec)
        write_dicom_series(str(root / name), vol)
    return str(root)
