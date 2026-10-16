"""CaseIngestor testleri: DICOM/NIfTI/HDF5/ZIP okuma ve faz ayrıştırma."""

import os
import zipfile

import numpy as np
import pytest

from backend.core.ingestion import CaseIngestor, kernel_to_code
from tests.conftest import make_volume, write_dicom_series


def test_multiphase_dicom_directory(multiphase_case_dir):
    case = CaseIngestor().ingest(multiphase_case_dir)
    assert case.is_multiphase
    assert set(case.phases) == {"phase_30", "phase_55", "phase_75", "phase_85"}
    for pv in case.phases.values():
        assert pv.volume.dtype == np.float32
        assert pv.volume.shape == (5, 128, 128)
    # DICOM üst verisi çıkarıldı mı?
    assert case.metadata["kvp"] == 120.0
    assert case.metadata["mas"] == 300.0
    assert case.metadata["slice_thickness"] == 0.6
    assert case.metadata["convolution_kernel"] == "B30f"
    assert case.metadata["kernel_code"] == 30


def test_dicom_hu_conversion_is_exact(tmp_path):
    """RescaleSlope*pixel + RescaleIntercept dönüşümü kayıpsız olmalı."""
    vol = make_volume(n_slices=3, noise_sigma=0.0, vessel_hu=400.0)
    write_dicom_series(str(tmp_path / "s"), vol)
    case = CaseIngestor().ingest(str(tmp_path / "s"))
    out = case.phases["default"].volume
    # uint16 kuantalama nedeniyle en fazla 1 HU yuvarlama farkı
    assert np.max(np.abs(out - np.round(vol))) <= 1.0


def test_zip_archive_roundtrip(multiphase_case_dir, tmp_path):
    archive = tmp_path / "case_demo.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for dirpath, _, filenames in os.walk(multiphase_case_dir):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                zf.write(full, os.path.relpath(full, multiphase_case_dir))
    case = CaseIngestor().ingest(str(archive))
    assert case.case_id == "case_demo"
    assert len(case.phases) == 4


def test_nifti_roundtrip(tmp_path):
    import nibabel as nib

    vol = make_volume(n_slices=4)
    # Dahili (Z, Y, X) -> NIfTI (X, Y, Z)
    nii = nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), affine=np.eye(4))
    path = tmp_path / "vol.nii.gz"
    nib.save(nii, str(path))

    case = CaseIngestor().ingest(str(path))
    out = case.phases["default"].volume
    assert out.shape == vol.shape
    np.testing.assert_allclose(out, vol, atol=1e-3)


def test_hdf5_roundtrip(tmp_path):
    import h5py

    vol = make_volume(n_slices=4)
    path = tmp_path / "vol.hdf5"
    with h5py.File(path, "w") as hf:
        hf.create_dataset("ct/volume", data=vol)

    case = CaseIngestor().ingest(str(path))
    np.testing.assert_allclose(case.phases["default"].volume, vol, atol=1e-5)


def test_kernel_code_stability():
    assert kernel_to_code("B30f") == 30
    assert kernel_to_code("b30F") == 30
    assert kernel_to_code("") == 0
    assert kernel_to_code("SOFT") == kernel_to_code("SOFT")


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        CaseIngestor().ingest("/yok/boyle/bir/yol")


def test_flat_pacs_export_groups_series_and_phases(tmp_path):
    """Düz PACS klasörü: seriler UID ile, kardiyak fazlar yüzde etiketiyle ayrılır.

    Gerçek vaka senaryosu (nilufer): tek klasörde scout + çok fazlı seri.
    """
    flat = tmp_path / "flat_case"
    # Çok fazlı retrospektif seri: 6 kesit, %35 ve %75 fazları (3'er kesit)
    vol = make_volume(n_slices=6, noise_sigma=10.0, vessel_hu=400.0)
    write_dicom_series(
        str(flat), vol, series_number=303, series_description="0-90",
        cardiac_pcts=[35.0, 35.0, 35.0, 75.0, 75.0, 75.0],
        filename_prefix="kardiyak",
    )
    # Farklı boyutlu scout serisi (ayrı UID, ayrı şekil)
    scout = make_volume(n_slices=2, shape=(64, 96), noise_sigma=5.0)
    write_dicom_series(str(flat), scout, series_number=1,
                       series_description="Scout", filename_prefix="scout")

    case = CaseIngestor().ingest(str(flat))
    assert set(case.phases) == {"s303_0-90_%35", "s303_0-90_%75", "s1_Scout"}
    assert case.phases["s303_0-90_%35"].shape == (3, 128, 128)
    assert case.phases["s303_0-90_%75"].shape == (3, 128, 128)
    assert case.phases["s1_Scout"].shape == (2, 64, 96)
    # Vaka üst verisi en çok kesitli (aksiyel) fazdan gelir
    assert case.metadata["kvp"] == 120.0


def test_reingest_ignores_export_folders(tmp_path):
    """İçine IYILESTIRILMIS_ klasörü yazılmış vaka yeniden yüklenebilmeli:
    dışa aktarım klasörü faz adayı sayılmaz, kök dosyalar okunur."""
    flat = tmp_path / "vaka"
    vol = make_volume(n_slices=4, noise_sigma=10.0, vessel_hu=400.0)
    write_dicom_series(str(flat), vol, series_number=5,
                       series_description="Aksiyel")
    # Önceki bir dışa aktarımı simüle et
    export_dir = flat / "IYILESTIRILMIS_Aksiyel_rdn_a1.00_20250101-000000"
    write_dicom_series(str(export_dir), vol, series_number=1005,
                       series_description="Aksiyel IYILESTIRILMIS")

    case = CaseIngestor().ingest(str(flat))
    # Dışa aktarım klasörü yok sayılır; kökteki tek seri "default" fazıdır
    assert set(case.phases) == {"default"}
    assert case.phases["default"].num_slices == 4


def test_flat_dicom_is_lazy_and_slice_access_works(tmp_path):
    vol = make_volume(n_slices=5, noise_sigma=0.0, vessel_hu=300.0)
    write_dicom_series(str(tmp_path / "f"), vol)
    case = CaseIngestor().ingest(str(tmp_path / "f"))
    pv = case.phases["default"]
    assert pv.is_lazy  # ingest piksel çözmedi
    sl = pv.get_slice(2)
    assert sl.shape == (128, 128)
    assert np.max(np.abs(sl - np.round(vol[2]))) <= 1.0
    # Tam hacim erişimi tembel yüklemeyi tamamlar
    assert pv.volume.shape == (5, 128, 128)
    assert not pv.is_lazy
