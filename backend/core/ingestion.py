"""Vaka yutma modülü: .dcm, .nii, .nii.gz, .hdf5, .zip, .rar okuyucu.

DICOM kesitleri `RescaleSlope * pixel + RescaleIntercept` dönüşümüyle gerçek
float32 HU tensörüne çevrilir (pydicom birincil; çözülemezse SimpleITK yedek).

Faz ayrıştırma iki düzeyde çalışır:
1. Alt klasörler: her görüntü içeren alt klasör ayrı faz adayıdır.
2. DICOM başlıkları: düz PACS dışa aktarımlarında (tek klasörde binlerce
   dosya) seriler `SeriesInstanceUID` ile gruplanır; çok fazlı kardiyak
   seriler `NominalPercentageOfCardiacPhase` (0020,9241) etiketiyle faz
   başına bölünür. Scout/farklı boyutlu kesitler kendi grubunda kalır.

Bellek: DICOM fazları TEMBEL yüklenir — ingest sırasında yalnız başlıklar
okunur (hızlı), piksel verisi ilk erişimde kesit kesit çözülür ve önbelleğe
alınır. 3 GB'lık düz klasörler bu sayede saniyeler içinde yutulur.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np

from backend import config

logger = logging.getLogger(__name__)

DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima"}
NIFTI_EXTENSIONS = {".nii"}  # .nii.gz çift uzantı olarak ayrıca ele alınır
HDF5_EXTENSIONS = {".hdf5", ".h5", ".hdf"}
ARCHIVE_EXTENSIONS = {".zip", ".rar"}

# Dışa aktarımlar orijinal vaka klasörünün İÇİNE yazılır (exporter.py);
# vaka yeniden yüklendiğinde bu klasörler faz adayı sayılmaz.
EXPORT_DIR_PREFIX = "IYILESTIRILMIS_"


def _extension_of(path: str) -> str:
    """Çift uzantıyı (.nii.gz) tek parça döndürür."""
    lower = path.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(lower)[1]


class PhaseVolume:
    """Tek bir fazın 3B hacmi; tembel (lazy) veya hazır (eager) olabilir.

    Tembel modda `shape` başlıklardan bilinir, pikseller ilk erişimde
    `slice_loader(idx)` ile çözülür. `volume` özelliği tam hacmi kurup
    önbelleğe alır; `get_slice`/`center_slices` yalnız gereken kesitleri çözer.
    """

    def __init__(
        self,
        name: str,
        volume: Optional[np.ndarray] = None,
        spacing: tuple = (1.0, 1.0, 1.0),
        metadata: Optional[dict] = None,
        source_paths: Optional[list] = None,
        shape: Optional[tuple] = None,
        slice_loader: Optional[Callable[[int], np.ndarray]] = None,
    ) -> None:
        self.name = name
        self.spacing = spacing
        self.metadata = metadata or {}
        self.source_paths = source_paths or []
        self._slice_loader = slice_loader
        self._slice_cache: dict[int, np.ndarray] = {}
        self._volume: Optional[np.ndarray] = None
        if volume is not None:
            self._volume = np.ascontiguousarray(volume, dtype=np.float32)
            self._shape = self._volume.shape
        elif shape is not None and slice_loader is not None:
            self._shape = tuple(int(s) for s in shape)
        else:
            raise ValueError("PhaseVolume: volume ya da (shape + slice_loader) gerekli")

    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple:
        return self._volume.shape if self._volume is not None else self._shape

    @property
    def num_slices(self) -> int:
        return int(self.shape[0])

    @property
    def is_lazy(self) -> bool:
        return self._volume is None

    @property
    def volume(self) -> np.ndarray:
        """Tam hacim; tembel fazda tüm kesitleri çözüp önbelleğe alır."""
        if self._volume is None:
            self._volume = np.stack(
                [self.get_slice(i) for i in range(self.num_slices)], axis=0
            ).astype(np.float32)
            self._slice_cache.clear()
        return self._volume

    def get_slice(self, idx: int) -> np.ndarray:
        if not (0 <= idx < self.num_slices):
            raise IndexError(f"Kesit indeksi aralık dışı: {idx}/{self.num_slices}")
        if self._volume is not None:
            return self._volume[idx]
        cached = self._slice_cache.get(idx)
        if cached is None:
            cached = np.ascontiguousarray(self._slice_loader(idx), dtype=np.float32)
            self._slice_cache[idx] = cached
        return cached

    def center_slices(self, count: int = config.PHASE_CENTER_SLICES) -> np.ndarray:
        """Hacmin merkez `count` kesiti (tembel fazda yalnız bunlar çözülür)."""
        z = self.num_slices
        take = min(count, z)
        start = max(0, z // 2 - take // 2)
        stop = min(z, start + take)
        if self._volume is not None:
            return self._volume[start:stop]
        return np.stack([self.get_slice(i) for i in range(start, stop)], axis=0)


class IngestedCase:
    """Bir vakanın tüm faz adayları ve vaka düzeyi üst verisi."""

    def __init__(self, case_id: str, source_path: str,
                 phases: Optional[dict] = None, metadata: Optional[dict] = None) -> None:
        self.case_id = case_id
        self.source_path = source_path
        self.phases: dict[str, PhaseVolume] = phases or {}
        self.metadata: dict = metadata or {}

    @property
    def is_multiphase(self) -> bool:
        return len(self.phases) > 1


class CaseIngestor:
    """PACS'tan dışa aktarılan dosya/klasör/arşivleri HU hacimlerine çevirir."""

    def __init__(self, workdir: Optional[str] = None) -> None:
        self._workdir = workdir  # arşiv çıkarma hedefi; None -> tempfile

    # ------------------------------------------------------------------ #
    # Genel giriş noktası
    # ------------------------------------------------------------------ #
    def ingest(self, path: str) -> IngestedCase:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vaka yolu bulunamadı: {path}")

        if os.path.isdir(path):
            return self._ingest_directory(path)

        ext = _extension_of(path)
        if ext in ARCHIVE_EXTENSIONS:
            extracted = self._extract_archive(path)
            case = self._ingest_directory(extracted)
            case.source_path = path
            case.case_id = os.path.splitext(os.path.basename(path))[0]
            return case

        # Tekil dosya -> tek fazlı vaka
        phases = self._load_single_file(path)
        case = IngestedCase(
            case_id=os.path.splitext(os.path.basename(path))[0],
            source_path=path,
            phases=phases,
            metadata=dict(next(iter(phases.values())).metadata),
        )
        return case

    # ------------------------------------------------------------------ #
    # Arşiv çözme
    # ------------------------------------------------------------------ #
    def _extract_archive(self, path: str) -> str:
        target = self._workdir or tempfile.mkdtemp(prefix="ct_case_")
        os.makedirs(target, exist_ok=True)
        ext = _extension_of(path)
        if ext == ".zip":
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target)
        elif ext == ".rar":
            try:
                import rarfile
            except ImportError as exc:  # pragma: no cover - ortam bağımlı
                raise RuntimeError(
                    ".rar desteği için 'rarfile' paketi (ve unar/unrar aracı) gerekli"
                ) from exc
            with rarfile.RarFile(path) as rf:
                rf.extractall(target)
        else:  # pragma: no cover - ingest() yalnız bilinen uzantıları yollar
            raise ValueError(f"Desteklenmeyen arşiv türü: {ext}")
        logger.info("Arşiv çözüldü: %s -> %s", path, target)
        return target

    # ------------------------------------------------------------------ #
    # Klasör yutma
    # ------------------------------------------------------------------ #
    def _ingest_directory(self, root: str) -> IngestedCase:
        case = IngestedCase(case_id=os.path.basename(root.rstrip(os.sep)),
                            source_path=root)

        phase_dirs = self._find_phase_directories(root)
        if not phase_dirs:
            raise ValueError(f"Klasörde okunabilir görüntü dosyası yok: {root}")

        for dir_name, directory in phase_dirs.items():
            try:
                groups = self._load_phase_directory(dir_name, directory)
            except Exception:
                logger.exception("Faz klasörü okunamadı, atlanıyor: %s", directory)
                continue
            for name, pv in groups.items():
                pv.name = name
                case.phases[name] = pv

        if not case.phases:
            raise ValueError(f"Hiçbir faz hacmi okunamadı: {root}")

        # Vaka düzeyi üst veri: en çok kesitli fazın DICOM başlıklarından
        richest = max(case.phases.values(), key=lambda p: p.num_slices)
        case.metadata = dict(richest.metadata)
        return case

    def _find_phase_directories(self, root: str) -> dict[str, str]:
        """Görüntü içeren alt klasörleri bulur; yoksa kökün kendisi tek adaydır."""

        def has_images(d: str) -> bool:
            try:
                entries = os.listdir(d)
            except OSError:
                return False
            for name in entries:
                ext = _extension_of(name)
                if ext in DICOM_EXTENSIONS | NIFTI_EXTENSIONS | HDF5_EXTENSIONS or ext == ".nii.gz":
                    return True
                full = os.path.join(d, name)
                if os.path.isfile(full) and not ext and _looks_like_dicom(full):
                    return True
            return False

        subdirs = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
            and not d.startswith(EXPORT_DIR_PREFIX)  # kendi çıktılarımızı atla
        )
        image_subdirs = {d: os.path.join(root, d) for d in subdirs
                         if has_images(os.path.join(root, d))}

        # Arşivler bazen tek bir sarmalayıcı klasör içerir; bir seviye derine in
        if not image_subdirs and len(subdirs) == 1 and not has_images(root):
            return self._find_phase_directories(os.path.join(root, subdirs[0]))

        # Kökteki görüntüler, görüntülü alt klasörler olsa bile aday kalır
        # (düz PACS klasörü + içine yazılmış başka klasörler karışık durumu)
        result = dict(image_subdirs)
        if has_images(root):
            result["default"] = root
        return result

    def _load_phase_directory(self, dir_name: str, directory: str) -> dict[str, PhaseVolume]:
        """Bir klasörün içeriğini bir veya daha çok faza çevirir."""
        files = sorted(
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and not f.startswith(".")
        )
        dicom_files = [f for f in files if _extension_of(f) in DICOM_EXTENSIONS or
                       (not _extension_of(f) and _looks_like_dicom(f))]
        nifti_files = [f for f in files if _extension_of(f) in NIFTI_EXTENSIONS or
                       _extension_of(f) == ".nii.gz"]
        hdf5_files = [f for f in files if _extension_of(f) in HDF5_EXTENSIONS]

        if dicom_files:
            groups = self._group_dicom_phases(dicom_files)
            return _name_groups(dir_name, groups)
        if nifti_files:
            return {dir_name if dir_name != "default" else "default":
                    self._load_nifti(dir_name, nifti_files[0])}
        if hdf5_files:
            return {dir_name if dir_name != "default" else "default":
                    self._load_hdf5(dir_name, hdf5_files[0])}
        return {}

    # ------------------------------------------------------------------ #
    # DICOM: başlıklardan seri/faz gruplama + tembel hacimler
    # ------------------------------------------------------------------ #
    def _group_dicom_phases(self, files: list[str]) -> dict[str, PhaseVolume]:
        """Düz DICOM dosya listesini seri + kardiyak faz gruplarına böler.

        - `SeriesInstanceUID` -> seri grubu (scout, kemik, aksiyel... ayrışır)
        - Seri içinde `NominalPercentageOfCardiacPhase` birden çok değer
          alıyorsa her yüzde ayrı fazdır (örn. '0-90' retrospektif seri).
        - Grup içinde baskın (Rows, Columns) boyutuna uymayan kesitler atılır.
        """
        import pydicom

        headers: list[tuple[str, object]] = []
        for f in files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception:
                logger.warning("DICOM başlığı okunamadı, atlandı: %s", f)
                continue
            if not getattr(ds, "Rows", None) or not getattr(ds, "Columns", None):
                continue  # görüntü taşımayan nesne (SR, doz raporu...)
            headers.append((f, ds))
        if not headers:
            raise ValueError("Klasörde piksel verili DICOM yok")

        by_series: dict[str, list] = defaultdict(list)
        for f, ds in headers:
            by_series[str(getattr(ds, "SeriesInstanceUID", "tek-seri"))].append((f, ds))

        groups: list[tuple[tuple, str, list]] = []  # (sıralama anahtarı, ad, öğeler)
        for items in by_series.values():
            first = items[0][1]
            series_no = getattr(first, "SeriesNumber", None)
            desc = str(getattr(first, "SeriesDescription", "") or "").strip()
            label = _sanitize_name(
                (f"s{series_no}_" if series_no is not None else "") + (desc or "seri")
            )

            pcts = sorted({_cardiac_pct(ds) for _, ds in items} - {None})
            if len(pcts) >= 2:
                by_pct: dict[Optional[float], list] = defaultdict(list)
                for f, ds in items:
                    by_pct[_cardiac_pct(ds)].append((f, ds))
                for pct, sub in by_pct.items():
                    suffix = f"_%{pct:g}" if pct is not None else "_%bilinmiyor"
                    groups.append(((_num_or_inf(series_no), pct if pct is not None else 1e9),
                                   label + suffix, sub))
            else:
                groups.append(((_num_or_inf(series_no), 0.0), label, items))

        phases: dict[str, PhaseVolume] = {}
        for _, label, items in sorted(groups, key=lambda g: g[0]):
            # Baskın kesit boyutu; uymayanlar (ör. karma scout) atılır
            shape_counts = Counter((int(ds.Rows), int(ds.Columns)) for _, ds in items)
            dominant = shape_counts.most_common(1)[0][0]
            items = [(f, ds) for f, ds in items
                     if (int(ds.Rows), int(ds.Columns)) == dominant]
            items.sort(key=_slice_sort_key)

            name = label
            counter = 2
            while name in phases:  # ad çakışması (aynı açıklamalı iki seri)
                name = f"{label}_{counter}"
                counter += 1
            phases[name] = _lazy_dicom_phase(name, items, dominant)
        return phases

    # ------------------------------------------------------------------ #
    # Tekil dosya okuyucular
    # ------------------------------------------------------------------ #
    def _load_single_file(self, path: str) -> dict[str, PhaseVolume]:
        ext = _extension_of(path)
        if ext in DICOM_EXTENSIONS or (not ext and _looks_like_dicom(path)):
            groups = self._group_dicom_phases([path])
            return _name_groups("default", groups)
        if ext in NIFTI_EXTENSIONS or ext == ".nii.gz":
            return {"default": self._load_nifti("default", path)}
        if ext in HDF5_EXTENSIONS:
            return {"default": self._load_hdf5("default", path)}
        raise ValueError(f"Desteklenmeyen dosya türü: {path}")

    def _load_nifti(self, name: str, path: str) -> PhaseVolume:
        import nibabel as nib

        img = nib.load(path)
        data = np.asanyarray(img.dataobj).astype(np.float32)
        # NIfTI (X, Y, Z) -> dahili (Z, Y, X)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            data = np.transpose(data, (2, 1, 0))
        elif data.ndim == 4:  # 4B seri: ilk zaman noktası
            data = np.transpose(data[..., 0], (2, 1, 0))
        else:
            raise ValueError(f"Beklenmeyen NIfTI boyutu {data.shape}: {path}")
        zooms = img.header.get_zooms()[:3]
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0])) if len(zooms) == 3 else (1.0, 1.0, 1.0)
        data = np.clip(data, config.HU_MIN, config.HU_MAX)
        return PhaseVolume(name=name, volume=np.ascontiguousarray(data),
                           spacing=spacing, metadata={}, source_paths=[path])

    def _load_hdf5(self, name: str, path: str) -> PhaseVolume:
        import h5py

        with h5py.File(path, "r") as hf:
            dataset = _largest_dataset(hf)
            if dataset is None:
                raise ValueError(f"HDF5 içinde 2B/3B veri kümesi yok: {path}")
            data = dataset[()].astype(np.float32)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        data = np.clip(data, config.HU_MIN, config.HU_MAX)
        return PhaseVolume(name=name, volume=data, metadata={}, source_paths=[path])


# ---------------------------------------------------------------------- #
# DICOM yardımcıları
# ---------------------------------------------------------------------- #
def _read_dicom_slice_hu(path: str) -> np.ndarray:
    """Tek DICOM dosyasını float32 HU kesitine çevirir (pydicom -> SITK yedek)."""
    import pydicom

    try:
        ds = pydicom.dcmread(path)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = ds.pixel_array.astype(np.float32) * slope + intercept
    except Exception:
        logger.warning("pydicom piksel çözemedi, SimpleITK deneniyor: %s", path)
        import SimpleITK as sitk

        img = sitk.ReadImage(path)  # SITK slope/intercept'i kendisi uygular
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        hu = arr[0] if arr.ndim == 3 else arr
    # Üretici dolgu değerleri (örn. GE FOV dışı -3024) geçerli HU aralığına
    # kırpılır; aksi halde ΔI fark haritası FOV dışında anlamsız şişer.
    return np.clip(hu, config.HU_MIN, config.HU_MAX)


def _lazy_dicom_phase(name: str, items: list, shape2d: tuple) -> PhaseVolume:
    """Sıralı (dosya, başlık) listesinden tembel PhaseVolume kurar."""
    paths = [f for f, _ in items]
    first = items[0][1]
    spacing = _dicom_spacing(first, items)
    metadata = _dicom_metadata(first)
    pct = _cardiac_pct(first)
    if pct is not None:
        metadata["cardiac_phase_pct"] = pct

    def load_slice(idx: int, _paths=paths) -> np.ndarray:
        return _read_dicom_slice_hu(_paths[idx])

    return PhaseVolume(
        name=name,
        shape=(len(paths), shape2d[0], shape2d[1]),
        spacing=spacing,
        metadata=metadata,
        source_paths=paths,
        slice_loader=load_slice,
    )


def _name_groups(dir_name: str, groups: dict[str, PhaseVolume]) -> dict[str, PhaseVolume]:
    """Alt klasör adı ile başlık gruplarını tutarlı faz adlarına bağlar.

    - Alt klasör + tek grup  -> alt klasör adı (eski davranış korunur)
    - Kök ("default") + tek grup -> "default"
    - Birden çok grup -> grup adları (alt klasördeyse "klasör/grup")
    """
    if len(groups) == 1:
        pv = next(iter(groups.values()))
        return {dir_name: pv}
    if dir_name == "default":
        return groups
    return {f"{dir_name}/{g}": pv for g, pv in groups.items()}


def _cardiac_pct(ds) -> Optional[float]:
    value = getattr(ds, "NominalPercentageOfCardiacPhase", None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _num_or_inf(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _sanitize_name(label: str) -> str:
    label = re.sub(r"[^\w%.-]+", "_", label.strip())
    return re.sub(r"_+", "_", label).strip("_") or "seri"


def _slice_sort_key(item):
    """Kesit sırası: ImagePositionPatient z > InstanceNumber > dosya adı."""
    f, ds = item
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) == 3:
        return (0, float(ipp[2]), f)
    inst = getattr(ds, "InstanceNumber", None)
    if inst is not None:
        return (1, float(inst), f)
    return (2, 0.0, f)


def _looks_like_dicom(path: str) -> bool:
    """128 bayt önsöz + 'DICM' imzasını denetler."""
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def _dicom_spacing(first, items) -> tuple:
    pixel = getattr(first, "PixelSpacing", [1.0, 1.0])
    try:
        row, col = float(pixel[0]), float(pixel[1])
    except (TypeError, ValueError, IndexError):
        row = col = 1.0
    thickness = float(getattr(first, "SliceThickness", 1.0) or 1.0)
    if len(items) >= 2:
        ipp0 = getattr(items[0][1], "ImagePositionPatient", None)
        ipp1 = getattr(items[1][1], "ImagePositionPatient", None)
        if ipp0 is not None and ipp1 is not None:
            dz = abs(float(ipp1[2]) - float(ipp0[2]))
            if dz > 1e-6:
                thickness = dz
    return (thickness, row, col)


_KERNEL_CODE_CACHE: dict[str, int] = {}


def kernel_to_code(kernel: str) -> int:
    """ConvolutionKernel metnini kararlı bir tamsayı koda çevirir (X_i özelliği)."""
    kernel = (kernel or "").strip().upper()
    if not kernel:
        return 0
    if kernel not in _KERNEL_CODE_CACHE:
        match = re.search(r"(\d+)", kernel)
        if match:
            _KERNEL_CODE_CACHE[kernel] = int(match.group(1))
        else:
            _KERNEL_CODE_CACHE[kernel] = (hash(kernel) % 1000) + 1000
    return _KERNEL_CODE_CACHE[kernel]


def _dicom_metadata(ds) -> dict:
    kernel_raw = getattr(ds, "ConvolutionKernel", "")
    if isinstance(kernel_raw, (list, tuple)) or kernel_raw.__class__.__name__ == "MultiValue":
        kernel_raw = "/".join(str(k) for k in kernel_raw)
    metadata = {
        "kvp": float(getattr(ds, "KVP", 0.0) or 0.0),
        "mas": float(getattr(ds, "XRayTubeCurrent", 0.0) or 0.0),
        "slice_thickness": float(getattr(ds, "SliceThickness", 0.0) or 0.0),
        "convolution_kernel": str(kernel_raw),
        "kernel_code": kernel_to_code(str(kernel_raw)),
        "modality": str(getattr(ds, "Modality", "")),
        "study_description": str(getattr(ds, "StudyDescription", "")),
        "series_description": str(getattr(ds, "SeriesDescription", "")),
    }
    return metadata


def _largest_dataset(group):
    """HDF5 ağacındaki en büyük 2B/3B veri kümesini derinlemesine arar."""
    import h5py

    best = None
    best_size = -1

    def visit(_name, obj):
        nonlocal best, best_size
        if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3) and obj.size > best_size:
            best, best_size = obj, obj.size

    group.visititems(visit)
    return best
