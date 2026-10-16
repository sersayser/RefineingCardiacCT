"""SQLAlchemy tabloları: presets, processed_cases, batch_jobs.

Şema, plan dokümanı 3.2 ile birebir uyumludur. SQLite dosyası
`database/clinical_learning.sqlite` altında tutulur; testler için
`create_engine_and_session(":memory:")` kullanılabilir.
"""

from __future__ import annotations

import datetime
import os

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from backend import config

def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


Base = declarative_base()


class Preset(Base):
    """Kullanıcının adlandırılmış kaydırıcı ön ayarları."""

    __tablename__ = "presets"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    model_type = Column(String, nullable=False, default="rdn")
    alpha = Column(Float, nullable=False, default=config.DEFAULT_PARAMS["alpha"])
    gamma = Column(Float, nullable=False, default=config.DEFAULT_PARAMS["gamma"])
    lambda_tv = Column(Float, nullable=False, default=config.DEFAULT_PARAMS["lambda_tv"])
    beta = Column(Float, nullable=False, default=config.DEFAULT_PARAMS["beta"])
    kappa_hu = Column(Float, nullable=False, default=config.DEFAULT_PARAMS["kappa_hu"])
    window_center = Column(Float, nullable=False, default=50.0)
    window_width = Column(Float, nullable=False, default=400.0)
    created_at = Column(DateTime, default=_utcnow)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lambda_tv": self.lambda_tv,
            "beta": self.beta,
            "kappa_hu": self.kappa_hu,
            "window_center": self.window_center,
            "window_width": self.window_width,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CaseRecord(Base):
    """Klinisyen onaylı işlenmiş vakalar: sürekli öğrenmenin veri kaynağı."""

    __tablename__ = "processed_cases"

    id = Column(Integer, primary_key=True)
    case_id = Column(String, nullable=False)
    folder_path = Column(Text, default="")
    selected_phase = Column(String, default="")
    phase_score = Column(Float, default=0.0)

    # X_i özellik vektörü bileşenleri
    kvp = Column(Float, default=0.0)
    mas = Column(Float, default=0.0)
    slice_thickness = Column(Float, default=0.0)
    kernel_code = Column(Float, default=0.0)
    sigma_hu_before = Column(Float, default=0.0)
    sigma_hu_after = Column(Float, default=0.0)
    gmm_peak_hu = Column(Float, default=0.0)
    niqe_before = Column(Float, default=0.0)
    niqe_after = Column(Float, default=0.0)
    ssim_inter = Column(Float, default=0.0)

    # Y_i hedef vektörü bileşenleri
    approved_model = Column(String, default="rdn")
    alpha = Column(Float, default=config.DEFAULT_PARAMS["alpha"])
    gamma = Column(Float, default=config.DEFAULT_PARAMS["gamma"])
    lambda_tv = Column(Float, default=config.DEFAULT_PARAMS["lambda_tv"])
    beta = Column(Float, default=config.DEFAULT_PARAMS["beta"])
    kappa_hu = Column(Float, default=config.DEFAULT_PARAMS["kappa_hu"])

    used_in_training = Column(Boolean, default=False)
    created_at = Column(DateTime, default=_utcnow)

    def as_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns
                if c.name != "created_at"}


class BatchJob(Base):
    """Toplu vaka analiz işlerinin kaydı (ikinci sekme)."""

    __tablename__ = "batch_jobs"

    id = Column(Integer, primary_key=True)
    status = Column(String, default="queued")   # queued | running | done | error
    total_cases = Column(Integer, default=0)
    completed_cases = Column(Integer, default=0)
    error = Column(Text, default="")
    created_at = Column(DateTime, default=_utcnow)


def create_engine_and_session(db_path: str | None = None):
    """Engine + Session fabrikası kurar ve tabloları oluşturur.

    db_path None -> config.DATABASE_PATH; ':memory:' -> bellek içi test DB.
    """
    if db_path is None:
        db_path = config.DATABASE_PATH
    if db_path == ":memory:":
        url = "sqlite:///:memory:"
    else:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        url = f"sqlite:///{db_path}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)
