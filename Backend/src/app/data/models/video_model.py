from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, Float, Boolean
from datetime import datetime
from typing import Optional
from .base import Base

class VideoModel(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="uploaded")
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    video_length: Mapped[Optional[float]] = mapped_column(Float)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)


    # Relationships
    analysis: Mapped[Optional["AnalysisModel"]] = relationship("AnalysisModel", back_populates="video", uselist=False)
