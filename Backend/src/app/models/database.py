# The following file contains the models used by SQLAlchemy to map and create the PostgreSQL database

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, DateTime, Float, ForeignKey, Text, Boolean
from datetime import datetime
from typing import Optional, List


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="player")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    # uploaded_videos: Mapped[List["Video"]] = relationship("Video", back_populates="uploader")
    # analyses: Mapped[List["Analysis"]] = relationship("Analysis", back_populates="player")


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uploading_player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id"), nullable=False
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), default="uploaded"
    )  # uploaded, processing, analyzed, error
    upload_timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    video_length: Mapped[Optional[float]] = mapped_column(Float)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    # uploader = Mapped["Player"] = relationship("Player", back_populates="uploaded_videos")
    # analysis: Mapped[Optional["Analysis"]] = relationship("Analysis", back_populates="video", uselist=False)


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    analysis_timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )
    summary_metrics_data: Mapped[Optional[str]] = mapped_column(Text)  # JSON string
    heatmap_data: Mapped[Optional[str]] = mapped_column(Text)  # JSON string

    # Relationships
    # video: Mapped["Video"] = relationship("Video", back_populates="analysis")
    # player: Mapped["Player"] = relationship("Player", back_populates="analyses")
