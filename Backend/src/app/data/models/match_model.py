from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, DateTime, ForeignKey, String, Float, LargeBinary
from datetime import datetime
from typing import List, Optional
from .base import Base


class MatchModel(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    match_players: Mapped[List["MatchPlayerModel"]] = relationship(
        "MatchPlayerModel", back_populates="match", cascade="all, delete-orphan"
    )
    analysis: Mapped[Optional["AnalysisModel"]] = relationship(
        "AnalysisModel", back_populates="match", uselist=False
    )


class MatchPlayerModel(Base):
    __tablename__ = "match_players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    player_identifier: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    match: Mapped["MatchModel"] = relationship("MatchModel", back_populates="match_players")
    summary_metrics: Mapped[Optional["SummaryMetricsModel"]] = relationship(
        "SummaryMetricsModel", back_populates="match_player", uselist=False, cascade="all, delete-orphan"
    )


class HitsModel(Base):
    """Hit type breakdown - standalone table, referenced by SummaryMetrics"""
    __tablename__ = "hits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    overhead_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lob: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    serve: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    groundstrokes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # Changed from backhand
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    summary_metrics: Mapped[Optional["SummaryMetricsModel"]] = relationship(
        "SummaryMetricsModel", back_populates="hits", uselist=False
    )


class HeatmapModel(Base):
    """Heatmap stored as binary PNG - standalone table, referenced by SummaryMetrics"""
    __tablename__ = "heatmaps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    heatmap: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    summary_metrics: Mapped[Optional["SummaryMetricsModel"]] = relationship(
        "SummaryMetricsModel", back_populates="heatmap", uselist=False
    )


class SummaryMetricsModel(Base):
    __tablename__ = "summary_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_player_id: Mapped[int] = mapped_column(ForeignKey("match_players.id"), nullable=False)
    total_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_rallies: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hits_id: Mapped[Optional[int]] = mapped_column(ForeignKey("hits.id"), nullable=True)
    heatmap_id: Mapped[Optional[int]] = mapped_column(ForeignKey("heatmaps.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    match_player: Mapped["MatchPlayerModel"] = relationship("MatchPlayerModel", back_populates="summary_metrics")
    hits: Mapped[Optional["HitsModel"]] = relationship("HitsModel", back_populates="summary_metrics")
    heatmap: Mapped[Optional["HeatmapModel"]] = relationship("HeatmapModel", back_populates="summary_metrics")
    rallies: Mapped[List["RallyModel"]] = relationship(
        "RallyModel", back_populates="summary_metrics", cascade="all, delete-orphan"
    )


class RallyModel(Base):
    """Individual rally data - duration only"""
    __tablename__ = "rallies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_metrics_id: Mapped[int] = mapped_column(ForeignKey("summary_metrics.id"), nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    summary_metrics: Mapped["SummaryMetricsModel"] = relationship("SummaryMetricsModel", back_populates="rallies")