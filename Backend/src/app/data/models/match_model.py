from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, DateTime, ForeignKey, String
from datetime import datetime
from typing import List
from .base import Base

class MatchModel(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    match_players: Mapped[List["MatchPlayerModel"]] = relationship("MatchPlayerModel", back_populates="match")
    analysis: Mapped["AnalysisModel"] = relationship("AnalysisModel", back_populates="match", uselist=False)

class MatchPlayerModel(Base):
    __tablename__ = "match_players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    player_identifier: Mapped[str] = mapped_column(String(20), nullable=False)  # "player_1", etc.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    match: Mapped["MatchModel"] = relationship("MatchModel", back_populates="match_players")
    summary_metrics: Mapped["SummaryMetricsModel"] = relationship("SummaryMetricsModel", back_populates="match_player", uselist=False)

class SummaryMetricsModel(Base):
    __tablename__ = "summary_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_player_id: Mapped[int] = mapped_column(ForeignKey("match_players.id"), nullable=False)
    total_hits: Mapped[int] = mapped_column(Integer, nullable=False)
    total_rallies: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    match_player: Mapped["MatchPlayerModel"] = relationship("MatchPlayerModel", back_populates="summary_metrics")
    heatmap: Mapped["HeatmapModel"] = relationship("HeatmapModel", back_populates="summary_metrics", uselist=False)
    rallies: Mapped[List["RallyModel"]] = relationship("RallyModel", back_populates="summary_metrics")
    hits: Mapped["HitsModel"] = relationship("HitsModel", back_populates="summary_metrics", uselist=False)

class HitsModel(Base):
    __tablename__ = "hits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_metrics_id: Mapped[int] = mapped_column(ForeignKey("summary_metrics.id"), nullable=False)
    hit_errors: Mapped[int] = mapped_column(Integer, nullable=False)
    overhead_hits: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    summary_metrics: Mapped["SummaryMetricsModel"] = relationship("SummaryMetricsModel", back_populates="hits")

class RallyModel(Base):
    __tablename__ = "rallies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_metrics_id: Mapped[int] = mapped_column(ForeignKey("summary_metrics.id"), nullable=False)
    hits: Mapped[int] = mapped_column(Integer, nullable=False)
    length_in_time: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    summary_metrics: Mapped["SummaryMetricsModel"] = relationship("SummaryMetricsModel", back_populates="rallies")

class HeatmapModel(Base):
    __tablename__ = "heatmaps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    summary_metrics_id: Mapped[int] = mapped_column(ForeignKey("summary_metrics.id"), nullable=False)
    offensive_zone_time: Mapped[float] = mapped_column(nullable=False)
    defensive_zone_time: Mapped[float] = mapped_column(nullable=False)
    transition_zone_time: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    summary_metrics: Mapped["SummaryMetricsModel"] = relationship("SummaryMetricsModel", back_populates="heatmap")
    coordinates: Mapped[List["HeatmapCoordModel"]] = relationship("HeatmapCoordModel", back_populates="heatmap")

class HeatmapCoordModel(Base):
    __tablename__ = "heatmap_coords"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    heatmap_id: Mapped[int] = mapped_column(ForeignKey("heatmaps.id"), nullable=False)
    x_coord: Mapped[float] = mapped_column(nullable=False)
    y_coord: Mapped[float] = mapped_column(nullable=False)
    intensity: Mapped[float] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    heatmap: Mapped["HeatmapModel"] = relationship("HeatmapModel", back_populates="coordinates")