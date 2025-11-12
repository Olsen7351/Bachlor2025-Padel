from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, DateTime, ForeignKey
from datetime import datetime
from .base import Base

class AnalysisModel(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[str] = mapped_column(ForeignKey("players.id"), nullable=False)  # Firebase string ID
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"), nullable=False)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=True)
    analysis_timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    player: Mapped["PlayerModel"] = relationship("PlayerModel")
    video: Mapped["VideoModel"] = relationship("VideoModel", back_populates="analysis")
    match: Mapped["MatchModel"] = relationship("MatchModel", back_populates="analysis")