from abc import ABC, abstractmethod
from typing import Optional, BinaryIO, Tuple, Dict, Any, List
from pathlib import Path

from ...domain.player import Player
from ...domain.video import Video, VideoStatus
from ...domain.analysis import Analysis


class IPlayerService(ABC):
    """
    Interface for Player business logic
    """
    
    @abstractmethod
    async def create_player(self, id: str, name: str, email: str, role: str = "player") -> Player:
        """Create a new player with validation"""
        pass
    
    @abstractmethod
    async def get_player_by_id(self, player_id: str) -> Player:
        """Get player by ID"""
        pass
    
    @abstractmethod
    async def get_player_by_email(self, email: str) -> Player:
        """Get player by email"""
        pass
    


class IVideoService(ABC):
    """Interface for video business operations"""
    
    @abstractmethod
    async def upload_video(
        self,
        file: BinaryIO,
        filename: str,
        content_type: str,
        file_size: int,
        player_id: str
    ) -> Video:
        """Upload and process video file"""
        pass
    
    @abstractmethod
    async def delete_video(self, video_id: int) -> bool:
        """Soft delete video"""
        pass
    
    @abstractmethod
    async def get_video_by_id(self, video_id: int) -> Optional[Video]:
        """Get video by ID"""
        pass
    
    @abstractmethod
    async def update_video_status(
        self,
        video_id: int,
        status: VideoStatus,
        error_message: Optional[str] = None
    ) -> Video:
        """Update video status"""
        pass
    
    @abstractmethod
    def validate_video_file(
        self, 
        filename: str, 
        file_size: int
    ) -> tuple[bool, Optional[str]]:
        """Validate video file"""
        pass
    
    @abstractmethod
    def get_allowed_formats(self) -> List[str]:
        """Get allowed formats"""
        pass
    
    @abstractmethod
    def get_max_file_size_mb(self) -> int:
        """Get max file size in MB"""
        pass


class IFileStorageService(ABC):
    """Interface for file storage operations"""
    
    @abstractmethod
    async def save_video(
        self, 
        file: BinaryIO, 
        original_filename: str, 
        player_id: str
    ) -> tuple[str, str]:
        """Save video file and return (storage_path, stored_filename)"""
        pass
    
    @abstractmethod
    async def delete_video(self, storage_path: str) -> bool:
        """Delete video file"""
        pass
    
    @abstractmethod
    def get_file_path(self, storage_path: str) -> Path:
        """Get absolute path for stored file"""
        pass
    
    @abstractmethod
    def file_exists(self, storage_path: str) -> bool:
        """Check if file exists"""
        pass


class IAnalysisService(ABC):
    """
    Interface for Analysis service
    Orchestrates analysis entity creation and AI processing
    """
    
    @abstractmethod
    async def create_analysis_for_video(
        self, 
        video_id: int, 
        player_id: str
    ) -> Analysis:
        """
        Create analysis entity chain for a video
        Creates Analysis, Match, and MatchPlayer entities
        
        Returns:
            Created Analysis entity
            
        Raises:
            VideoNotFoundException: If video doesn't exist
            AnalysisException: If creation fails
        """
        pass
    
    @abstractmethod
    async def store_analysis_results(
        self,
        match_id: int,
        ai_results: Dict[str, Any]
    ) -> None:
        """
        Store AI analysis results as SummaryMetrics
        
        Args:
            match_id: ID of the match
            ai_results: Dictionary with player stats from AI model
                Format: {
                    "player_1": {"hits": 245, "rallies": 45},
                    "player_2": {"hits": 238, "rallies": 45},
                    ...
                }
        """
        pass
    
    @abstractmethod
    async def complete_analysis(
        self,
        video_id: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Mark analysis as complete or failed
        Updates video status accordingly
        """
        pass
    
    @abstractmethod
    async def get_analysis_by_video(self, video_id: int) -> Analysis:
        """
        Get analysis for a video
        
        Raises:
            AnalysisNotFoundException: If no analysis found
        """
        pass


class IMatchService(ABC):
    """Interface for match business operations"""
    
    @abstractmethod
    async def get_match_overview(self, match_id: int) -> dict:
        """Get match overview with player statistics"""
        pass
    
    @abstractmethod
    async def get_match_statistics_by_set(self, match_id: int, set_number: int) -> dict:
        """Get match statistics filtered by set"""
        pass
    
    @abstractmethod
    async def get_hit_comparison_chart_data(self, match_id: int) -> dict:
        """Get data for hit comparison chart"""
        pass
    
    @abstractmethod
    async def get_player_hit_count(self, match_id: int, player_identifier: str) -> int:
        """Get hit count for specific player"""
        pass
