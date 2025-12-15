from abc import ABC, abstractmethod
from typing import Optional, List, BinaryIO, Dict
from pathlib import Path

from ...domain.player import Player
from ...domain.video import Video, VideoStatus
from ...domain.analysis import Analysis


class IPlayerService(ABC):
    """Interface for Player business operations"""
    
    @abstractmethod
    async def create_player(
        self, 
        id: str,
        name: str, 
        email: str, 
        role: str = "player"
    ) -> Player:
        """Create a new player"""
        pass
    
    @abstractmethod
    async def get_player_by_id(self, player_id: str) -> Player:
        """Get player by ID"""
        pass
    
    @abstractmethod
    async def get_all_players(self) -> List[Player]:
        """Get all players"""
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
        """
        Save video file to storage
        
        Returns:
            Tuple of (storage_path, stored_filename)
        """
        pass
    
    @abstractmethod
    async def delete_video(self, storage_path: str) -> bool:
        """Delete video file from storage"""
        pass
    
    @abstractmethod
    def get_file_path(self, storage_path: str) -> Path:
        """Get absolute path for a stored file"""
        pass
    
    @abstractmethod
    def file_exists(self, storage_path: str) -> bool:
        """Check if file exists in storage"""
        pass


class IVideoService(ABC):
    """Interface for Video business operations"""
    
    @abstractmethod
    async def upload_video(
        self, 
        file: BinaryIO, 
        filename: str, 
        content_type: str,
        file_size: int,
        player_id: str,
        court_number: Optional[int] = None
    ) -> Video:
        """Upload and process a video file"""
        pass
    
    @abstractmethod
    async def delete_video(self, video_id: int) -> bool:
        """Soft delete a video"""
        pass
    
    @abstractmethod
    def get_allowed_formats(self) -> List[str]:
        """Get list of allowed video formats"""
        pass
    
    @abstractmethod
    def get_max_file_size_mb(self) -> int:
        """Get maximum allowed file size in MB"""
        pass
    
    @abstractmethod
    def get_video_path(self, video: Video) -> Path:
        """Get the full path to a video file"""
        pass


class IMatchService(ABC):
    """Interface for Match business operations"""
    
    @abstractmethod
    async def get_match_overview(self, match_id: int) -> Dict:
        """
        Get match overview with player statistics
        Implements UC-04 Success Scenario S1
        """
        pass
    
    @abstractmethod
    async def get_player_hit_count(
        self, 
        match_id: int, 
        player_identifier: str
    ) -> int:
        """Get hit count for a specific player in a match"""
        pass


class IRallyService(ABC):
    """Interface for Rally business operations"""
    
    @abstractmethod
    async def get_rally_analysis(self, match_id: int) -> Dict:
        """
        UC-08 S1: Get rally analysis for a match
        
        Returns overview of rally lengths including:
        - Total number of rallies
        - Average rally duration
        - Min/Max duration
        - Distribution of rally lengths
        """
        pass
    
    @abstractmethod
    async def get_rally_duration_distribution(self, match_id: int) -> Dict:
        """Get rally duration distribution for visualization"""
        pass
    
    @abstractmethod
    async def get_match_total_rallies(self, match_id: int) -> int:
        """Get total rally count for a match"""
        pass


class IHeatmapService(ABC):
    """Interface for Heatmap business operations"""
    
    @abstractmethod
    async def get_player_heatmap(
        self, 
        match_id: int, 
        player_identifier: str
    ):
        """
        UC-02 S1: Get heatmap for a specific player in a match
        
        Returns:
            PlayerHeatmapResponse with base64 encoded heatmap
        """
        pass
    
    @abstractmethod
    async def get_player_heatmap_raw(
        self, 
        match_id: int, 
        player_identifier: str
    ) -> bytes:
        """Get raw heatmap bytes for direct image response"""
        pass
    
    @abstractmethod
    async def get_heatmap_comparison(
        self, 
        match_id: int,
        player_identifiers: Optional[List[str]] = None
    ):
        """UC-02 S2: Get multiple heatmaps for comparison"""
        pass
    
    @abstractmethod
    async def get_available_heatmap_players(self, match_id: int) -> List[str]:
        """Get list of players who have heatmap data available"""
        pass


class IAnalysisService(ABC):
    """Interface for Analysis business operations"""
    
    @abstractmethod
    async def create_analysis_for_video(
        self, 
        video_id: int, 
        player_id: str
    ) -> Analysis:
        """Create analysis entity chain for a video"""
        pass
    
    @abstractmethod
    async def run_ml_analysis(
        self,
        video_id: int,
        video_path: Path,
        court_number: int,
        fps: float = 30.0
    ):
        """Run ML inference on a video"""
        pass
    
    @abstractmethod
    async def store_analysis_results(
        self,
        match_id: int,
        ml_result
    ) -> None:
        """Store ML analysis results in the database"""
        pass
    
    @abstractmethod
    async def complete_analysis(
        self,
        video_id: int,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Mark analysis as complete (or failed)"""
        pass
    