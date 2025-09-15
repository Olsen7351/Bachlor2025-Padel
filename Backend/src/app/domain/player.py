from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Player:
    """Domain model for Player entity"""
    id: Optional[int]
    name: str
    email: str
    password: str
    role: str = "player"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None