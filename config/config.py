from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Config:
    model_config: Dict[str, Any] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    temperature: float = 0.7
    top_k: int = 5
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = {
                "model_name": "vinai/phobert-base",
                "device": "cpu",
                "max_length": 256,
                "batch_size": 32
            } 