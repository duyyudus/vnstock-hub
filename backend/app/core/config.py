from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/vnstock_hub"
    
    # API
    api_v1_prefix: str = "/api/v1"
    
    # CORS
    cors_origins: str = '["http://localhost:5173","http://localhost:3000"]'
    
    # Stock Fetching
    vn100_limit: int = 100
    vn30_limit: int = 30
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from JSON string."""
        return json.loads(self.cors_origins)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
