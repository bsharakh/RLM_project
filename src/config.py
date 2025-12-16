"""Configuration loader for RLM project."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RLM system."""
    
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    RLM_MODEL = os.getenv("RLM_MODEL", "gpt-4")
    SUBLM_MODEL = os.getenv("SUBLM_MODEL", "gpt-4o-mini")
    
    # Iteration Configuration
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
    MAX_DEPTH = int(os.getenv("MAX_DEPTH", "1"))
    
    # Metrics Configuration
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_MODEL = os.getenv("METRICS_MODEL", "gpt-4o-mini")
    
    # Logging Configuration
    ENABLE_LOGGING = os.getenv("ENABLE_LOGGING", "true").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    LOG_TO_CONSOLE = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in .env file"
            )
        return True