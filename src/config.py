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
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "7"))
    MAX_DEPTH = int(os.getenv("MAX_DEPTH", "1"))
    
    # Chunking Configuration
    ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING", "true").lower() == "true"
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "100000"))
    CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "map_based")
    
    # Metrics Configuration
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_MODEL = os.getenv("METRICS_MODEL", "gpt-4o-mini")
    
    # Early Stopping Configuration
    EARLY_STOP_THRESHOLD = float(os.getenv("EARLY_STOP_THRESHOLD", "0.8"))
    EARLY_STOP_METRIC = os.getenv("EARLY_STOP_METRIC", "overall")
    MIN_ACCURACY_THRESHOLD = float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.7"))
    
    # REPL Split Threshold (in characters)
    SPLIT_THRESHOLD = int(os.getenv("SPLIT_THRESHOLD", "50000"))
    
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