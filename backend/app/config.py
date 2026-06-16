"""
TrafficGuard AI — Configuration Settings
All settings loaded from environment variables with sensible defaults.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # App
    APP_NAME: str = "TrafficGuard AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = "trafficguard-secret-change-in-production"

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR}/trafficguard.db"

    # Paths
    MODELS_DIR: Path = BASE_DIR / "models_weights"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    ANNOTATED_DIR: Path = BASE_DIR / "annotated"
    SAMPLE_IMAGES_DIR: Path = BASE_DIR / "sample_images"

    # YOLOv8 model filenames (placed in MODELS_DIR)
    MODEL_GENERAL: str = "yolov8m.pt"          # COCO — vehicles & persons
    MODEL_POSE: str = "yolov8m-pose.pt"        # Pose — occupant counting
    MODEL_HELMET: str = "helmet_best.pt"        # Community fine-tuned
    MODEL_SEATBELT: str = "seatbelt_best.pt"   # Community fine-tuned
    MODEL_PLATE: str = "plate_best.pt"         # License plate detection

    # Detection thresholds (per model, tunable)
    CONF_GENERAL: float = 0.45
    CONF_HELMET: float = 0.40
    CONF_SEATBELT: float = 0.40
    CONF_PLATE: float = 0.35
    NMS_IOU: float = 0.50

    # Violation detection parameters
    HELMET_OVERLAP_THRESHOLD: float = 0.25   # IoU: head region vs. helmet box
    TRIPLE_RIDING_PERSON_COUNT: int = 3       # Min persons on motorcycle
    STOP_LINE_Y_RATIO: float = 0.6           # Y position of stop line (0-1)
    PARKING_STATIC_FRAMES: int = 30          # Frames before marking as parked

    # OCR
    OCR_LANGUAGES: list = ["en"]
    PLATE_REGEX: str = r"[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}"

    # CORS
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000", "*"]

    # Pagination
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

# Ensure required directories exist
for d in [settings.MODELS_DIR, settings.UPLOADS_DIR, settings.ANNOTATED_DIR]:
    d.mkdir(parents=True, exist_ok=True)
