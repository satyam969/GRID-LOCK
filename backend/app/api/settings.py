"""
Settings API — Dynamically adjust AI thresholds and system configurations.
"""
from fastapi import APIRouter, HTTPException
import json
from pathlib import Path
from app.schemas import AppSettings

router = APIRouter(prefix="/settings", tags=["Settings"])

SETTINGS_FILE = Path(__file__).parent.parent.parent / "settings.json"

DEFAULT_SETTINGS = {
    "conf_general": 0.45,
    "conf_helmet": 0.40,
    "conf_seatbelt": 0.40,
    "conf_plate": 0.35,
    "enable_clahe": False,
    "auto_generate_challans": False,
    "email_alerts": True,
    "admin_email": "admin@trafficguard.ai"
}

def load_settings():
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except:
        return DEFAULT_SETTINGS

def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=4)


@router.get("", response_model=AppSettings)
async def get_settings():
    """Retrieve current system settings."""
    return load_settings()


@router.put("", response_model=AppSettings)
async def update_settings(settings: AppSettings):
    """Update system settings (saves to JSON to persist across reboots)."""
    data = settings.model_dump()
    save_settings(data)
    
    # We could also dynamically update app.config.settings here if needed
    from app.config import settings as app_cfg
    app_cfg.CONF_GENERAL = data["conf_general"]
    app_cfg.CONF_HELMET = data["conf_helmet"]
    app_cfg.CONF_SEATBELT = data["conf_seatbelt"]
    app_cfg.CONF_PLATE = data["conf_plate"]
    
    return data
