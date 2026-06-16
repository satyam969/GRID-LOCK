"""
TrafficGuard AI — FastAPI Application Entry Point
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db
from app.core.detector import model_registry
from app.api import upload, violations, analytics, vehicles, challans, cameras, settings as settings_api, reports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB + load models. Shutdown: cleanup."""
    logger.info("🚦 TrafficGuard AI starting up...")

    # Initialize database tables
    await init_db()
    logger.info("✅ Database initialized")

    # Load all detection models
    model_registry.load_all()
    logger.info("✅ All models loaded")

    # Ensure static dirs exist
    settings.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("TrafficGuard AI shutting down.")


app = FastAPI(
    title="TrafficGuard AI",
    description=(
        "Production-ready traffic violation detection system using YOLOv8 computer vision. "
        "Detects helmet non-compliance, seatbelt violations, triple riding, wrong-side driving, "
        "stop-line violations, red-light violations, and illegal parking."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (annotated evidence images) ───────────────────────────────────
app.mount("/annotated", StaticFiles(directory=str(settings.ANNOTATED_DIR)), name="annotated")
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")

# ── API Routes ─────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"
app.include_router(upload.router, prefix=API_PREFIX)
app.include_router(violations.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)
app.include_router(vehicles.router, prefix=API_PREFIX)
app.include_router(challans.router, prefix=API_PREFIX)
app.include_router(cameras.router, prefix=API_PREFIX)
app.include_router(settings_api.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    """System health check — returns model status and DB connectivity."""
    return JSONResponse({
        "status": "ok",
        "version": settings.APP_VERSION,
        "models_loaded": model_registry.status(),
        "database": "connected",
    })


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "TrafficGuard AI API",
        "docs": "/api/docs",
        "health": "/health",
        "version": settings.APP_VERSION,
    }
