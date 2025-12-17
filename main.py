# main.py
from dotenv import load_dotenv
import os

# ✅ LOAD .ENV TRƯỚC KHI IMPORT BẤT CỨ THỨ GÌ
load_dotenv()

# Debug: Print để check
print("="*50)
print("Environment Variables:")
print(f"SUPABASE_URL: {os.getenv('SUPABASE_URL', 'NOT SET')[:30]}...")
print(f"SUPABASE_KEY: {os.getenv('SUPABASE_KEY', 'NOT SET')[:30]}...")
print(f"MODEL_PATH: {os.getenv('MODEL_PATH', 'NOT SET')}")
print("="*50)

# Sau đó mới import routers
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routers import predict, media, reports
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TikTok Fake News Detection API",
    version="2.0.0",
    description="AI-powered fake news detection for TikTok videos"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(media.router, prefix="/api/v1", tags=["Media"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])

@app.get("/")
def root():
    return {
        "service": "TikTok Fake News Detection",
        "version": "2.2.2",
        "status": "online",
        "endpoints": {
            "predict": "/api/v1/predict",
            "process_media": "/api/v1/process-media",
            "report": "/api/v1/report",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    model_exists = os.path.exists(os.getenv("MODEL_PATH", "./models/han_rag_model.onnx"))
    supabase_connected = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"))
    
    return {
        "status": "healthy",
        "model": "loaded" if model_exists else "missing",
        "database": "connected" if supabase_connected else "not configured"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
