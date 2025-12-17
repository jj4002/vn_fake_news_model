# routers/reports.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from services.supabase_client import SupabaseService  # ← Đổi từ SupabaseDB

logger = logging.getLogger(__name__)
router = APIRouter()

db = SupabaseService()  # ← Đổi từ SupabaseDB()

class ReportRequest(BaseModel):
    video_id: str
    reported_prediction: str
    reason: Optional[str] = None

@router.post("/report")
async def report_prediction(request: ReportRequest):
    """User report endpoint"""
    try:
        logger.info(f"📝 Report received for video: {request.video_id}")
        
        success = db.save_report(
            video_id=request.video_id,
            reported_prediction=request.reported_prediction,
            reason=request.reason
        )
        
        if success:
            return {
                "status": "success", 
                "message": "Report saved successfully"
            }
        else:
            raise HTTPException(
                status_code=400, 
                detail="Failed to save report. Video may not exist."
            )
            
    except Exception as e:
        logger.error(f"❌ Report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/pending")
async def get_pending_reports(limit: int = 50):
    """Get pending reports for admin review"""
    try:
        reports = db.get_disputed_videos(limit=limit)
        return {
            "reports": reports, 
            "count": len(reports)
        }
    except Exception as e:
        logger.error(f"❌ Get reports error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
