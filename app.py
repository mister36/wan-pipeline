import os
import shutil
import subprocess
import tempfile
import uuid
import json
from pathlib import Path
from datetime import datetime

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import logging

from models.wan_model import WANModel
from models.facefusion_model import FaceFusionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WAN Video Pipeline API",
    description="API for generating videos from headshots and prompts using WAN 2.2 T2V/I2V and FaceFusion",
    version="1.0.0"
)

# Create necessary directories
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)

# Simple job storage - in a real system you'd use a database
jobs = {}

class JobManager:
    """Simple job management system"""
    
    @staticmethod
    def create_job(prompt: str, headshot_filename: str) -> str:
        """Create a new job and return job ID"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "id": job_id,
            "status": "queued",
            "prompt": prompt,
            "headshot_filename": headshot_filename,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None,
            "video_path": None
        }
        
        jobs[job_id] = job_data
        
        # Save job to disk for persistence
        job_file = JOBS_DIR / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
        
        return job_id
    
    @staticmethod
    def update_job_status(job_id: str, status: str, error: str = None, video_path: str = None):
        """Update job status"""
        if job_id not in jobs:
            return False
        
        jobs[job_id]["status"] = status
        if error:
            jobs[job_id]["error"] = error
        if video_path:
            jobs[job_id]["video_path"] = video_path
        if status == "completed":
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        # Save updated job to disk
        job_file = JOBS_DIR / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(jobs[job_id], f, indent=2)
        
        return True
    
    @staticmethod
    def get_job(job_id: str) -> dict:
        """Get job details"""
        return jobs.get(job_id)
    
    @staticmethod
    def load_jobs_from_disk():
        """Load existing jobs from disk on startup"""
        for job_file in JOBS_DIR.glob("*.json"):
            try:
                with open(job_file, "r") as f:
                    job_data = json.load(f)
                    jobs[job_data["id"]] = job_data
            except Exception as e:
                logger.warning(f"Could not load job file {job_file}: {e}")

def clone_facefusion():
    """Clone FaceFusion repository if it doesn't exist"""
    facefusion_path = Path("facefusion")
    
    if facefusion_path.exists():
        logger.info("FaceFusion directory already exists, skipping clone")
        return
    
    logger.info("Cloning FaceFusion repository...")
    try:
        result = subprocess.run([
            "git", "clone", 
            "https://github.com/facefusion/facefusion.git",
            str(facefusion_path)
        ], check=True, capture_output=True, text=True)
        
        logger.info("FaceFusion cloned successfully")
        logger.info(f"Clone output: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone FaceFusion: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"Failed to clone FaceFusion repository: {e.stderr}")

class VideoPipeline:
    def __init__(self):
        self.device = "cuda"
        logger.info(f"Using device: {self.device}")
        
        # Initialize model wrappers
        self.wan_model = WANModel(device=self.device)
        self.facefusion_model = FaceFusionModel()
        
    async def load_models(self):
        """Initialize models - WAN models will be loaded on-demand to save memory"""
        # Initialize models (no heavy loading at startup)
        self.wan_model.load_model()
        self.facefusion_model.load_model()
        
        logger.info("Model wrappers initialized successfully - WAN models will load on-demand")
    
    def generate_initial_video(self, prompt: str, output_path: str) -> str:
        """Generate initial near-still vertical video using WAN 2.2 T2V from prompt"""
        return self.wan_model.generate_video_from_prompt(
            prompt=prompt,
            output_path=output_path,
            width=720,
            height=1280,  # Vertical format for portrait shots
            num_frames=81,
            fps=16
        )
    
    def extract_frame_zero(self, video_path: str, frame_output_path: str) -> str:
        """Extract frame 0 from the generated video"""
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError("Could not read frame from video")
        
        cv2.imwrite(frame_output_path, frame)
        cap.release()
        
        logger.info(f"Frame 0 extracted to: {frame_output_path}")
        return frame_output_path
    
    def swap_faces(self, headshot_path: str, base_frame_path: str, output_path: str) -> str:
        """Swap face from headshot into the base frame using FaceFusion"""
        return self.facefusion_model.swap_faces(
            source_image_path=headshot_path,
            target_image_path=base_frame_path,
            output_path=output_path
        )
    
    def generate_final_video(self, face_swapped_frame_path: str, prompt: str, output_path: str) -> str:
        """Generate final video from face-swapped frame using WAN i2v"""
        return self.wan_model.generate_video_from_image(
            image_path=face_swapped_frame_path,
            prompt=prompt,
            output_path=output_path,
            num_frames=81,
            fps=16
        )

# Initialize pipeline
pipeline = VideoPipeline()

def process_video_generation(job_id: str, headshot_path: str, prompt: str):
    """Process video generation for a job"""
    try:
        JobManager.update_job_status(job_id, "processing")
        logger.info(f"Starting video generation for job {job_id}")
        
        # Create working directory for this job
        job_temp_dir = TEMP_DIR / job_id
        job_temp_dir.mkdir(exist_ok=True)
        
        # Step 1: Generate initial video with WAN 2.2 T2V
        initial_video_path = job_temp_dir / "initial_video.mp4"
        pipeline.generate_initial_video(prompt, str(initial_video_path))
        
        # Step 2: Extract frame 0
        base_frame_path = job_temp_dir / "base_frame.png"
        pipeline.extract_frame_zero(str(initial_video_path), str(base_frame_path))
        
        # Step 3: Swap faces with FaceFusion
        face_swapped_path = job_temp_dir / "face_swapped.png"
        pipeline.swap_faces(headshot_path, str(base_frame_path), str(face_swapped_path))
        
        # Step 4: Generate final video with WAN 2.2 I2V
        final_video_path = VIDEOS_DIR / f"{job_id}.mp4"
        pipeline.generate_final_video(str(face_swapped_path), prompt, str(final_video_path))
        
        # Clean up WAN models to free memory after generation
        pipeline.wan_model.cleanup_models()
        
        # Update job status
        JobManager.update_job_status(job_id, "completed", video_path=str(final_video_path))
        logger.info(f"Video generation completed for job {job_id}")
        
        # Clean up temporary job directory
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        
    except Exception as e:
        error_msg = f"Video generation failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        JobManager.update_job_status(job_id, "failed", error=error_msg)
        
        # Clean up on error
        job_temp_dir = TEMP_DIR / job_id
        if job_temp_dir.exists():
            shutil.rmtree(job_temp_dir, ignore_errors=True)

@app.on_event("startup")
async def startup_event():
    """Clone FaceFusion, load jobs, and load models on startup"""
    # Load existing jobs from disk
    JobManager.load_jobs_from_disk()
    logger.info(f"Loaded {len(jobs)} existing jobs")
    
    # Clone FaceFusion repository if needed
    clone_facefusion()
    
    # Load models
    await pipeline.load_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "WAN Video Pipeline API is running"}

@app.post("/generate-video/")
async def generate_video(
    background_tasks: BackgroundTasks,
    headshot: UploadFile = File(..., description="Headshot image file"),
    prompt: str = Form(..., description="Text prompt for video generation")
):
    """
    Start video generation job and return job ID immediately
    
    Returns a job ID immediately while processing begins in the background.
    """
    
    # Validate inputs
    if not headshot.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Create job
    job_id = JobManager.create_job(prompt, headshot.filename)
    
    # Save uploaded headshot permanently for processing
    headshot_path = TEMP_DIR / f"headshot_{job_id}_{headshot.filename}"
    with open(headshot_path, "wb") as buffer:
        shutil.copyfileobj(headshot.file, buffer)
    
    logger.info(f"Created job {job_id} - Prompt: {prompt}")
    
    # Start processing in background
    background_tasks.add_task(process_video_generation, job_id, str(headshot_path), prompt)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video generation job created and processing started. Use /job-status/{job_id} to check progress and /get-video/{job_id} to download when complete."
    }

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a video generation job
    """
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
        "prompt": job["prompt"]
    }

@app.get("/get-video/{job_id}")
async def get_video(job_id: str):
    """
    Download the generated video for a completed job
    """
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Video is not ready. Job status: {job['status']}"
        )
    
    video_path = job.get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"generated_video_{job_id}.mp4"
    )

@app.get("/jobs")
async def list_jobs():
    """
    List all jobs (for debugging/admin purposes)
    """
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
                "prompt": job["prompt"][:50] + "..." if len(job["prompt"]) > 50 else job["prompt"]
            }
            for job_id, job in jobs.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
