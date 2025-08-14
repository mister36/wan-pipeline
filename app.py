import os
import shutil
import uuid
import json
from pathlib import Path
from datetime import datetime

import torch
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import logging

from models.wan_model import WANModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WAN Media Pipeline API",
    description="API for generating images and videos using WAN 2.2 T2V with Instagirl lora and I2V models",
    version="2.0.0"
)

# Create necessary directories
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

JOBS_DIR = Path("jobs")
JOBS_DIR.mkdir(exist_ok=True)

VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models_cache")
MODELS_DIR.mkdir(exist_ok=True)

# Simple job storage - in a real system you'd use a database
jobs = {}

class JobManager:
    """Simple job management system"""
    
    @staticmethod
    def create_job(prompt: str, job_type: str, input_filename: str = None) -> str:
        """Create a new job and return job ID"""
        job_id = str(uuid.uuid4())
        
        job_data = {
            "id": job_id,
            "status": "queued",
            "type": job_type,  # "image" or "video"
            "prompt": prompt,
            "input_filename": input_filename,  # For I2V jobs
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None,
            "output_path": None  # Can be image or video path
        }
        
        jobs[job_id] = job_data
        
        # Save job to disk for persistence
        job_file = JOBS_DIR / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job_data, f, indent=2)
        
        return job_id
    
    @staticmethod
    def update_job_status(job_id: str, status: str, error: str = None, output_path: str = None):
        """Update job status"""
        if job_id not in jobs:
            return False
        
        jobs[job_id]["status"] = status
        if error:
            jobs[job_id]["error"] = error
        if output_path:
            jobs[job_id]["output_path"] = output_path
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

def download_instagirl_lora():
    """Download Instagirl lora if it doesn't exist"""
    lora_path = MODELS_DIR / "instagirl_lora.safetensors"
    
    if lora_path.exists():
        logger.info("Instagirl lora already exists, skipping download")
        return str(lora_path)
    
    logger.info("Downloading Instagirl lora...")
    lora_url = "https://huggingface.co/mister36/instagirl-v2-hinoise/resolve/main/Instagirlv2.0_hinoise.safetensors?download=true"
    
    try:
        response = requests.get(lora_url, stream=True)
        response.raise_for_status()
        
        with open(lora_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Instagirl lora downloaded successfully: {lora_path}")
        return str(lora_path)
        
    except Exception as e:
        logger.error(f"Failed to download Instagirl lora: {e}")
        raise RuntimeError(f"Failed to download Instagirl lora: {e}")

class MediaPipeline:
    def __init__(self):
        self.device = "cuda"
        logger.info(f"Using device: {self.device}")
        
        # Initialize model wrapper (lora path will be set during load_models)
        self.wan_model = None
        
    async def load_models(self):
        """Initialize models - WAN models will be loaded on-demand to save memory"""
        # Download Instagirl lora first
        lora_path = download_instagirl_lora()
        
        # Initialize WAN model with lora path
        self.wan_model = WANModel(device=self.device, instagirl_lora_path=lora_path)
        self.wan_model.load_model()
        
        logger.info("Model wrapper initialized successfully - WAN models will load on-demand")
    
    def generate_image_from_prompt(self, prompt: str, output_path: str) -> str:
        """Generate single image using WAN 2.2 T2V with Instagirl lora"""
        return self.wan_model.generate_single_frame_from_prompt(
            prompt=prompt,
            output_path=output_path,
            width=720,
            height=1280  # Vertical format for portrait shots
        )
    
    def generate_video_from_image(self, image_path: str, prompt: str, output_path: str) -> str:
        """Generate video from image and prompt using WAN 2.2 I2V"""
        return self.wan_model.generate_video_from_image(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            num_frames=81,
            fps=16
        )

# Initialize pipeline
pipeline = MediaPipeline()

def process_image_generation(job_id: str, prompt: str):
    """Process image generation for T2V job"""
    try:
        JobManager.update_job_status(job_id, "processing")
        logger.info(f"Starting image generation for job {job_id}")
        
        # Generate image with WAN 2.2 T2V and Instagirl lora
        output_image_path = IMAGES_DIR / f"{job_id}.png"
        pipeline.generate_image_from_prompt(prompt, str(output_image_path))
        
        # Clean up WAN models to free memory after generation
        pipeline.wan_model.cleanup_models()
        
        # Update job status
        JobManager.update_job_status(job_id, "completed", output_path=str(output_image_path))
        logger.info(f"Image generation completed for job {job_id}")
        
    except Exception as e:
        error_msg = f"Image generation failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        JobManager.update_job_status(job_id, "failed", error=error_msg)

def process_video_generation(job_id: str, image_path: str, prompt: str):
    """Process video generation for I2V job"""
    try:
        JobManager.update_job_status(job_id, "processing")
        logger.info(f"Starting video generation for job {job_id}")
        
        # Generate video with WAN 2.2 I2V
        output_video_path = VIDEOS_DIR / f"{job_id}.mp4"
        pipeline.generate_video_from_image(image_path, prompt, str(output_video_path))
        
        # Clean up WAN models to free memory after generation
        pipeline.wan_model.cleanup_models()
        
        # Update job status
        JobManager.update_job_status(job_id, "completed", output_path=str(output_video_path))
        logger.info(f"Video generation completed for job {job_id}")
        
        # Clean up input image
        if Path(image_path).exists() and TEMP_DIR.name in str(image_path):
            os.remove(image_path)
        
    except Exception as e:
        error_msg = f"Video generation failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        JobManager.update_job_status(job_id, "failed", error=error_msg)
        
        # Clean up input image on error
        if Path(image_path).exists() and TEMP_DIR.name in str(image_path):
            os.remove(image_path)

@app.on_event("startup")
async def startup_event():
    """Load jobs and initialize models on startup"""
    
    # Load existing jobs from disk
    JobManager.load_jobs_from_disk()
    logger.info(f"Loaded {len(jobs)} existing jobs")
    
    # Load models
    await pipeline.load_models()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "WAN Media Pipeline API is running"}

@app.post("/generate-image/")
async def generate_image(
    background_tasks: BackgroundTasks,
    prompt: str = Form(..., description="Text prompt for image generation")
):
    """
    Generate a single image from prompt using WAN 2.2 T2V with Instagirl lora
    
    Returns a job ID immediately while processing begins in the background.
    """
    
    # Validate inputs
    if len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Create job
    job_id = JobManager.create_job(prompt, "image")
    
    logger.info(f"Created image generation job {job_id} - Prompt: {prompt}")
    
    # Start processing in background
    background_tasks.add_task(process_image_generation, job_id, prompt)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Image generation job created and processing started. Use /job-status/{job_id} to check progress and /get-image/{job_id} to download when complete."
    }

@app.post("/generate-video-from-image/")
async def generate_video_from_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Input image file"),
    prompt: str = Form(..., description="Text prompt for video generation")
):
    """
    Generate a video from image and prompt using WAN 2.2 I2V
    
    Returns a job ID immediately while processing begins in the background.
    """
    
    # Validate inputs
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Create job
    job_id = JobManager.create_job(prompt, "video", image.filename)
    
    # Save uploaded image temporarily for processing
    image_path = TEMP_DIR / f"input_{job_id}_{image.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    
    logger.info(f"Created video generation job {job_id} - Prompt: {prompt}")
    
    # Start processing in background
    background_tasks.add_task(process_video_generation, job_id, str(image_path), prompt)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video generation job created and processing started. Use /job-status/{job_id} to check progress and /get-video/{job_id} to download when complete."
    }

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of an image or video generation job
    """
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "type": job["type"],
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
        "prompt": job["prompt"]
    }

@app.get("/get-image/{job_id}")
async def get_image(job_id: str):
    """
    Download the generated image for a completed image generation job
    """
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["type"] != "image":
        raise HTTPException(status_code=400, detail="Job is not an image generation job")
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Image is not ready. Job status: {job['status']}"
        )
    
    image_path = job.get("output_path")
    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=f"generated_image_{job_id}.png"
    )

@app.get("/get-video/{job_id}")
async def get_video(job_id: str):
    """
    Download the generated video for a completed video generation job
    """
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["type"] != "video":
        raise HTTPException(status_code=400, detail="Job is not a video generation job")
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Video is not ready. Job status: {job['status']}"
        )
    
    video_path = job.get("output_path")
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
                "type": job["type"],
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
