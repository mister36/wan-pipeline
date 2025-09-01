import os
import shutil
import uuid
import json
import zipfile
import tempfile
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
    description="API for generating images and videos using WAN 2.2 T2V with Instagirl lora and I2V with Lightning LoRAs",
    version="2.1.0"
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
            "type": job_type,  # "image", "video", or "image_batch"
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
    def create_batch_job(prompts: list[str], job_type: str = "image_batch") -> str:
        """Create a new batch job for multiple prompts and return job ID"""
        job_id = str(uuid.uuid4())
        
        # Create individual image entries for each prompt
        images = []
        for i, prompt in enumerate(prompts):
            images.append({
                "index": i,
                "prompt": prompt,
                "status": "queued",
                "output_path": None,
                "error": None
            })
        
        job_data = {
            "id": job_id,
            "status": "queued",
            "type": job_type,
            "prompts": prompts,  # Store original prompts list
            "images": images,  # Individual image tracking
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None,
            "total_images": len(prompts),
            "completed_images": 0,
            "failed_images": 0
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
    def update_batch_image_status(job_id: str, image_index: int, status: str, output_path: str = None, error: str = None):
        """Update status of individual image in batch job"""
        if job_id not in jobs:
            return False
        
        job = jobs[job_id]
        if job["type"] != "image_batch" or image_index >= len(job["images"]):
            return False
        
        # Update individual image status
        job["images"][image_index]["status"] = status
        if output_path:
            job["images"][image_index]["output_path"] = output_path
        if error:
            job["images"][image_index]["error"] = error
        
        # Update counters
        if status == "completed":
            job["completed_images"] += 1
        elif status == "failed":
            job["failed_images"] += 1
        
        # Check if entire batch is complete
        total_processed = job["completed_images"] + job["failed_images"]
        if total_processed == job["total_images"]:
            if job["failed_images"] == 0:
                job["status"] = "completed"
            elif job["completed_images"] == 0:
                job["status"] = "failed"
            else:
                job["status"] = "partially_completed"
            job["completed_at"] = datetime.now().isoformat()
        elif status == "processing" and job["status"] == "queued":
            job["status"] = "processing"
        
        # Save updated job to disk
        job_file = JOBS_DIR / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job, f, indent=2)
        
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
        # Download Instagirl LoRA (Lightning LoRAs will be downloaded on-demand from HuggingFace)
        instagirl_lora_path = download_instagirl_lora()
        
        # Initialize WAN model with Instagirl lora path
        self.wan_model = WANModel(
            device=self.device, 
            instagirl_lora_path=instagirl_lora_path
        )
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
    
    def generate_video_from_image(self, image_path: str, prompt: str, output_path: str, duration_seconds: float = 5.0) -> str:
        """Generate video from image and prompt using WAN 2.2 I2V"""
        # Calculate number of frames based on duration and fixed fps of 16
        fps = 16
        num_frames = int(duration_seconds * fps)
        
        return self.wan_model.generate_video_from_image(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            num_frames=num_frames,
            fps=fps
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
        
        # Update job status
        JobManager.update_job_status(job_id, "completed", output_path=str(output_image_path))
        logger.info(f"Image generation completed for job {job_id}")
        
    except Exception as e:
        error_msg = f"Image generation failed: {str(e)}"
        logger.error(f"Job {job_id} failed: {error_msg}")
        JobManager.update_job_status(job_id, "failed", error=error_msg)

def process_batch_image_generation(job_id: str, prompts: list[str]):
    """Process batch image generation for multiple prompts - model stays loaded between generations"""
    job = JobManager.get_job(job_id)
    if not job:
        logger.error(f"Batch job {job_id} not found")
        return
    
    logger.info(f"Starting batch image generation for job {job_id} with {len(prompts)} prompts")
    
    try:
        # Ensure T2V model is loaded for batch processing
        pipeline.wan_model._load_t2v_model()
        
        # Process each prompt sequentially while keeping model loaded
        for i, prompt in enumerate(prompts):
            try:
                logger.info(f"Processing image {i+1}/{len(prompts)} for job {job_id}: {prompt}")
                
                # Update individual image status to processing
                JobManager.update_batch_image_status(job_id, i, "processing")
                
                # Generate image with WAN 2.2 T2V and Instagirl lora
                output_image_path = IMAGES_DIR / f"{job_id}_image_{i}.png"
                pipeline.generate_image_from_prompt(prompt, str(output_image_path))
                
                # Update individual image status to completed
                JobManager.update_batch_image_status(job_id, i, "completed", output_path=str(output_image_path))
                logger.info(f"Image {i+1}/{len(prompts)} completed for job {job_id}")
                
            except Exception as e:
                error_msg = f"Image {i+1} generation failed: {str(e)}"
                logger.error(f"Job {job_id}, image {i+1} failed: {error_msg}")
                JobManager.update_batch_image_status(job_id, i, "failed", error=error_msg)
                # Continue with next image even if one fails
        
        logger.info(f"Batch image generation completed for job {job_id}.")
        
    except Exception as e:
        error_msg = f"Batch image generation failed: {str(e)}"
        logger.error(f"Batch job {job_id} failed: {error_msg}")
        JobManager.update_job_status(job_id, "failed", error=error_msg)

def process_video_generation(job_id: str, image_path: str, prompt: str, duration_seconds: float = 5.0):
    """Process video generation for I2V job"""
    try:
        JobManager.update_job_status(job_id, "processing")
        logger.info(f"Starting video generation for job {job_id}")
        
        # Generate video with WAN 2.2 I2V
        output_video_path = VIDEOS_DIR / f"{job_id}.mp4"
        pipeline.generate_video_from_image(image_path, prompt, str(output_video_path), duration_seconds)
        
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

@app.post("/generate-images-batch/")
async def generate_images_batch(
    background_tasks: BackgroundTasks,
    prompts: str = Form(..., description="JSON array of text prompts for batch image generation")
):
    """
    Generate multiple images from prompts using WAN 2.2 T2V with Instagirl lora
    
    The model stays loaded between generations for efficiency.
    Returns a job ID immediately while processing begins in the background.
    """
    
    # Parse prompts from JSON string
    try:
        prompts_list = json.loads(prompts)
        if not isinstance(prompts_list, list):
            raise ValueError("Prompts must be a JSON array")
        if len(prompts_list) == 0:
            raise ValueError("At least one prompt is required")
        if len(prompts_list) > 20:  # Reasonable limit to prevent abuse
            raise ValueError("Maximum 20 prompts allowed per batch")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid prompts format: {str(e)}")
    
    # Validate each prompt
    for i, prompt in enumerate(prompts_list):
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            raise HTTPException(status_code=400, detail=f"Prompt {i+1} cannot be empty")
    
    # Create batch job
    job_id = JobManager.create_batch_job(prompts_list, "image_batch")
    
    logger.info(f"Created batch image generation job {job_id} with {len(prompts_list)} prompts")
    
    # Start processing in background
    background_tasks.add_task(process_batch_image_generation, job_id, prompts_list)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "total_images": len(prompts_list),
        "message": f"Batch image generation job created with {len(prompts_list)} prompts. Use /job-status/{job_id} to check progress and /get-batch-images/{job_id} to download when complete."
    }

@app.post("/generate-video-from-image/")
async def generate_video_from_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Input image file"),
    prompt: str = Form(..., description="Text prompt for video generation"),
    duration_seconds: float = Form(5.0, description="Duration of the generated video in seconds (default: 5.0)", ge=0.5, le=30.0)
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
    
    logger.info(f"Created video generation job {job_id} - Prompt: {prompt}, Duration: {duration_seconds}s")
    
    # Start processing in background
    background_tasks.add_task(process_video_generation, job_id, str(image_path), prompt, duration_seconds)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video generation job created and processing started. Use /job-status/{job_id} to check progress and /get-video/{job_id} to download when complete."
    }

@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of an image, video, or batch generation job
    """
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Handle batch jobs
    if job["type"] == "image_batch":
        return {
            "job_id": job_id,
            "status": job["status"],
            "type": job["type"],
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),
            "error": job.get("error"),
            "total_images": job["total_images"],
            "completed_images": job["completed_images"],
            "failed_images": job["failed_images"],
            "progress_percentage": round((job["completed_images"] + job["failed_images"]) / job["total_images"] * 100, 1),
            "images": job["images"]  # Individual image status
        }
    
    # Handle single image/video jobs
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

@app.get("/get-batch-image/{job_id}/{image_index}")
async def get_batch_image(job_id: str, image_index: int):
    """
    Download a specific image from a batch generation job
    """
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["type"] != "image_batch":
        raise HTTPException(status_code=400, detail="Job is not a batch image generation job")
    
    if image_index < 0 or image_index >= len(job["images"]):
        raise HTTPException(status_code=400, detail="Invalid image index")
    
    image_info = job["images"][image_index]
    if image_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Image {image_index} is not ready. Status: {image_info['status']}"
        )
    
    image_path = image_info.get("output_path")
    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=f"batch_{job_id}_image_{image_index}.png"
    )

@app.get("/get-batch-images/{job_id}")
async def get_batch_images(job_id: str):
    """
    Download all completed images from a batch generation job as a zip file
    """
    job = JobManager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["type"] != "image_batch":
        raise HTTPException(status_code=400, detail="Job is not a batch image generation job")
    
    # Get all completed images
    completed_images = [img for img in job["images"] if img["status"] == "completed" and img["output_path"]]
    
    if len(completed_images) == 0:
        raise HTTPException(status_code=400, detail="No completed images available for download")
    
    # Create temporary zip file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        zip_path = tmp_zip.name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img in completed_images:
                image_path = Path(img["output_path"])
                if image_path.exists():
                    # Add file to zip with descriptive name
                    zip_filename = f"image_{img['index']}_{image_path.name}"
                    zipf.write(image_path, zip_filename)
        
        # Return zip file and schedule cleanup
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=f"batch_images_{job_id}.zip",
            background=lambda: os.unlink(zip_path) if os.path.exists(zip_path) else None
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

@app.post("/admin/cleanup-models")
async def cleanup_models():
    """
    Manually cleanup models to free GPU memory (admin endpoint)
    
    This endpoint allows administrators to manually clean up WAN models 
    when they want to free GPU memory. Models will be automatically 
    reloaded when needed for the next generation.
    """
    try:
        if pipeline.wan_model:
            current_model = pipeline.wan_model.get_current_model()
            pipeline.wan_model.cleanup_models()
            
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.memory_allocated() / 1024**3
                return {
                    "status": "success",
                    "message": "Models cleaned up successfully",
                    "previous_model": current_model,
                    "gpu_memory_allocated_gb": round(gpu_memory_gb, 2)
                }
            else:
                return {
                    "status": "success", 
                    "message": "Models cleaned up successfully",
                    "previous_model": current_model,
                    "gpu_memory_allocated_gb": 0
                }
        else:
            return {
                "status": "info",
                "message": "No models are currently loaded",
                "gpu_memory_allocated_gb": 0 if not torch.cuda.is_available() else round(torch.cuda.memory_allocated() / 1024**3, 2)
            }
    except Exception as e:
        logger.error(f"Failed to cleanup models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup models: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
