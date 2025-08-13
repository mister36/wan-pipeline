import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
        """Load WAN and FaceFusion models"""
        # Load models
        self.wan_model.load_model()
        self.facefusion_model.load_model()
        
        logger.info("Models loaded successfully")
    
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

@app.on_event("startup")
async def startup_event():
    """Clone FaceFusion and load models on startup"""
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
    headshot: UploadFile = File(..., description="Headshot image file"),
    prompt: str = Form(..., description="Text prompt for video generation")
):
    """
    Generate a vertical portrait video from a headshot and text prompt
    
    Pipeline:
    1. Generate initial near-still vertical video with WAN 2.2 T2V based on prompt
    2. Extract frame 0 as base still image
    3. Swap face from headshot into base still using FaceFusion headless mode
    4. Generate final video from face-swapped still using WAN I2V
    """
    
    # Validate inputs
    if not headshot.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Create temporary working directory
    with tempfile.TemporaryDirectory(dir=TEMP_DIR) as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save uploaded headshot
        headshot_path = temp_path / f"headshot_{headshot.filename}"
        with open(headshot_path, "wb") as buffer:
            shutil.copyfileobj(headshot.file, buffer)
        
        logger.info(f"Processing request - Prompt: {prompt}")
        
        # Step 1: Generate initial video with WAN 2.2 T2V
        initial_video_path = temp_path / "initial_video.mp4"
        pipeline.generate_initial_video(prompt, str(initial_video_path))
        
        # Step 2: Extract frame 0
        base_frame_path = temp_path / "base_frame.png"
        pipeline.extract_frame_zero(str(initial_video_path), str(base_frame_path))
        
        # Step 3: Swap faces with FaceFusion
        face_swapped_path = temp_path / "face_swapped.png"
        pipeline.swap_faces(str(headshot_path), str(base_frame_path), str(face_swapped_path))
        
        # Step 4: Generate final video with WAN 2.2 I2V
        final_video_path = temp_path / "final_video.mp4"
        pipeline.generate_final_video(str(face_swapped_path), prompt, str(final_video_path))
        
        logger.info("Video generation completed")
        
        # Return the generated video directly
        return FileResponse(
            path=str(final_video_path),
            media_type="video/mp4",
            filename="generated_video.mp4"
        )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
