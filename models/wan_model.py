"""
WAN 2.2 Text-to-Video and Image-to-Video Model Integration

This module provides integration with both WAN 2.2 T2V and I2V models for generating videos from prompts and images.
"""

import torch
import logging
import gc
from typing import Optional
from pathlib import Path
from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class WANModel:
    """Wrapper for WAN 2.2 Text-to-Video and Image-to-Video models with memory-efficient loading"""
    
    def __init__(self, device: str = "cuda", instagirl_lora_path: str = None):
        self.device = device
        self.t2v_pipeline = None  # Text-to-Video pipeline
        self.i2v_pipeline = None  # Image-to-Video pipeline
        self.vae = None
        self.t2v_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        self.i2v_model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        self.instagirl_lora_path = instagirl_lora_path
        self.dtype = torch.bfloat16
        self.current_model = None  # Track which model is currently loaded
        
    def load_model(self):
        """Initialize model - no longer pre-loads models to save memory"""
        logger.info("WAN Model wrapper initialized - models will be loaded on-demand")
        
    def _cleanup_memory(self):
        """Clean up GPU memory by unloading models and running garbage collection"""
        if self.t2v_pipeline is not None:
            del self.t2v_pipeline
            self.t2v_pipeline = None
            
        if self.i2v_pipeline is not None:
            del self.i2v_pipeline
            self.i2v_pipeline = None
            
        if self.vae is not None:
            del self.vae
            self.vae = None
            
        self.current_model = None
        
        # Force garbage collection and clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU memory cleared. Current allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def _load_t2v_model(self):
        """Load T2V model and VAE, unloading I2V if necessary"""
        if self.current_model == "t2v" and self.t2v_pipeline is not None:
            return  # Already loaded
            
        logger.info("Loading WAN 2.2 T2V model...")
        
        # Unload I2V model if it's currently loaded
        if self.current_model == "i2v":
            logger.info("Unloading I2V model to free memory...")
            if self.i2v_pipeline is not None:
                del self.i2v_pipeline
                self.i2v_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
        
        # Load VAE for T2V model if not already loaded
        if self.vae is None:
            logger.info("Loading VAE...")
            self.vae = AutoencoderKLWan.from_pretrained(
                self.t2v_model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        # Load T2V pipeline
        logger.info("Loading T2V pipeline...")
        self.t2v_pipeline = WanPipeline.from_pretrained(
            self.t2v_model_id, 
            vae=self.vae,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        )
        self.t2v_pipeline.to(self.device)
        
        # Load Instagirl lora for T2V if path is provided
        if self.instagirl_lora_path:
            logger.info("Loading Instagirl lora...")
            self.t2v_pipeline.load_lora_weights(self.instagirl_lora_path)
        
        self.current_model = "t2v"
        
        logger.info(f"T2V model loaded. GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def _load_i2v_model(self):
        """Load I2V model, unloading T2V if necessary"""
        if self.current_model == "i2v" and self.i2v_pipeline is not None:
            return  # Already loaded
            
        logger.info("Loading WAN 2.2 I2V model...")
        
        # Unload T2V model if it's currently loaded
        if self.current_model == "t2v":
            logger.info("Unloading T2V model to free memory...")
            if self.t2v_pipeline is not None:
                del self.t2v_pipeline
                self.t2v_pipeline = None
            # Keep VAE as it might be useful for other operations
            gc.collect()
            torch.cuda.empty_cache()
        
        # Load I2V pipeline
        logger.info("Loading I2V pipeline...")
        self.i2v_pipeline = WanImageToVideoPipeline.from_pretrained(
            self.i2v_model_id, 
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        )
        self.i2v_pipeline.to(self.device)
        self.current_model = "i2v"
        
        logger.info(f"I2V model loaded. GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def generate_video_from_prompt(
        self, 
        prompt: str, 
        output_path: str,
        width: int = 720,
        height: int = 1280,  # Vertical format for portrait shots
        num_frames: int = 9,  # Reduced from 81 for faster generation when only extracting frame 0
        fps: int = 16
    ) -> str:
        """
        Generate a video from a text prompt using WAN 2.2 T2V
        
        Args:
            prompt: Text description for video generation
            output_path: Path to save the generated video
            width: Video width in pixels
            height: Video height in pixels
            num_frames: Number of frames to generate
            fps: Frames per second
            
        Returns:
            Path to the generated video file
        """
        # Load T2V model on-demand
        self._load_t2v_model()
            
        logger.info(f"Generating video from prompt: '{prompt}' using WAN 2.2 T2V")
        
        # Define negative prompt (from WAN example)
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        with torch.no_grad():
            output = self.t2v_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
                num_inference_steps=40,
            ).frames[0]
            
            # Export to video
            export_to_video(output, output_path, fps=fps)
        
        logger.info(f"Video generated and saved to: {output_path}")
        return output_path
    
    def generate_single_frame_from_prompt(
        self, 
        prompt: str, 
        output_path: str,
        width: int = 720,
        height: int = 1280  # Vertical format for portrait shots
    ) -> str:
        """
        Generate a single frame from a text prompt using WAN 2.2 T2V with Instagirl lora
        
        Args:
            prompt: Text description for image generation
            output_path: Path to save the generated image
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Path to the generated image file
        """
        # Load T2V model on-demand
        self._load_t2v_model()
            
        logger.info(f"Generating single frame from prompt: '{prompt}' using WAN 2.2 T2V with Instagirl lora")
        
        # Define negative prompt (from WAN example)
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        with torch.no_grad():
            output = self.t2v_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=1,  # Generate only one frame
                guidance_scale=4.0,
                guidance_scale_2=3.0,
                num_inference_steps=40,
            ).frames[0]
            
            # Save the single frame as an image
            frame = output[0]  # Get the first (and only) frame
            
            # Convert numpy array to PIL Image if necessary
            if isinstance(frame, np.ndarray):
                # Ensure the array is in the correct format (0-255, uint8)
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                frame = Image.fromarray(frame)
            
            frame.save(output_path)
        
        logger.info(f"Single frame generated and saved to: {output_path}")
        return output_path
    
    def generate_video_from_image(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        num_frames: int = 81,
        fps: int = 16
    ) -> str:
        """
        Generate a video from an image and prompt using WAN 2.2 I2V
        
        Args:
            image_path: Path to the input image
            prompt: Text description for video generation
            output_path: Path to save the generated video
            num_frames: Number of frames to generate
            fps: Frames per second
            
        Returns:
            Path to the generated video file
        """
        # Load I2V model on-demand
        self._load_i2v_model()
            
        logger.info(f"Generating video from image: {image_path} with prompt: '{prompt}' using WAN 2.2 I2V")
        
        # Load and process the input image
        image = load_image(image_path)
        
        # Calculate optimal dimensions based on WAN 2.2 requirements
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = self.i2v_pipeline.vae_scale_factor_spatial * self.i2v_pipeline.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        
        # Resize image to calculated dimensions
        image = image.resize((width, height))
        
        # Define negative prompt (from WAN example)
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        generator = torch.Generator(device=self.device).manual_seed(0)
        
        with torch.no_grad():
            output = self.i2v_pipeline(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=3.5,
                num_inference_steps=40,
                generator=generator,
            ).frames[0]
            
            # Export to video
            export_to_video(output, output_path, fps=fps)
        
        logger.info(f"Video generated and saved to: {output_path}")
        return output_path
    

    
    def cleanup_models(self):
        """Public method to clean up all models and free memory"""
        logger.info("Cleaning up WAN models to free memory...")
        self._cleanup_memory()
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently loaded model type"""
        return self.current_model
