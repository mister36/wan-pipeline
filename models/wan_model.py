"""
WAN 2.2 Text-to-Video and Image-to-Video Model Integration

This module provides integration with both WAN 2.2 T2V and I2V models for generating videos from prompts and images.
"""

import torch
import logging
from typing import Optional
from pathlib import Path
from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class WANModel:
    """Wrapper for WAN 2.2 Text-to-Video and Image-to-Video models"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.t2v_pipeline = None  # Text-to-Video pipeline
        self.i2v_pipeline = None  # Image-to-Video pipeline
        self.vae = None
        self.model_loaded = False
        self.t2v_model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        self.i2v_model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        self.dtype = torch.bfloat16
        
    def load_model(self):
        """Load both WAN 2.2 T2V and I2V models"""
        logger.info("Loading WAN 2.2 models...")
        
        # Load VAE for T2V model
        logger.info("Loading VAE...")
        self.vae = AutoencoderKLWan.from_pretrained(
            self.t2v_model_id, 
            subfolder="vae", 
            torch_dtype=torch.float32
        )
        
        # Load T2V pipeline
        logger.info("Loading T2V pipeline...")
        self.t2v_pipeline = WanPipeline.from_pretrained(
            self.t2v_model_id, 
            vae=self.vae,
            torch_dtype=self.dtype
        )
        self.t2v_pipeline.to(self.device)
        
        # Load I2V pipeline
        logger.info("Loading I2V pipeline...")
        self.i2v_pipeline = WanImageToVideoPipeline.from_pretrained(
            self.i2v_model_id, 
            torch_dtype=self.dtype
        )
        self.i2v_pipeline.to(self.device)
        
        logger.info("WAN 2.2 models loaded successfully")
        self.model_loaded = True
    
    def generate_video_from_prompt(
        self, 
        prompt: str, 
        output_path: str,
        width: int = 720,
        height: int = 1280,  # Vertical format for portrait shots
        num_frames: int = 81,
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
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
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
        Generate a single frame from a text prompt using WAN 2.2 T2V (near-still shot)
        
        Args:
            prompt: Text description for image generation
            output_path: Path to save the generated image
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Path to the generated image file
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        logger.info(f"Generating single frame from prompt: '{prompt}' using WAN 2.2 T2V")
        
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
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
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
    

