"""
WAN 2.2 Text-to-Video and Image-to-Video Model Integration

This module provides integration with both WAN 2.2 T2V and I2V models for generating videos from prompts and images.
"""

import torch
import logging
import gc
import os
import random
from typing import Optional
from pathlib import Path
from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, load_image
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from huggingface_hub import hf_hub_download
import safetensors.torch as st
from PIL import Image
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Constants for first-last frame image processing (from HF Space)
MAX_DIMENSION = 832
MIN_DIMENSION = 480
DIMENSION_MULTIPLE = 16
SQUARE_SIZE = 480

class WANModel:
    """Wrapper for WAN 2.2 Text-to-Video and Image-to-Video models with memory-efficient loading"""
    
    def __init__(self, device: str = "cuda", instagirl_lora_path: str = None):
        self.device = device
        self.t2v_pipeline = None  # Text-to-Video pipeline
        self.i2v_pipeline = None  # Image-to-Video pipeline
        self.i2v_first_last_pipeline = None  # Image-to-Video First-Last Frame pipeline
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
            
        if self.i2v_first_last_pipeline is not None:
            del self.i2v_first_last_pipeline
            self.i2v_first_last_pipeline = None
            
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
        
        # Load Instagirl lora for T2V if path is provided
        if self.instagirl_lora_path:
            logger.info("Loading Instagirl lora...")
            self.t2v_pipeline.load_lora_weights(self.instagirl_lora_path)
        
        self.current_model = "t2v"
        
        logger.info(f"T2V model loaded. GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def _load_i2v_model(self):
        """Load I2V model with dual Lightning LoRAs, unloading T2V if necessary"""
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
            torch_dtype=self.dtype
        )
        self.i2v_pipeline.to(self.device)
        
        # Load Wan 2.2 Lightning LoRAs for I2V - dual LoRA setup
        logger.info("Loading Wan 2.2 Lightning LoRAs for I2V pipeline...")
        
        # 1) High-noise LoRA → transformer
        logger.info("Downloading and loading high-noise LoRA to transformer...")
        hi_path = hf_hub_download(
            "lightx2v/Wan2.2-Lightning",
            "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"
        )
        self.i2v_pipeline.load_lora_weights(hi_path)  # applies to pipe.transformer
        
        # 2) Low-noise LoRA → transformer_2
        logger.info("Downloading and loading low-noise LoRA to transformer_2...")
        lo_path = hf_hub_download(
            "lightx2v/Wan2.2-Lightning",
            "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors"
        )
        state = st.load_file(lo_path)
        state = _convert_non_diffusers_wan_lora_to_diffusers(state)
        self.i2v_pipeline.transformer_2.load_lora_adapter(state)  # apply to the low-noise expert
        
        # Keep the default Wan scheduler (no LCM override for Lightning LoRAs)
        logger.info("Using default Wan scheduler for Lightning LoRAs...")
        
        self.current_model = "i2v"
        
        logger.info(f"I2V model with Lightning LoRAs loaded. GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def _load_i2v_first_last_model(self):
        """Load I2V model for first-last frame with fused Lightning LoRAs"""
        if self.current_model == "i2v_first_last" and self.i2v_first_last_pipeline is not None:
            return  # Already loaded
            
        logger.info("Loading WAN 2.2 I2V model for first-last frame...")
        
        # Unload other models if currently loaded
        if self.current_model in ["t2v", "i2v"]:
            logger.info(f"Unloading {self.current_model} model to free memory...")
            if self.current_model == "t2v" and self.t2v_pipeline is not None:
                del self.t2v_pipeline
                self.t2v_pipeline = None
            elif self.current_model == "i2v" and self.i2v_pipeline is not None:
                del self.i2v_pipeline
                self.i2v_pipeline = None
            gc.collect()
            torch.cuda.empty_cache()
        
        # Load I2V pipeline
        logger.info("Loading I2V pipeline for first-last frame...")
        self.i2v_first_last_pipeline = WanImageToVideoPipeline.from_pretrained(
            self.i2v_model_id,
            torch_dtype=self.dtype
        )
        self.i2v_first_last_pipeline.to(self.device)
        
        # Load and fuse Lightning LoRA adapters following HF Space configuration
        logger.info("Loading/fusing Lightning LoRA adapters...")
        self.i2v_first_last_pipeline.load_lora_weights(
            "Kijai/WanVideo_comfy", 
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            adapter_name="lightx2v"
        )
        
        # Load second LoRA for transformer_2 (low noise expert)
        kwargs_lora = {"load_into_transformer_2": True}
        self.i2v_first_last_pipeline.load_lora_weights(
            "Kijai/WanVideo_comfy", 
            weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
            adapter_name="lightx2v_2",
            **kwargs_lora
        )
        
        # Configure and fuse LoRA adapters with HF Space scale values
        self.i2v_first_last_pipeline.set_adapters(
            ["lightx2v", "lightx2v_2"], 
            adapter_weights=[1.0, 1.0]
        )
        self.i2v_first_last_pipeline.fuse_lora(
            adapter_names=["lightx2v"], 
            lora_scale=3.0, 
            components=["transformer"]
        )
        self.i2v_first_last_pipeline.fuse_lora(
            adapter_names=["lightx2v_2"], 
            lora_scale=1.0, 
            components=["transformer_2"]
        )
        self.i2v_first_last_pipeline.unload_lora_weights()
        logger.info("LoRA fusion completed successfully")
        
        self.current_model = "i2v_first_last"
        
        logger.info(f"I2V first-last frame model loaded. GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    def process_image_for_video(self, image: Image.Image) -> Image.Image:
        """
        Process image following HF Space logic for first-last frame.
        This determines the final canvas size for the video.
        """
        width, height = image.size
        if width == height:
            return image.resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)
        
        original_width, original_height = width, height
        aspect_ratio = original_width / original_height
        
        # Calculate new dimensions based on aspect ratio
        if aspect_ratio > 1:  # Landscape
            new_width = min(original_width, MAX_DIMENSION)
            new_height = new_width / aspect_ratio
        else:  # Portrait
            new_height = min(original_height, MAX_DIMENSION)
            new_width = new_height * aspect_ratio
        
        # Scale if below minimum dimension
        if min(new_width, new_height) < MIN_DIMENSION:
            if new_width < new_height:
                scale = MIN_DIMENSION / new_width
            else:
                scale = MIN_DIMENSION / new_height
            new_width *= scale
            new_height *= scale
        
        # Round to multiple of DIMENSION_MULTIPLE
        final_width = int(round(new_width / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
        final_height = int(round(new_height / DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE)
        
        # Ensure minimum dimensions
        final_width = max(final_width, MIN_DIMENSION if aspect_ratio < 1 else SQUARE_SIZE)
        final_height = max(final_height, MIN_DIMENSION if aspect_ratio > 1 else SQUARE_SIZE)
        
        return image.resize((final_width, final_height), Image.Resampling.LANCZOS)
    
    def resize_and_crop_to_match(self, target_image: Image.Image, reference_image: Image.Image) -> Image.Image:
        """
        Resize and crop target image to exactly match reference image dimensions.
        This ensures perfect frame alignment and prevents ghosting artifacts.
        """
        ref_width, ref_height = reference_image.size
        target_width, target_height = target_image.size
        
        # Calculate scale to cover the reference dimensions
        scale = max(ref_width / target_width, ref_height / target_height)
        new_width = int(target_width * scale)
        new_height = int(target_height * scale)
        
        # Resize and center crop
        resized = target_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left = (new_width - ref_width) // 2
        top = (new_height - ref_height) // 2
        
        return resized.crop((left, top, left + ref_width, top + ref_height))
    
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
                num_inference_steps=20,
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
                num_inference_steps=20,
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
        fps: int = 16,
        resolution: str = "480p"
    ) -> str:
        """
        Generate a video from an image and prompt using WAN 2.2 I2V
        
        Args:
            image_path: Path to the input image
            prompt: Text description for video generation
            output_path: Path to save the generated video
            num_frames: Number of frames to generate (calculated from duration_seconds * fps in the API layer)
            fps: Frames per second (fixed at 16 fps for consistent quality)
            resolution: Target resolution - "480p" (default) or "720p"
            
        Returns:
            Path to the generated video file
        """
        # Load I2V model on-demand
        self._load_i2v_model()
            
        logger.info(f"Generating video from image: {image_path} with prompt: '{prompt}' using WAN 2.2 I2V at {resolution}")
        
        # Load and process the input image
        image = load_image(image_path)
        
        # Calculate optimal dimensions based on WAN 2.2 requirements and resolution choice
        if resolution == "720p":
            max_area = 720 * 1280  # 720p area for higher quality
        else:  # Default to 480p
            max_area = 480 * 832   # Original 480p area
            
        aspect_ratio = image.height / image.width
        mod_value = self.i2v_pipeline.vae_scale_factor_spatial * self.i2v_pipeline.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        
        # Resize image to calculated dimensions
        image = image.resize((width, height))
        
        # Define negative prompt (from WAN example)
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        
        generator = torch.Generator(device=self.device).manual_seed(random.randint(0, 2**32 - 1))
        
        with torch.no_grad():
            output = self.i2v_pipeline(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=1.0,  # Lightning LoRA works with CFG=1.0
                num_inference_steps=5,  # Lightning LoRA optimized for 4-6 steps, using 5
                generator=generator,
            ).frames[0]
            
            # Export to video
            export_to_video(output, output_path, fps=fps)
        
        logger.info(f"Video generated and saved to: {output_path}")
        return output_path
    
    def generate_video_from_first_last_frame(
        self,
        start_image_path: str,
        end_image_path: str,
        prompt: str,
        output_path: str,
        duration_seconds: float = 5.0,
        num_inference_steps: int = 8,
        guidance_scale: float = 1.0,
        guidance_scale_2: float = 1.0,
        shift: float = 8.0,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate a video from start and end frame images using WAN 2.2 I2V first-last frame
        
        Args:
            start_image_path: Path to the start frame image
            end_image_path: Path to the end frame image
            prompt: Text description for the transition
            output_path: Path to save the generated video
            duration_seconds: Video duration in seconds
            num_inference_steps: Number of inference steps (default 8 for fused LoRAs)
            guidance_scale: Guidance scale for high noise (default 1.0)
            guidance_scale_2: Guidance scale for low noise (default 1.0)
            shift: Scheduler shift parameter (default 8.0)
            seed: Random seed for reproducibility (None for random)
            
        Returns:
            Path to the generated video file
        """
        # Load I2V first-last model on-demand
        self._load_i2v_first_last_model()
        
        logger.info(f"Generating video from first-last frame - Start: {start_image_path}, End: {end_image_path}")
        logger.info(f"Prompt: '{prompt}', Duration: {duration_seconds}s, Steps: {num_inference_steps}, Shift: {shift}")
        
        # Load images
        start_img = Image.open(start_image_path).convert("RGB")
        end_img = Image.open(end_image_path).convert("RGB")
        
        # Process images to prevent ghosting artifacts
        processed_start = self.process_image_for_video(start_img)
        processed_end = self.resize_and_crop_to_match(end_img, processed_start)
        
        # Calculate video parameters
        fixed_fps = 16.0
        min_frames_model = 5
        max_frames_model = 121
        num_frames = int(round(fixed_fps * duration_seconds))
        num_frames = max(min_frames_model, min(num_frames, max_frames_model))
        
        target_height, target_width = processed_start.height, processed_start.width
        
        logger.info(f"Video dimensions: {target_width}x{target_height}, Frames: {num_frames}, FPS: {fixed_fps}")
        
        # Handle seed
        if seed is None:
            seed = torch.randint(0, 2**31 - 1, (1,)).item()
        logger.info(f"Using seed: {seed}")
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Configure scheduler with shift parameter
        self.i2v_first_last_pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.i2v_first_last_pipeline.scheduler.config, shift=shift
        )
        
        # Define negative prompt (from WAN example)
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走,过曝，"
        
        with torch.no_grad():
            output = self.i2v_first_last_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=processed_start,
                last_image=processed_end,
                num_frames=num_frames,
                height=target_height,
                width=target_width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                generator=generator,
                output_type="pil",
            ).frames[0]
            
            # Export to video
            export_to_video(output, output_path, fps=int(fixed_fps))
        
        logger.info(f"First-last frame video generated and saved to: {output_path}")
        return output_path
    
    def cleanup_models(self):
        """Public method to clean up all models and free memory"""
        logger.info("Cleaning up WAN models to free memory...")
        self._cleanup_memory()
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently loaded model type"""
        return self.current_model
