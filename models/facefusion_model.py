"""
FaceFusion Model Integration

This module provides integration with FaceFusion for face swapping functionality using headless mode.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class FaceFusionModel:
    """Wrapper for FaceFusion face swapping model using headless mode"""
    
    def __init__(self, facefusion_path: Optional[str] = None):
        self.model_loaded = False
        # Default to facefusion directory in the project root
        if facefusion_path is None:
            self.facefusion_path = Path(__file__).parent.parent / "facefusion"
        else:
            self.facefusion_path = Path(facefusion_path)
        
    def load_model(self):
        """Check if FaceFusion is available"""
        logger.info("Checking FaceFusion installation...")
        
        # Check if facefusion directory exists
        if not self.facefusion_path.exists():
            raise RuntimeError(
                f"FaceFusion not found at {self.facefusion_path}. "
                "The repository should have been cloned automatically on startup. "
                "If you see this error, please check the server startup logs."
            )
        
        # Check if facefusion.py exists
        facefusion_script = self.facefusion_path / "facefusion.py"
        if not facefusion_script.exists():
            raise RuntimeError(
                f"facefusion.py not found at {facefusion_script}. "
                "The FaceFusion repository may not have been cloned properly. "
                "Please check the server startup logs."
            )
        
        logger.info(f"FaceFusion found at: {self.facefusion_path}")
        self.model_loaded = True
    
    def swap_faces(
        self, 
        source_image_path: str, 
        target_image_path: str, 
        output_path: str
    ) -> str:
        """
        Swap face from source image to target image using FaceFusion headless mode
        
        Args:
            source_image_path: Path to the source image (headshot)
            target_image_path: Path to the target image (base frame)
            output_path: Path to save the result
            
        Returns:
            Path to the output image with swapped face
        """
        if not self.model_loaded:
            raise RuntimeError("FaceFusion not loaded. Call load_model() first.")
            
        logger.info(f"Swapping face from {source_image_path} to {target_image_path} using FaceFusion headless mode")
        
        # Construct FaceFusion command
        facefusion_script = self.facefusion_path / "facefusion.py"
        cmd = [
            "python", str(facefusion_script),
            "headless-run",
            "--source", source_image_path,
            "--target", target_image_path,
            "--output", output_path,
            # Add recommended options for better quality
            "--face-detector-model", "yolo_face",
            "--face-selector-mode", "best",
            "--face-mask-types", "box",
            "--face-mask-blur", "0.3",
            "--face-mask-padding", "0,0,0,0",
            "--output-video-quality", "95"
        ]
        
        try:
            # Run FaceFusion
            logger.info(f"Running FaceFusion command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"FaceFusion completed successfully. Output: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FaceFusion failed with return code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            raise RuntimeError(f"FaceFusion face swapping failed: {e.stderr}")
        
        # Verify output file was created
        if not Path(output_path).exists():
            raise RuntimeError(f"FaceFusion did not create output file: {output_path}")
        
        logger.info(f"Face swapping completed successfully. Result saved to: {output_path}")
        return output_path

