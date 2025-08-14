#!/usr/bin/env python3
"""
Test script for FaceFusion headless face swapping
Swaps face from potus2_0.jpg onto Tom_Brady_2019.jpg
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_facefusion_swap():
    """Test FaceFusion face swapping with specified images"""
    
    # Image paths (you'll need to place these in your test environment)
    source_image = "potus2_0.jpg"  # Headshot to extract face from
    target_image = "Tom_Brady_2019.jpg"  # Image to swap face onto
    
    # Output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image = f"face_swap_result_{timestamp}.png"
    
    # FaceFusion path - adjust this based on your environment
    # For local testing, use the cloned facefusion directory
    facefusion_path = Path("facefusion")
    if not facefusion_path.exists():
        logger.error(f"FaceFusion directory not found at {facefusion_path}")
        logger.error("Make sure you've cloned the FaceFusion repository")
        return False
    
    facefusion_script = facefusion_path / "facefusion.py"
    if not facefusion_script.exists():
        logger.error(f"facefusion.py not found at {facefusion_script}")
        return False
    
    # Check if input images exist
    if not Path(source_image).exists():
        logger.error(f"Source image not found: {source_image}")
        logger.error("Please place potus2_0.jpg in the current directory")
        return False
    
    if not Path(target_image).exists():
        logger.error(f"Target image not found: {target_image}")
        logger.error("Please place Tom_Brady_2019.jpg in the current directory")
        return False
    
    # Construct FaceFusion command
    cmd = [
        "python", str(facefusion_script),
        "headless-run",
        "--processors", "face_swapper",
        "--face-swapper-model", "inswapper_128",
        "--source", source_image,
        "--target", target_image,
        "--output-path", output_image,
        # Quality settings
        "--face-detector-model", "yolo_face",
        "--face-selector-mode", "one",
        "--face-mask-types", "box",
        "--face-mask-blur", "0.3",
        "--output-video-quality", "95"
    ]
    
    logger.info("Starting FaceFusion face swap test...")
    logger.info(f"Source (face donor): {source_image}")
    logger.info(f"Target (face recipient): {target_image}")
    logger.info(f"Output: {output_image}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run FaceFusion
        logger.info("Running FaceFusion...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        logger.info("FaceFusion completed successfully!")
        logger.info(f"Stdout: {result.stdout}")
        
        # Verify output file was created
        if Path(output_image).exists():
            file_size = Path(output_image).stat().st_size
            logger.info(f"✅ Output file created: {output_image} ({file_size} bytes)")
            logger.info("Face swap test completed successfully!")
            return True
        else:
            logger.error(f"❌ Output file not created: {output_image}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FaceFusion failed with return code {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("FaceFusion Headless Test")
    logger.info("=" * 60)
    
    success = test_facefusion_swap()
    
    if success:
        logger.info("🎉 Test completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
