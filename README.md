# WAN Media Pipeline API

A FastAPI server for generating images and videos using WAN 2.2 Text-to-Video (T2V) and Image-to-Video (I2V) models with Instagirl LoRA integration.

## Pipeline Overview

The media generation pipeline supports three main workflows:

1. **Text-to-Image Generation**: Uses WAN 2.2 T2V with Instagirl LoRA to generate high-quality portrait images from text prompts
2. **Image-to-Video Generation**: Uses WAN 2.2 I2V to animate existing images based on text prompts, creating dynamic videos
3. **First-Last Frame Video Generation**: Uses WAN 2.2 I2V with fused Lightning LoRAs to generate smooth transitions between two keyframe images

## Installation

1. Clone this repository:

```bash
git clone https://github.com/mister36/wan-pipeline.git
cd wan-pipeline
```

2. Install system dependencies:

```bash
# Install ffmpeg (required for video frame interpolation)
# On Ubuntu/Debian:
sudo apt update && sudo apt install -y ffmpeg

# On macOS:
brew install ffmpeg

# On other systems, ensure ffmpeg is available in PATH
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Note**: The requirements include PyTorch nightly builds with CUDA 12.8+ support for optimal GPU performance. The Instagirl LoRA model will be automatically downloaded on first startup.

4. Start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Usage

The API provides asynchronous job-based processing with background tasks for efficient resource management.

### Generate Image from Text

**POST** `/generate-image/`

Generate a single image from a text prompt using WAN 2.2 T2V with Instagirl LoRA.

**Parameters:**

-   `prompt` (form field): Text description for image generation

**Example:**

```bash
curl -X POST "http://localhost:8000/generate-image/" \
  -F "prompt=A beautiful woman with long flowing hair standing in a magical forest"
```

**Response:**

```json
{
	"job_id": "12345678-1234-1234-1234-123456789abc",
	"status": "queued",
	"message": "Image generation job created and processing started. Use /job-status/{job_id} to check progress and /get-image/{job_id} to download when complete."
}
```

### Generate Multiple Images from Text (Batch)

**POST** `/generate-images-batch/`

Generate multiple images from text prompts using WAN 2.2 T2V with Instagirl LoRA. The model stays loaded between generations for maximum efficiency, making this much faster than multiple single requests.

**Parameters:**

-   `prompts` (form field): JSON array of text descriptions for batch image generation (max 20 prompts)

**Example:**

```bash
curl -X POST "http://localhost:8000/generate-images-batch/" \
  -F 'prompts=["A beautiful woman with long flowing hair", "A serene landscape with mountains", "A futuristic cityscape at night"]'
```

**Response:**

```json
{
	"job_id": "12345678-1234-1234-1234-123456789abc",
	"status": "queued",
	"total_images": 3,
	"message": "Batch image generation job created with 3 prompts. Use /job-status/{job_id} to check progress and /get-batch-images/{job_id} to download when complete."
}
```

### Generate Video from Image

**POST** `/generate-video-from-image/`

Generate a video from an uploaded image and text prompt using WAN 2.2 I2V. The output video is automatically interpolated to 32 fps using ffmpeg's minterpolate filter for smooth motion.

**Parameters:**

-   `image` (file): Input image file
-   `prompt` (form field): Text description for video animation
-   `duration_seconds` (optional): Video duration in seconds (default: 5.0, range: 0.5-30.0)
-   `resolution` (optional): Video resolution - "480p" (default) or "720p"

**Example:**

```bash
curl -X POST "http://localhost:8000/generate-video-from-image/" \
  -F "image=@path/to/your/image.jpg" \
  -F "prompt=The person in the image smiling and looking around" \
  -F "duration_seconds=5.0" \
  -F "resolution=480p"
```

### Generate Video from First and Last Frame

**POST** `/generate-video-from-first-last/`

Generate a video transitioning smoothly between two keyframe images using WAN 2.2 I2V with fused Lightning LoRAs. This specialized endpoint uses advanced LoRA fusion techniques to ensure precise control over both start and end frames while preventing ghosting artifacts through intelligent image alignment.

**Parameters:**

-   `start_image` (file): Start frame image file
-   `end_image` (file): End frame image file
-   `prompt` (form field): Text description for the transition (default: "animate")
-   `negative_prompt` (optional): Negative prompt for quality control
-   `duration_seconds` (optional): Video duration in seconds (default: 5.0, range: 0.5-10.0)
-   `num_inference_steps` (optional): Number of inference steps (default: 8, range: 1-30)
-   `guidance_scale` (optional): Guidance scale for high noise (default: 1.0, range: 0.0-10.0)
-   `guidance_scale_2` (optional): Guidance scale for low noise (default: 1.0, range: 0.0-10.0)
-   `shift` (optional): Scheduler shift parameter (default: 8.0, range: 1.0-10.0)
-   `seed` (optional): Random seed for reproducibility (None for random)

**Example:**

```bash
curl -X POST "http://localhost:8000/generate-video-from-first-last/" \
  -F "start_image=@path/to/start_frame.jpg" \
  -F "end_image=@path/to/end_frame.jpg" \
  -F "prompt=smooth transition with natural movement" \
  -F "duration_seconds=5.0" \
  -F "num_inference_steps=8" \
  -F "shift=8.0" \
  -F "seed=42"
```

**Key Features:**

-   **Dual Guidance Control**: Separate guidance scales for high and low noise levels
-   **Automatic Image Alignment**: Images are processed to match dimensions and prevent ghosting
-   **Fused Lightning LoRAs**: Uses specialized LoRA fusion with different scales for transformer components
-   **Configurable Scheduler**: Adjustable shift parameter for fine-tuning motion dynamics
-   **Reproducible Results**: Optional seed parameter for consistent outputs

### Job Management Endpoints

**GET** `/job-status/{job_id}`

Check the status of a generation job (single image, batch images, or video). For batch jobs, includes progress information and individual image statuses.

**GET** `/get-image/{job_id}`

Download the generated image for a completed single image generation job.

**GET** `/get-batch-image/{job_id}/{image_index}`

Download a specific image from a completed batch generation job.

**GET** `/get-batch-images/{job_id}`

Download all completed images from a batch generation job as a ZIP file.

**GET** `/get-video/{job_id}`

Download the generated video for a completed video generation job.

**GET** `/jobs`

List all jobs (for debugging/admin purposes).

### Other Endpoints

-   **GET** `/`: Health check endpoint

## Testing

You can test the API using curl commands or any HTTP client:

```bash
# Test single image generation
curl -X POST "http://localhost:8000/generate-image/" \
  -F "prompt=A beautiful portrait of a woman with flowing hair"

# Test batch image generation
curl -X POST "http://localhost:8000/generate-images-batch/" \
  -F 'prompts=["A beautiful portrait of a woman", "A serene mountain landscape", "A futuristic city"]'

# Test image-to-video generation
curl -X POST "http://localhost:8000/generate-video-from-image/" \
  -F "image=@path/to/image.jpg" \
  -F "prompt=The person smiling and moving gracefully" \
  -F "duration_seconds=5.0" \
  -F "resolution=480p"

# Test first-last frame video generation
curl -X POST "http://localhost:8000/generate-video-from-first-last/" \
  -F "start_image=@path/to/start.jpg" \
  -F "end_image=@path/to/end.jpg" \
  -F "prompt=smooth natural transition" \
  -F "duration_seconds=5.0" \
  -F "num_inference_steps=8" \
  -F "seed=42"

# Check job status (replace job_id with actual ID from response)
curl "http://localhost:8000/job-status/{job_id}"

# Download completed single image
curl "http://localhost:8000/get-image/{job_id}" --output generated_image.png

# Download completed video
curl "http://localhost:8000/get-video/{job_id}" --output generated_video.mp4

# Download specific image from batch (index 0, 1, 2, etc.)
curl "http://localhost:8000/get-batch-image/{job_id}/0" --output batch_image_0.png

# Download all batch images as ZIP
curl "http://localhost:8000/get-batch-images/{job_id}" --output batch_images.zip
```

## Model Integration

### Current Implementation

The server uses **real WAN 2.2 model integrations** with memory-efficient loading:

-   **WAN 2.2 T2V**: Text-to-video generation with Instagirl LoRA for high-quality portrait generation
-   **WAN 2.2 I2V**: Image-to-video generation for animating static images with Lightning LoRAs
-   **WAN 2.2 I2V First-Last Frame**: Specialized first-last frame video generation with fused Lightning LoRAs for precise keyframe control
-   **Automatic Model Management**: Models are loaded on-demand and unloaded when not in use to optimize GPU memory

### Model Loading Behavior

1. **On-Demand Loading**: Models are only loaded when needed for specific tasks
2. **Memory Optimization**: Only one model variant is loaded at a time (T2V, I2V, or I2V First-Last)
3. **Automatic Cleanup**: Models are automatically unloaded when switching between variants to free GPU memory
4. **Hardware Detection**: Automatically uses CUDA if available, with proper memory management

### LoRA Integration

-   **T2V Model**: Uses Instagirl LoRA (automatically downloaded) for enhanced portrait generation
-   **I2V Model**: Uses dual Lightning LoRAs from `lightx2v/Wan2.2-Lightning` (high-noise and low-noise experts)
-   **I2V First-Last Model**: Uses fused Lightning LoRAs from `Kijai/WanVideo_comfy` with specialized fusion scales (3.0 for transformer, 1.0 for transformer_2)

### Features

-   **Advanced LoRA Fusion**: First-last frame model uses specialized LoRA fusion techniques for optimal quality
-   **Job-Based Processing**: Asynchronous background processing with status tracking
-   **Memory Efficient**: Dynamic model loading/unloading to support multiple concurrent requests
-   **Intelligent Image Processing**: Automatic dimension alignment and cropping to prevent ghosting artifacts
-   **Vertical Format Optimization**: Configured for portrait-oriented content (720x1280)

## Directory Structure

```
wan-pipeline/
├── app.py                    # FastAPI application with job management
├── requirements.txt          # Python dependencies (PyTorch nightly + CUDA)
├── README.md                 # This file
├── models/
│   ├── __init__.py
│   ├── wan_model.py         # WAN 2.2 T2V/I2V model wrapper
│   └── wan_model_append.py  # Additional model utilities
├── temp/                    # Temporary files during processing
├── jobs/                    # Job status persistence (JSON files)
├── images/                  # Generated image outputs
├── videos/                  # Generated video outputs
└── models_cache/            # Cached model files (Instagirl LoRA)
```

## Configuration

### Hardware Requirements

-   **GPU**: NVIDIA GPU with CUDA 12.8+ support (for Blackwell/RTX 40/50 series optimization)
-   **VRAM**: 12GB+ recommended for optimal performance (models are loaded/unloaded dynamically)
-   **RAM**: 16GB+ recommended for large model handling
-   **Storage**: Sufficient space for model cache (~2-3GB) and outputs

### Environment Variables

Optional environment variables for configuration:

-   `CUDA_VISIBLE_DEVICES`: Specify which GPU to use (e.g., "0")
-   Standard PyTorch CUDA environment variables for memory management

## Job Management

The API uses a sophisticated job management system:

-   **Asynchronous Processing**: All generation tasks run in background, allowing immediate response
-   **Status Tracking**: Real-time job status updates (queued, processing, completed, failed)
-   **Persistence**: Jobs are saved to disk for recovery across server restarts
-   **Error Handling**: Comprehensive error reporting with detailed messages

### Job Status Flow

1. **queued**: Job created and waiting to start
2. **processing**: Model is loaded and generation is in progress
3. **completed**: Generation finished, output ready for download
4. **failed**: Error occurred, check error message in job status

## Performance Considerations

-   **Memory Management**: Models are loaded on-demand and unloaded after use
-   **GPU Optimization**: Uses PyTorch nightly for latest CUDA optimizations
-   **Concurrent Requests**: Background job processing supports multiple simultaneous requests
-   **Storage**: Monitor disk space for job persistence and output files
