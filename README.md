# WAN Media Pipeline API

A FastAPI server for generating images and videos using WAN 2.2 Text-to-Video (T2V) and Image-to-Video (I2V) models with Instagirl LoRA integration.

## Pipeline Overview

The media generation pipeline supports two main workflows:

1. **Text-to-Image Generation**: Uses WAN 2.2 T2V with Instagirl LoRA to generate high-quality portrait images from text prompts
2. **Image-to-Video Generation**: Uses WAN 2.2 I2V to animate existing images based on text prompts, creating dynamic videos

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

### Generate Video from Image

**POST** `/generate-video-from-image/`

Generate a video from an uploaded image and text prompt using WAN 2.2 I2V. The output video is automatically interpolated to 32 fps using ffmpeg's minterpolate filter for smooth motion.

**Parameters:**

-   `image` (file): Input image file
-   `prompt` (form field): Text description for video animation

**Example:**

```bash
curl -X POST "http://localhost:8000/generate-video-from-image/" \
  -F "image=@path/to/your/image.jpg" \
  -F "prompt=The person in the image smiling and looking around"
```

### Job Management Endpoints

**GET** `/job-status/{job_id}`

Check the status of a generation job.

**GET** `/get-image/{job_id}`

Download the generated image for a completed image generation job.

**GET** `/get-video/{job_id}`

Download the generated video for a completed video generation job.

**GET** `/jobs`

List all jobs (for debugging/admin purposes).

### Other Endpoints

-   **GET** `/`: Health check endpoint

## Testing

You can test the API using curl commands or any HTTP client:

```bash
# Test image generation
curl -X POST "http://localhost:8000/generate-image/" \
  -F "prompt=A beautiful portrait of a woman with flowing hair"

# Check job status (replace job_id with actual ID from response)
curl "http://localhost:8000/job-status/{job_id}"

# Download completed image
curl "http://localhost:8000/get-image/{job_id}" --output generated_image.png
```

## Model Integration

### Current Implementation

The server uses **real WAN 2.2 model integrations** with memory-efficient loading:

-   **WAN 2.2 T2V**: Text-to-video generation with Instagirl LoRA for high-quality portrait generation
-   **WAN 2.2 I2V**: Image-to-video generation for animating static images
-   **Automatic Model Management**: Models are loaded on-demand and unloaded when not in use to optimize GPU memory

### Model Loading Behavior

1. **On-Demand Loading**: Models are only loaded when needed for specific tasks
2. **Memory Optimization**: Only one model type (T2V or I2V) is loaded at a time
3. **Automatic Cleanup**: Models are automatically unloaded after generation to free GPU memory
4. **Hardware Detection**: Automatically uses CUDA if available, with proper memory management

### Features

-   **Instagirl LoRA Integration**: Automatically downloads and applies Instagirl LoRA for enhanced portrait generation
-   **Job-Based Processing**: Asynchronous background processing with status tracking
-   **Memory Efficient**: Dynamic model loading/unloading to support multiple concurrent requests
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
