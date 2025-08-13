# WAN Video Pipeline API

A FastAPI server that generates videos from headshots and text prompts using WAN 2.2 i2v and FaceFusion technologies.

## Pipeline Overview

The video generation pipeline works in four main steps:

1. **Initial Video Generation**: Uses WAN 2.2 i2v to generate a near-still vertical shot based on the text prompt
2. **Frame Extraction**: Extracts frame 0 from the initial video as the base still image
3. **Face Swapping**: Uses FaceFusion to swap the face from the uploaded headshot into the base still
4. **Final Video Generation**: Feeds the face-swapped still back into WAN i2v to create the final animated video

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd wan-pipeline
```

2. Clone FaceFusion (Required):

```bash
git clone https://github.com/facefusion/facefusion.git
```

Your directory structure should look like:

```
wan-pipeline/
├── app.py
├── models/
├── facefusion/          # <- FaceFusion repository
│   ├── facefusion.py    # <- This file must exist
│   └── ...
├── requirements.txt
└── README.md
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Setup FaceFusion dependencies (if needed):

```bash
cd facefusion
pip install -r requirements.txt
cd ..
```

5. Start the server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Usage

### Generate Video Endpoint

**POST** `/generate-video/`

Upload a headshot image and provide a text prompt to generate a video.

**Parameters:**

-   `headshot` (file): Image file containing a headshot
-   `prompt` (string): Text description for the video scene

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/generate-video/" \
  -F "headshot=@path/to/your/headshot.jpg" \
  -F "prompt=A person standing in a beautiful garden with flowers blooming around them"
```

**Response:**

```json
{
	"message": "Video generated successfully",
	"session_id": "12345678-1234-1234-1234-123456789abc",
	"video_path": "outputs/video_12345678-1234-1234-1234-123456789abc.mp4",
	"download_url": "/download-video/12345678-1234-1234-1234-123456789abc"
}
```

### Download Video Endpoint

**GET** `/download-video/{session_id}`

Download the generated video using the session ID.

### Other Endpoints

-   **GET** `/`: Health check endpoint
-   **DELETE** `/cleanup/{session_id}`: Delete a generated video file

## Testing

Run the test client to verify the API is working:

```bash
python test_client.py
```

This will:

1. Check if the server is running
2. Create a test headshot image (if none exists)
3. Send a video generation request
4. Download the generated video

## Model Integration Status

### Current Implementation

The server includes **real model integrations** with automatic fallbacks:

-   **WAN 2.2 Video Generation**: Uses WAN 2.2 T2V and I2V models for high-quality text-to-video and image-to-video generation
-   **Face Swapping**: Uses FaceFusion in headless mode for professional-grade face swapping

### Model Loading Behavior

1. **On startup**, the server attempts to load real models:

    - **WAN 2.2 T2V/I2V**: Downloads automatically from Hugging Face
    - **FaceFusion**: Uses the cloned FaceFusion repository in headless mode

2. **If model loading fails** (e.g., insufficient GPU memory, missing dependencies):

    - Falls back to high-quality placeholder implementations
    - Still provides functional video generation with basic effects

3. **Automatic hardware optimization**:
    - Uses CUDA if available, falls back to CPU
    - Enables memory-efficient attention and VAE slicing
    - Implements model CPU offloading for large models

### Installation Options

**Option 1: Full Installation (Recommended)**

```bash
pip install -r requirements.txt
```

**Option 2: Lightweight Installation (CPU only, faster)**

```bash
pip install fastapi uvicorn python-multipart Pillow opencv-python numpy
```

**Option 3: GPU Installation with CUDA**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Directory Structure

```
wan-pipeline/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── models/
│   ├── __init__.py
│   ├── wan_model.py       # WAN model wrapper
│   └── facefusion_model.py # FaceFusion model wrapper
├── temp/                  # Temporary files during processing
└── outputs/               # Generated video outputs
```

## Configuration

### Hardware Requirements

-   **GPU**: NVIDIA GPU with CUDA support recommended for model inference
-   **RAM**: 8GB+ recommended
-   **Storage**: Sufficient space for temporary files and video outputs

### Environment Variables

You can configure the following environment variables:

-   `CUDA_VISIBLE_DEVICES`: Specify which GPU to use
-   `MODEL_CACHE_DIR`: Directory for cached model files

## Error Handling

The API includes comprehensive error handling for:

-   Invalid file uploads
-   Model loading failures
-   Video processing errors
-   Disk space issues

All errors return appropriate HTTP status codes with detailed error messages.

## Performance Considerations

-   Video generation is computationally intensive
-   Consider implementing request queuing for multiple concurrent requests
-   Monitor disk space as temporary files can accumulate
-   Use background tasks for long-running processes

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
