# OmniParser Streamlined API

A clean, minimal API server for UI element detection and parsing, optimized for Google Colab and local development.

## Features

- 🚀 **Streamlined**: Removed all Gradio UI and unnecessary boilerplate
- 🌐 **Platform Independent**: Works on local machines and Google Colab
- ⚡ **Fast API**: RESTful API with automatic documentation
- 🔧 **Configurable**: Full control over model parameters
- 📊 **Comprehensive**: Returns detailed parsing results with bounding boxes
- 🖼️ **Visual Output**: Provides labeled images with detected elements

## Quick Start

### Local Development

1. **Start the API server:**

```bash
python omniparser_api.py --port 8000
```

2. **Test the API:**

```bash
python test_api_client.py --image /path/to/your/image.png
```

### Google Colab

1. **Setup the environment:**

```python
# Run the setup script
exec(open('colab_setup.py').read())
```

2. **Start the API in a notebook cell:**

```python
exec(open('colab_api_start.py').read())
```

3. **Test with your images:**

```python
exec(open('colab_test_code.py').read())
```

## API Endpoints

### POST `/parse`

Parse UI elements from a base64 encoded image.

**Request Body:**

```json
{
  "base64_image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "som_model_path": "/path/to/model.pt", // optional
  "caption_model_name": "florence2", // optional
  "caption_model_path": "/path/to/model", // optional
  "device": "cpu", // cpu or cuda
  "box_threshold": 0.05 // detection threshold
}
```

**Response:**

```json
{
  "success": true,
  "som_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "parsed_content_list": [
    {
      "type": "text",
      "bbox": [0.1, 0.2, 0.3, 0.4],
      "interactivity": false,
      "content": "Button Text",
      "source": "box_ocr_content_ocr"
    }
  ],
  "processing_time": 45.2,
  "num_elements": 25,
  "error_message": null
}
```

### GET `/health`

Health check endpoint.

### GET `/config`

Get current configuration and available options.

## Configuration Options

| Parameter            | Type   | Default     | Description                    |
| -------------------- | ------ | ----------- | ------------------------------ |
| `base64_image`       | string | required    | Base64 encoded image           |
| `som_model_path`     | string | auto        | Path to SOM detection model    |
| `caption_model_name` | string | "florence2" | Caption model name             |
| `caption_model_path` | string | auto        | Path to caption model          |
| `device`             | string | "cpu"       | Device to use (cpu/cuda)       |
| `box_threshold`      | float  | 0.05        | Detection confidence threshold |

## Usage Examples

### Python Client

```python
import requests
import base64

# Convert image to base64
with open("image.png", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode()

# Make API request
response = requests.post("http://localhost:8000/parse", json={
    "base64_image": base64_image,
    "device": "cpu",
    "box_threshold": 0.05
})

result = response.json()
print(f"Detected {result['num_elements']} elements")
```

### cURL

```bash
curl -X POST "http://localhost:8000/parse" \
  -H "Content-Type: application/json" \
  -d '{
    "base64_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "device": "cpu",
    "box_threshold": 0.05
  }'
```

### JavaScript/Node.js

```javascript
const fs = require("fs");
const axios = require("axios");

const imageBase64 = fs.readFileSync("image.png", "base64");

axios
  .post("http://localhost:8000/parse", {
    base64_image: imageBase64,
    device: "cpu",
    box_threshold: 0.05,
  })
  .then((response) => {
    console.log(`Detected ${response.data.num_elements} elements`);
  });
```

## File Structure

```
omniparser-api/
├── omniparser_api.py          # Main API server
├── test_api_client.py         # Test client
├── colab_setup.py            # Google Colab setup
├── requirements.txt          # Dependencies
├── weights/                  # Model files
│   ├── icon_detect/
│   │   └── model.pt
│   └── icon_caption_florence/
└── util/                     # Core utilities
    └── omniparser.py
```

## Performance Tips

1. **Use GPU when available**: Set `device: "cuda"` for faster processing
2. **Adjust threshold**: Lower values detect more elements but may include noise
3. **Batch processing**: Process multiple images by making concurrent requests
4. **Model caching**: The API caches models between requests with same config

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure model files are in the correct `weights/` directory
2. **CUDA errors**: Use `device: "cpu"` if GPU setup is problematic
3. **Memory issues**: Reduce image size or use CPU mode
4. **Timeout**: Increase request timeout for large images

### Debug Mode

Start the server with debug logging:

```bash
python omniparser_api.py --reload
```

## Google Colab Specific Notes

1. **File Upload**: Use Colab's file upload widget to upload images
2. **GPU Access**: Enable GPU runtime for faster processing
3. **Persistent Storage**: Mount Google Drive to save results
4. **Port Access**: Use ngrok or similar for external access

## Dependencies

- FastAPI >= 0.104.0
- Uvicorn >= 0.24.0
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- PaddlePaddle >= 2.5.0
- Ultralytics >= 8.0.0

Install all dependencies:

```bash
pip install -r requirements.txt
```

## License

Same as the original OmniParser project.
