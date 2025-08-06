
# OmniParser API - Google Colab All-in-One Setup
# Run this cell to set up everything

import os
import sys
import subprocess
import threading
import time
import base64
import requests
from pathlib import Path
from PIL import Image
import io

print("🚀 Setting up OmniParser API for Google Colab...")

# Install dependencies
print("📦 Installing dependencies...")
dependencies = [
    "fastapi", "uvicorn[standard]", "pydantic", "requests", 
    "pillow", "torch", "torchvision", "ultralytics", 
    "transformers", "paddlepaddle", "paddleocr", 
    "opencv-python", "numpy", "matplotlib"
]

for dep in dependencies:
    print(f"Installing {dep}...")
    subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                  capture_output=True, text=True)

print("✅ Dependencies installed!")

# Create project structure
print("📁 Creating project structure...")
directories = ["weights/icon_detect", "weights/icon_caption_florence", "util"]
for dir_path in directories:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

print("✅ Project structure created!")

# API Server Code (embedded)
api_code = """
import sys
import os
import base64
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Add current directory to path
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
try:
    from util.omniparser import Omniparser
except ImportError:
    print("⚠️  Warning: util.omniparser not found. Please upload your util/ directory.")
    Omniparser = None

class ParseRequest(BaseModel):
    base64_image: str = Field(..., description="Base64 encoded image")
    som_model_path: Optional[str] = Field(None, description="Path to SOM model")
    caption_model_name: str = Field("florence2", description="Caption model name")
    caption_model_path: Optional[str] = Field(None, description="Path to caption model")
    device: str = Field("cpu", description="Device to use (cpu/cuda)")
    box_threshold: float = Field(0.05, description="Box detection threshold")

class ParseResponse(BaseModel):
    success: bool
    som_image_base64: Optional[str] = None
    parsed_content_list: List[Dict[str, Any]] = []
    processing_time: float
    num_elements: int
    error_message: Optional[str] = None

class OmniParserAPI:
    def __init__(self):
        self.app = FastAPI(title="OmniParser API", version="1.0.0")
        self.omniparser = None
        self.current_config = None
        self._setup_routes()
    
    def _get_default_paths(self):
        weights_dir = PROJECT_ROOT / "weights"
        return {
            "som_model_path": str(weights_dir / "icon_detect" / "model.pt"),
            "caption_model_path": str(weights_dir / "icon_caption_florence")
        }
    
    def _initialize_omniparser(self, config):
        if Omniparser is None:
            raise Exception("Omniparser not available. Please upload util/ directory.")
        
        defaults = self._get_default_paths()
        final_config = {
            "som_model_path": config.get("som_model_path") or defaults["som_model_path"],
            "caption_model_name": config.get("caption_model_name", "florence2"),
            "caption_model_path": config.get("caption_model_path") or defaults["caption_model_path"],
            "device": config.get("device", "cpu"),
            "BOX_TRESHOLD": config.get("box_threshold", 0.05)
        }
        
        if self.current_config != final_config:
            print(f"Initializing OmniParser...")
            self.omniparser = Omniparser(final_config)
            self.current_config = final_config
            print("✅ OmniParser initialized!")
    
    def _setup_routes(self):
        @self.app.post("/parse", response_model=ParseResponse)
        async def parse_image(request: ParseRequest):
            try:
                start_time = time.time()
                
                config = {
                    "som_model_path": request.som_model_path,
                    "caption_model_name": request.caption_model_name,
                    "caption_model_path": request.caption_model_path,
                    "device": request.device,
                    "box_threshold": request.box_threshold
                }
                
                self._initialize_omniparser(config)
                som_image_base64, parsed_content_list = self.omniparser.parse(request.base64_image)
                processing_time = time.time() - start_time
                
                return ParseResponse(
                    success=True,
                    som_image_base64=som_image_base64,
                    parsed_content_list=parsed_content_list,
                    processing_time=processing_time,
                    num_elements=len(parsed_content_list)
                )
                
            except Exception as e:
                return ParseResponse(
                    success=False,
                    processing_time=time.time() - start_time,
                    num_elements=0,
                    error_message=str(e)
                )
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "message": "OmniParser API is running"}
        
        @self.app.get("/config")
        async def get_current_config():
            defaults = self._get_default_paths()
            return {
                "current_config": self.current_config,
                "default_paths": defaults,
                "omniparser_available": Omniparser is not None
            }
    
    def run(self, host="0.0.0.0", port=8000):
        uvicorn.run(self.app, host=host, port=port, log_level="info")

# Global API instance
api = OmniParserAPI()
"""

# Write API code to file
with open("omniparser_api_colab.py", "w") as f:
    f.write(api_code)

print("✅ API code created!")

# Helper functions
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_base64_image(base64_string, output_path):
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def start_api_server():
    """Start the API server in background"""
    print("🚀 Starting API server...")
    
    # Import and start the API
    exec(open("omniparser_api_colab.py").read())
    
    def run_server():
        api.run(host="0.0.0.0", port=8000)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(5)
    
    # Test if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running on port 8000!")
            return True
    except:
        pass
    
    print("❌ Failed to start API server")
    return False

def test_api_with_image(image_path, device="cpu", threshold=0.05):
    """Test the API with an image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"🖼️  Testing with image: {image_path}")
    
    # Convert image to base64
    base64_image = image_to_base64(image_path)
    
    # Make API request
    try:
        response = requests.post(
            "http://localhost:8000/parse",
            json={
                "base64_image": base64_image,
                "device": device,
                "box_threshold": threshold
            },
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Detected {result['num_elements']} elements in {result['processing_time']:.2f}s")
            
            # Save labeled image
            if result['som_image_base64']:
                if save_base64_image(result['som_image_base64'], "labeled_output.png"):
                    print("✅ Labeled image saved as 'labeled_output.png'")
            
            return result
        else:
            print(f"❌ API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

# Global variables for easy access
globals().update({
    'start_api_server': start_api_server,
    'test_api_with_image': test_api_with_image,
    'image_to_base64': image_to_base64,
    'save_base64_image': save_base64_image
})

print("\n🎉 Setup complete!")
print("\nNext steps:")
print("1. Upload your model files to weights/ directories")
print("2. Upload your util/ directory with omniparser.py")
print("3. Run: start_api_server()")
print("4. Test with: test_api_with_image('/path/to/your/image.png')")
print("\nExample usage:")
print("  start_api_server()")
print("  result = test_api_with_image('your_image.png', device='cuda')")
