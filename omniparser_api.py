#!/usr/bin/env python3
"""
Streamlined OmniParser API for Google Colab and local use
"""
from util.omniparser import Omniparser
import sys
import os
import base64
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Get the project root directory (platform independent)
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


class ParseRequest(BaseModel):
    """Request model for parsing images"""
    base64_image: str = Field(..., description="Base64 encoded image")
    som_model_path: Optional[str] = Field(
        None, description="Path to SOM model (optional)")
    caption_model_name: str = Field(
        "florence2", description="Caption model name")
    caption_model_path: Optional[str] = Field(
        None, description="Path to caption model (optional)")
    device: str = Field("cpu", description="Device to use (cpu/cuda)")
    box_threshold: float = Field(0.05, description="Box detection threshold")


class ParseResponse(BaseModel):
    """Response model for parsing results"""
    success: bool
    som_image_base64: Optional[str] = None
    parsed_content_list: List[Dict[str, Any]] = []
    processing_time: float
    num_elements: int
    error_message: Optional[str] = None


class OmniParserAPI:
    def __init__(self):
        self.app = FastAPI(
            title="OmniParser API",
            description="Streamlined API for UI element detection and parsing",
            version="1.0.0"
        )
        self.omniparser = None
        self.current_config = None
        self._setup_routes()

    def _get_default_paths(self) -> Dict[str, str]:
        """Get default model paths (platform independent)"""
        weights_dir = PROJECT_ROOT / "weights"
        return {
            "som_model_path": str(weights_dir / "icon_detect" / "model.pt"),
            "caption_model_path": str(weights_dir / "icon_caption_florence")
        }

    def _initialize_omniparser(self, config: Dict[str, Any]) -> None:
        """Initialize or reinitialize omniparser with new config"""
        # Use default paths if not provided
        defaults = self._get_default_paths()

        final_config = {
            "som_model_path": config.get("som_model_path") or defaults["som_model_path"],
            "caption_model_name": config.get("caption_model_name", "florence2"),
            "caption_model_path": config.get("caption_model_path") or defaults["caption_model_path"],
            "device": config.get("device", "cpu"),
            "BOX_TRESHOLD": config.get("box_threshold", 0.05)
        }

        # Only reinitialize if config changed
        if self.current_config != final_config:
            print(f"Initializing OmniParser with config: {final_config}")
            self.omniparser = Omniparser(final_config)
            self.current_config = final_config
            print("OmniParser initialized successfully!")

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.post("/parse", response_model=ParseResponse)
        async def parse_image(request: ParseRequest):
            """Parse UI elements from base64 encoded image"""
            try:
                start_time = time.time()

                # Initialize omniparser with request config
                config = {
                    "som_model_path": request.som_model_path,
                    "caption_model_name": request.caption_model_name,
                    "caption_model_path": request.caption_model_path,
                    "device": request.device,
                    "box_threshold": request.box_threshold
                }

                self._initialize_omniparser(config)

                # Parse the image
                som_image_base64, parsed_content_list = self.omniparser.parse(
                    request.base64_image)

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
            """Health check endpoint"""
            return {"status": "healthy", "message": "OmniParser API is running"}

        @self.app.get("/config")
        async def get_current_config():
            """Get current model configuration"""
            defaults = self._get_default_paths()
            return {
                "current_config": self.current_config,
                "default_paths": defaults,
                "available_devices": ["cpu", "cuda"] if self._cuda_available() else ["cpu"]
            }

    def _cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Global API instance
api = OmniParserAPI()
app = api.app

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OmniParser API Server")
    parser.add_argument("--host", type=str,
                        default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload")

    args = parser.parse_args()

    print(f"Starting OmniParser API on {args.host}:{args.port}")
    print(f"Project root: {PROJECT_ROOT}")

    api.run(host=args.host, port=args.port, reload=args.reload)
