#!/usr/bin/env python3
"""
Test client for OmniParser API
"""
import base64
import requests
import json
import time
from pathlib import Path
from PIL import Image
import io


def image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_string = base64.b64encode(image_data).decode('utf-8')
        return base64_string


def save_base64_image(base64_string: str, output_path: str):
    """Save base64 image to file"""
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        img.save(output_path)
        print(f"Image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")


def test_api(api_url: str = "http://localhost:8000", image_path: str = None):
    """Test the OmniParser API"""

    # Use default test image if none provided
    if image_path is None:
        image_path = "/Users/vento/Desktop/Screenshots/peter_pan_ui.png"

    print(f"Testing API at: {api_url}")
    print(f"Using image: {image_path}")

    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return

    # Test health endpoint
    try:
        response = requests.get(f"{api_url}/health")
        print(f"Health check: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
        return

    # Test config endpoint
    try:
        response = requests.get(f"{api_url}/config")
        config_info = response.json()
        print(f"API Config: {json.dumps(config_info, indent=2)}")
    except Exception as e:
        print(f"Config check failed: {e}")

    # Convert image to base64
    print("Converting image to base64...")
    base64_image = image_to_base64(image_path)
    print(f"Image converted (length: {len(base64_image)} chars)")

    # Prepare request
    request_data = {
        "base64_image": base64_image,
        "device": "cpu",  # Change to "cuda" if you have GPU
        "box_threshold": 0.05,
        "caption_model_name": "florence2"
    }

    print("Sending parse request...")
    start_time = time.time()

    try:
        response = requests.post(
            f"{api_url}/parse",
            json=request_data,
            timeout=300  # 5 minute timeout
        )

        if response.status_code == 200:
            result = response.json()
            request_time = time.time() - start_time

            print(f"✅ Request completed in {request_time:.2f} seconds")
            print(
                f"✅ Processing time: {result['processing_time']:.2f} seconds")
            print(f"✅ Success: {result['success']}")
            print(f"✅ Elements detected: {result['num_elements']}")

            if result['success']:
                # Save labeled image if available
                if result['som_image_base64']:
                    save_base64_image(
                        result['som_image_base64'], "api_labeled_output.png")

                # Print first few elements
                print("\n=== FIRST 5 DETECTED ELEMENTS ===")
                for i, element in enumerate(result['parsed_content_list'][:5]):
                    print(f"Element {i+1}: {element}")

                if len(result['parsed_content_list']) > 5:
                    print(
                        f"... and {len(result['parsed_content_list']) - 5} more elements")
            else:
                print(
                    f"❌ Error: {result.get('error_message', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")

    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Request failed: {e}")


def test_different_configs(api_url: str = "http://localhost:8000", image_path: str = None):
    """Test API with different configurations"""

    if image_path is None:
        image_path = "/Users/vento/Desktop/Screenshots/peter_pan_ui.png"

    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return

    base64_image = image_to_base64(image_path)

    # Test different thresholds
    thresholds = [0.01, 0.05, 0.1]

    for threshold in thresholds:
        print(f"\n=== Testing with threshold: {threshold} ===")

        request_data = {
            "base64_image": base64_image,
            "device": "cpu",
            "box_threshold": threshold,
            "caption_model_name": "florence2"
        }

        try:
            response = requests.post(
                f"{api_url}/parse", json=request_data, timeout=300)
            if response.status_code == 200:
                result = response.json()
                print(
                    f"Threshold {threshold}: {result['num_elements']} elements detected in {result['processing_time']:.2f}s")
            else:
                print(
                    f"Error with threshold {threshold}: {response.status_code}")
        except Exception as e:
            print(f"Failed with threshold {threshold}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test OmniParser API")
    parser.add_argument("--url", type=str,
                        default="http://localhost:8000", help="API URL")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--test-configs", action="store_true",
                        help="Test different configurations")

    args = parser.parse_args()

    if args.test_configs:
        test_different_configs(args.url, args.image)
    else:
        test_api(args.url, args.image)
