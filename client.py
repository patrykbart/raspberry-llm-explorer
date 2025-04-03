import json
import base64
import requests
import logging

PROMPT = "What is in this picture?."
URL = "http://172.29.0.6:12345/infer"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to your image file for testing (ensure test.jpg exists in the same folder)
image_path = "image.jpg"

# Read the image file and encode it as a base64 string
with open(image_path, "rb") as img_file:
    img_bytes = img_file.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# Build the JSON payload; you can also pass a custom prompt if desired.
payload = {
    "image": img_b64,
    "prompt": PROMPT
}

try:
    logger.info("Sending request to %s", URL)
    response = requests.post(URL, json=payload, timeout=30)
    response.raise_for_status()

    response = json.loads(response.json())
    
    # Calculate the number of tokens
    token_count = len(response.get("context", []))

    # Convert durations from nanoseconds to seconds (assuming nanosecond values)
    load_duration_sec = response.get("load_duration", 0) / 1e9
    eval_duration_sec = response.get("eval_duration", 0) / 1e9
    prompt_eval_duration_sec = response.get("prompt_eval_duration", 0) / 1e9
    total_duration_sec = response.get("total_duration", 0) / 1e9

    # Log the formatted output
    result = {
        "model": response.get("model"),
        "response": response.get("response"),
        "n_prompt_tokens": response.get("prompt_eval_count"),
        "n_completion_tokens": response.get("eval_count"),
        "load_duration": load_duration_sec,
        "eval_duration": eval_duration_sec,
        "prompt_eval_duration": prompt_eval_duration_sec,
        "total_duration": total_duration_sec,
    }
    logger.info("Response received:\n%s", json.dumps(result, indent=4))

except Exception as e:
    logger.error("Error during API call: %s", e)