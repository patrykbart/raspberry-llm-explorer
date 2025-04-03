import time
import json
import base64
import requests
import logging
from picamera2 import Picamera2
from PIL import Image
import io

PROMPT = "What is in this picture?"
URL = "http://192.168.0.109:12345/infer"
CAPTURE_INTERVAL = 5  # seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)  # Warm-up

try:
    while True:
        # Capture image directly to memory (as a PIL Image)
        image = picam2.capture_image()
        # Ensure we have a PIL image; if not, convert using Image.fromarray()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # Save the image to a BytesIO buffer as JPEG
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        img_bytes = buffer.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "image": img_b64,
            "prompt": PROMPT
        }

        try:
            logger.info("Sending image to server: %s", URL)
            response = requests.post(URL, json=payload, timeout=30)
            response.raise_for_status()
            response = json.loads(response.json())

            # Extract and format response
            result = {
                "model": response.get("model"),
                "response": json.loads(response.get("response", "")),
                "n_prompt_tokens": response.get("prompt_eval_count"),
                "n_completion_tokens": response.get("eval_count"),
                "load_duration": response.get("load_duration", 0) / 1e9,
                "eval_duration": response.get("eval_duration", 0) / 1e9,
                "prompt_eval_duration": response.get("prompt_eval_duration", 0) / 1e9,
                "total_duration": response.get("total_duration", 0) / 1e9,
            }

            logger.info("Server responded:\n%s", json.dumps(result, indent=4))

        except Exception as e:
            logger.error("API call failed: %s", e)

        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    logger.info("Interrupted by user. Exiting...")
finally:
    picam2.stop()