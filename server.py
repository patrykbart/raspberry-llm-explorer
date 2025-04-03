import ollama
import logging
from flask import Flask, request, jsonify

SYSTEM_PROMPT = "Answer with max one word."
MODEL_NAME = "llava-custom"
PORT = 12345

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Create custom model based on llava
def create_custom_model():
    try:
        ollama.create(
            model=MODEL_NAME,
            from_="llava",
            system=SYSTEM_PROMPT,
        )
        logger.info("Custom model created successfully.")
    except Exception as e:
        logger.error(f"Error creating custom model: {e}")

create_custom_model()

# Preload the model on startup
def preload_model():
    try:
        ollama.generate(
            model=MODEL_NAME,
            prompt="",
            images=[],
            stream=False
        )
        logger.info("Model preloaded successfully.")
    except Exception as e:
        logger.error(f"Error preloading model: {e}")

preload_model()

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    if not data or "image" not in data:
        logger.warning("Request missing 'image' field in payload")
        return jsonify({"error": "JSON payload with 'image' field is required"}), 400

    image_b64 = data["image"]
    prompt = data.get("prompt", "What is in this picture?")
    
    logger.info(f"Processing inference request with prompt: {prompt[:50]}...")

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            images=[image_b64],
            stream=False,
            options={
                "temperature": 0.0,
                "seed": 42,
            }
        )
    except Exception as e:
        logger.error(f"Error generating response from Ollama: {e}")
        return jsonify({"error": f"Error generating response from Ollama: {e}"}), 500

    logger.info("Successfully generated response")
    return jsonify(response.model_dump_json())

if __name__ == '__main__':
    logger.info("Starting server on port %d", PORT)
    app.run(host="0.0.0.0", port=PORT)