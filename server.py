import json
import ollama
import logging
from flask import Flask, request, jsonify
from pydantic.json_schema import JsonSchemaValue
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You are an autonomous navigation controller for a car. Based on the camera image, your task is to decide the car's next movement to avoid obstacles and explore the world. Output a concise JSON command with these keys only:
- "m": movement command ("F" for forward, "B" for backward, "L" for left, "R" for right, "S" for stop),
- "s": speed as a percentage (0-100),
- "t": turn angle in degrees (0 if not turning),
- "d": duration in seconds.

Do not include any extra text. Output exactly one valid JSON object."""
MODEL_NAME = "llava-custom"
PORT = 12345

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CarMovementCommand(BaseModel):
    m: str = Field(
        ...,
        description="Movement command: 'F' for forward, 'B' for backward, 'L' for left, 'R' for right, 'S' for stop.",
        enum=["F", "B", "L", "R", "S"]
    )
    s: float = Field(
        ...,
        description="Speed as a percentage (0-100).",
        ge=0,
        le=100
    )
    t: float = Field(
        ...,
        description="Turn angle in degrees (0 if not turning).",
        ge=0,
        le=360
    )
    d: float = Field(
        ...,
        description="Duration in seconds for executing the action.",
        ge=0
    )

command_schema: JsonSchemaValue = CarMovementCommand.model_json_schema()
logging.info("CarMovementCommand JSON Schema:\n%s", json.dumps(command_schema, indent=4))

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
            format=command_schema,
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