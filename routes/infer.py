# -*- coding: utf-8 -*-
from flask import request, jsonify
from schema.model import MODEL_NAME, command_schema
from logic.rasp_comm import get_distance_from_pi
from logic.ai_client import ollama_generate
from routes.status import set_last_state
import logging

logger = logging.getLogger(__name__)



def register_infer(app):
    @app.route('/infer', methods=['POST'])
    def infer():
        data = request.get_json()
        if not data or "image" not in data:
            logger.warning("Brak pola 'image'")
            return jsonify({"error": "JSON payload with 'image' field is required"}), 400

        image_b64 = data["image"]
        distance_cm = get_distance_from_pi()
        distance_text = f"You have a built-in distance sensor in the front and you can use it. At this point, the distance to the nearest obstacle is {distance_cm:.1f} cm. "
        prompt = data.get("prompt", "What is in this picture?") + distance_text

        logger.info(f"Prompt: {prompt}")
        logger.info(f"Image base64 length: {len(image_b64)}")

        try:
            response = ollama_generate(
                model=MODEL_NAME,
                prompt=prompt,
                images=[image_b64],
                format=command_schema
            )
            logger.info(f"[o] Decyzja LLaMA: {response.response}")

            set_last_state(image_b64, prompt, response.response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return jsonify({"error": str(e)}), 500

        return jsonify(response.model_dump())