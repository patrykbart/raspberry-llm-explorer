import logging
from flask import jsonify
from logic.rasp_comm import get_distance_from_pi

logger = logging.getLogger(__name__)

def register_sensor(app):
    @app.route('/get_distance_sensor', methods=['GET'])
    def get_distance():
        try:
            distance = get_distance_from_pi()
            return jsonify({"distance": distance})
        except Exception as e:
            logger.error(f"Błąd przy pobieraniu odległości: {e}")
            return jsonify({"error": str(e)}), 500