
import time
import json
import logging
import socket
from flask import request, jsonify
from schema.model import RASPBERRY_PI_IP, RASPBERRY_PI_PORT
from logic.rasp_comm import send_command_to_car

logger = logging.getLogger(__name__)

def register_manual(app):
    @app.route('/manual_command', methods=['POST'])
    def manual_command():
        data = request.get_json()
        direction = data.get("direction", "").lower()
        duration = float(data.get("duration", 0))

        if direction == "przód":
            command = "DirForward"
        elif direction == "tył":
            command = "DirBack"
        else:
            return jsonify({"error": "Nieprawidłowy kierunek"}), 400

        send_command_to_car(command)
        time.sleep(duration)
        send_command_to_car("DirStop")

        return jsonify({"status": "ok"})

    @app.route('/manual', methods=['POST'])
    def manual():
        try:
            command = request.get_json()
            logger.info(f"[manual] Wysyłanie komendy do pojazdu: {command}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
                s.sendall(json.dumps(command).encode())
            return jsonify({"status": "ok"})
        except Exception as e:
            logger.error(f"[manual] Błąd przy wysyłaniu komendy: {e}")
            return jsonify({"error": str(e)}), 500