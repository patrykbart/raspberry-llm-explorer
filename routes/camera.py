import os
import glob
import time
import logging
import socket
import base64
import paramiko
from flask import request, jsonify, send_from_directory
from schema.model import RASPBERRY_PI_IP, RASPBERRY_PI_PORT

logger = logging.getLogger(__name__)
PHOTO_DIR = "/home/pi/New/RaspberryPi-Car"

def register_camera(app):
    @app.route('/take_photo', methods=['POST'])
    def take_photo():
        from routes.status import set_last_state  

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
                s.sendall("TakePhoto".encode())

            logger.info("Komenda TakePhoto wysłana do Pi")
            time.sleep(2.5)

            image_b64 = fetch_latest_photo_from_pi()
            if image_b64:
                set_last_state(image_b64, "", "")  
                return jsonify({"status": "Zdjęcie pobrane z Pi i ustawione!"})
            else:
                return jsonify({"error": "Nie udało się pobrać zdjęcia z Pi"}), 500

        except Exception as e:
            logger.error(f"Błąd przy robieniu i pobieraniu zdjęcia: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/camera_command', methods=['POST'])
    def camera_command():
        try:
            command = request.get_json().get("command", "")
            logger.info(f"Komenda kamery: {command}")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
                s.sendall(command.encode())
                time.sleep(1)

            return jsonify({"status": f"Wysłano komendę: {command}"})
        except Exception as e:
            logger.error(f"Błąd przy komendzie kamery: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/last_photos')
    def last_photos():
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(RASPBERRY_PI_IP, username="pi", password="raspberry")

            sftp = ssh.open_sftp()
            remote_folder = PHOTO_DIR
            files = sftp.listdir(remote_folder)
            photo_files = sorted(
                [f for f in files if f.startswith("photo_") and f.endswith(".jpg")],
                reverse=True
            )[:5]

            result = []
            for filename in photo_files:
                remote_path = f"{remote_folder}/{filename}"
                local_path = f"temp_{filename}"
                sftp.get(remote_path, local_path)
                with open(local_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    result.append(b64)
                os.remove(local_path)

            sftp.close()
            ssh.close()
            return jsonify({"images": result})

        except Exception as e:
            logger.error(f"Błąd ładowania zdjęć przez paramiko: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/photo_list')
    def photo_list():
        files = sorted(glob.glob("photo_*.jpg"), reverse=True)
        return jsonify([os.path.basename(f) for f in files])

    @app.route('/photos/<filename>')
    def photos(filename):
        return send_from_directory(".", filename)


def fetch_latest_photo_from_pi():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(RASPBERRY_PI_IP, username="pi", password="raspberry")

        sftp = ssh.open_sftp()
        remote_folder = PHOTO_DIR
        files = sftp.listdir(remote_folder)
        photo_files = [f for f in files if f.startswith("photo_") and f.endswith(".jpg")]
        if not photo_files:
            logger.warning("Brak zdjęć na Pi")
            return None

        latest_photo = sorted(photo_files, reverse=True)[0]
        local_path = f"./latest.jpg"
        remote_path = f"{remote_folder}/{latest_photo}"
        sftp.get(remote_path, local_path)
        sftp.close()
        ssh.close()

        with open(local_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    except Exception as e:
        logger.error(f"Błąd pobierania zdjęcia z Pi: {e}")
        return None