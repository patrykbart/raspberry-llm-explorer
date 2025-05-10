import socket
import logging
from schema.model import RASPBERRY_PI_IP, RASPBERRY_PI_PORT

logger = logging.getLogger(__name__)

def get_distance_from_pi():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RASPBERRY_PI_IP, 5055))  # Sonic sensor port
            s.sendall(b'GetDistance')
            distance = s.recv(1024).decode().strip()
            logger.info(f"Distance: {distance}")
            return float(distance)
    except Exception as e:
        logger.warning(f"Error in GetDistance: {e}")
        return -1

def send_command_to_car(command: str):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
            s.sendall(command.encode())
            return "OK"
    except Exception as e:
        logger.error(f"Error in command: {e}")
        return f"Error: {e}"