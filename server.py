from flask import Flask
from routes.infer import register_infer
from routes.manual import register_manual
from routes.camera import register_camera
from routes.sensor import register_sensor
from routes.status import register_status
from routes.ui import register_ui
import logging

app = Flask(__name__)

register_infer(app)
register_manual(app)
register_camera(app)
register_sensor(app)
register_status(app)
register_ui(app)

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    from schema.model import PORT
    app.run(host="0.0.0.0", port=PORT)