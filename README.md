# raspberry-llm-explorer

# AI Car Control – Autonomous Vehicle Interface with Local AI

This Flask-based application enables communication between a local computer and a Raspberry Pi-powered vehicle. The vehicle is guided by decisions made by a locally running LLaVA model (via Ollama), based on camera input and distance sensor data.

## Requirements

- Python 3.8+
- Raspberry Pi with `MainControlDriver.py` running
- Ollama installed with the `llava` model
- Required Python libraries (see below)

pip install flask paramiko pydantic requests

## Running the Server

python server.py

Accessible at: http://localhost:12345

Includes:
- Manual vehicle control
- Camera direction control
- Photo capture
- AI decision panel
- Distance sensor readout