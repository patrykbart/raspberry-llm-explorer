import json
import ollama
import time
import os
import paramiko
import glob
import base64
import requests
import logging
from flask import Flask, request, jsonify, render_template_string
from pydantic import BaseModel, Field
from pydantic.json_schema import JsonSchemaValue
import socket

MODEL_NAME = "llava-custom"
PORT = 12345
RASPBERRY_PI_IP = "192.168.137.147"
RASPBERRY_PI_PORT = 5053

SYSTEM_PROMPT = """You are an autonomous navigation controller for a car. Based on the camera image, your task is to decide the car's next movement to avoid obstacles and explore the world. The image you get always shows your latest location. Your main obiective is to explore and have fun. JSPN commands:
- \"m\": movement command. Use when not turning (\"F\" for forward, \"B\" for backward, \"L\" for left, \"R\" for right, \"S\" for stop),
- \"s\": speed as a percentage (0-100),
- \"t\": turn angle in degrees (0 if not turning),
- \"d\": duration in seconds (0-4).
- \"r\": sentence describing what you see and why you made this decision

Output exactly one sentence and then valid JSON object."""

# Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_distance_from_pi():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RASPBERRY_PI_IP, 5055)) # Osobny server na PI uruchomiony sonic.py
            s.sendall(b'GetDistance')
            distance = s.recv(1024).decode().strip()
            logger.info(f"Distance: {distance}")

            return float(distance)
    except Exception as e:
        logger.warning(f"Błąd pobierania odległości z Pi: {e}")
        return -1



def fetch_latest_photo_from_pi():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(RASPBERRY_PI_IP, username="pi", password="raspberry")  
        ssh.connect(RASPBERRY_PI_IP, username="pi", password="raspberry")  

        sftp = ssh.open_sftp()
        remote_folder = "/home/pi/New/RaspberryPi-Car"
        files = sftp.listdir(remote_folder)
        photo_files = [f for f in files if f.startswith("photo_") and f.endswith(".jpg")]
        if not photo_files:
            print("Brak zdjęć na Pi")
            return None

        latest_photo = sorted(photo_files, reverse=True)[0]
        local_path = f"./latest.jpg"
        remote_path = f"{remote_folder}/{latest_photo}"
        sftp.get(remote_path, local_path)
        sftp.close()
        ssh.close()

        with open(local_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return encoded

    except Exception as e:
        print(f"Błąd pobierania zdjęcia z Pi: {e}")
        return None

def send_command_to_car(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
            s.sendall(command.encode())
            return "OK"
    except Exception as e:
        logger.error(f"Błąd wysyłania komendy: {e}")
        return f"Błąd: {e}"

# Pydantic schema
class CarMovementCommand(BaseModel):
    m: str = Field(..., description="Movement command", enum=["F", "B", "L", "R", "S"])
    s: float = Field(..., ge=0, le=100)
    t: float = Field(..., ge=0, le=360)
    d: float = Field(..., ge=0, le=4)
    r: str = Field(...)

command_schema: JsonSchemaValue = CarMovementCommand.model_json_schema()
logger.info("CarMovementCommand JSON Schema:\n%s", json.dumps(command_schema, indent=4))

# Globals to hold the latest results
last_image_b64 = ""
last_prompt = ""
last_response = ""

# Init app
app = Flask(__name__)

# Model creation and preload
try:
    ollama.create(model=MODEL_NAME, from_="llava", system=SYSTEM_PROMPT)
    logger.info("Custom model created successfully.")
except Exception as e:
    logger.warning(f"Model create skipped or failed: {e}")

try:
    ollama.generate(model=MODEL_NAME, prompt="", images=[], stream=False)
    logger.info("Model preloaded successfully.")
except Exception as e:
    logger.warning(f"Model preload skipped or failed: {e}")


@app.route('/camera_pan', methods=['POST'])
def camera_pan():
    try:
        command = request.get_json().get("command", "")
        logger.info(f"[camera_pan] Komenda obrotu kamery (GPIO 5): {command}")

        if command not in ["PanLeft", "PanRight"]:
            return jsonify({"error": "Nieprawidłowa komenda"}), 400

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
            s.sendall(command.encode())

        return jsonify({"status": f"Obrót kamery: {command}"})
    except Exception as e:
        logger.error(f"[camera_pan] Błąd: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/infer', methods=['POST'])
def infer():
    global last_image_b64, last_prompt, last_response
    distance_cm = get_distance_from_pi()  # funkcja wywołująca czujnik przez TCP

    data = request.get_json()
    if not data or "image" not in data:
        logger.warning("Brak pola 'image' w żądaniu")
        return jsonify({"error": "JSON payload with 'image' field is required"}), 400

    image_b64 = data["image"]
    distance_cm = get_distance_from_pi()
    distance_text = f"You have a built-in distance sensor in the front and you can use it. At this point, the distance to the nearest obstacle is {distance_cm:.1f} cm. "
    prompt =  data.get("prompt", "What is in this picture?") + distance_text
  #  prompt = data.get("prompt", "What is in this picture?")

    logger.info(f"Prompt: {prompt}")
    logger.info(f"Image base64 length: {len(image_b64)}")

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            images=[image_b64],
            stream=False,
            format=command_schema,
            options={"temperature": 0.8, "seed": None, "num_ctx": 8192, "repeat_penalty": 1.2, "repeat_last_n": 128, "top_k": 50, "top_p": 0.95, "presence_penalty": 1.0, "frequency_penalty": 0.5})
        logger.info(f"[o] Decyzja LLaMA: {response.response}")

        last_image_b64 = image_b64
        last_prompt = prompt
        last_response = response.response

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify(response.model_dump())

PHOTO_DIR = "."  # lub inna ścieżka do zdjęc 

@app.route('/get_distance_sensor', methods=['GET'])
def get_distance():
    try:
        distance = get_distance_from_pi()
        return jsonify({"distance": distance})
    except Exception as e:
        logger.error(f"Błąd przy pobieraniu odległości: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/photos/<filename>')
def photos(filename):
    return send_from_directory(PHOTO_DIR, filename)

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

@app.route('/status')
def status():
    return jsonify({
        "image": last_image_b64,
        "prompt": last_prompt,
        "response": last_response
    })

PHOTO_DIR = "/home/pi/New/RaspberryPi-Car"  # Ścieżka, gdzie są zdjęcia

@app.route('/camera_command', methods=['POST'])
def camera_command():
    try:
        command = request.get_json().get("command", "")
        logger.info(f"Komenda kamery: {command}")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
            for _ in range(1):  # powtórz 1 razy dla widocznego efektu
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
        remote_folder = "/home/pi/New/RaspberryPi-Car"
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
    import glob
    files = sorted(glob.glob("photo_*.jpg"), reverse=True)
    return jsonify([os.path.basename(f) for f in files])

@app.route('/take_photo', methods=['POST'])
def take_photo():
    global last_image_b64
    try:
        # Polecenie do Pi, by zrobił zdjęcie
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RASPBERRY_PI_IP, RASPBERRY_PI_PORT))
            s.sendall("TakePhoto".encode())
        logger.info("Komenda TakePhoto wysłana do Pi")

        # Zapis zdjęcia
        time.sleep(2.5)

        # Pobierz zdjęcie z Pi
        image_b64 = fetch_latest_photo_from_pi()
        if image_b64:
            last_image_b64 = image_b64
            return jsonify({"status": "Zdjęcie pobrane z Pi i ustawione!"})
        else:
            return jsonify({"error": "Nie udało się pobrać zdjęcia z Pi"}), 500

    except Exception as e:
        logger.error(f"Błąd przy robieniu i pobieraniu zdjęcia: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
   return render_template_string('''
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>AI Car Control</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 2em;
        }

        .layout {
            display: flex;
            gap: 2em;
            max-width: 1200px;
            margin: auto;
        }

        .sidebar {
            flex: 1;
            background-color: #161b22;
            padding: 1.5em;
            border-radius: 12px;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
        }

        .main {
            flex: 2;
            background-color: #161b22;
            padding: 2em;
            border-radius: 12px;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
        }

        h2, h3 {
            color: #58a6ff;
            margin-top: 1em;
        }

        label, select, input, button {
            display: block;
            margin: 0.5em 0;
            font-size: 1em;
        }

        input, select {
            background-color: #0d1117;
            color: #c9d1d9;
            border: 1px solid #30363d;
            padding: 0.4em;
            border-radius: 6px;
            width: 100%;
        }

        button {
            background-color: #238636;
            color: white;
            border: none;
            padding: 0.5em 1em;
            border-radius: 6px;
            cursor: pointer;
        }

        button:hover {
            background-color: #2ea043;
        }

        img {
            max-width: 100%;
            margin-top: 1em;
            border-radius: 8px;
            border: 2px solid #30363d;
        }

        #prompt, #response {
            white-space: pre-wrap;
            background-color: #21262d;
            padding: 1em;
            border-radius: 6px;
            border-left: 4px solid #58a6ff;
            margin: 0.5em 0;
        }
    </style>
</head>
<body>
<div class="layout">
    <div class="sidebar">
        <h3> Ręczne sterowanie</h3>
        <form id="manualForm">
            <label for="direction">Kierunek:</label>
            <select id="direction">
                <option value="przód">Przód</option>
                <option value="tył">Tył</option>
            </select>

            <label for="duration">Czas jazdy (sekundy):</label>
            <input type="number" step="0.1" id="duration" value="1.5">

            <button type="submit">Wyślij komendę</button>
        </form>
        <div id="manualStatus"></div>

        <h3> Zdjęcie</h3>
        <button onclick="sendTakePhoto()">Zrób zdjęcie</button>
        <div id="photoStatus"></div>

       <h3> Sterowanie kamerą</h3>
<select id="cameraDirectionCamera">
    <option value="CamLeft">Dół</option>
    <option value="CamRight">Góra</option>
    <option value="PanLeft">Obróć w lewo</option>
    <option value="PanRight">Obróć w prawo</option>
</select>
<button onclick="sendCameraCommand('cameraDirectionCamera')">Wykonaj</button>

<h3> Sterowanie sensorem</h3>
<select id="cameraDirectionSensor">
    <option value="CamUp">Prawo</option>
    <option value="CamDown">Lewo</option>
</select>
<button onclick="sendCameraCommand('cameraDirectionSensor')">Wykonaj</button>

<h3>Zmierz odległość (cm):</h3>
<button onclick="measureDistance()">Pobierz odległość</button>
<div id="distanceResult">Czekam na pomiar...</div>
<div id="cameraStatus"></div>

        
    </div>

    <div class="main">
        <h2> AI Decision Panel</h2>
        <div id="prompt">Prompt: </div>
        <div id="response">Response: </div>
        <img id="photo" src="" alt="Brak zdjęcia">

<h3> Ostatnie 5 zdjęć</h3>
<button onclick="loadLastPhotos()">Pokaż zdjęcia</button>
<div id="lastPhotos" style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1em;"></div>

    </div>


<script>
async function loadHistory() {
    try {
        const res = await fetch('/photo_list');
        const list = await res.json();
        const container = document.getElementById('history');
        container.innerHTML = '';
        list.forEach(file => {
            const img = document.createElement('img');
            img.src = '/photos/' + file;
            img.style.width = '100px';
            img.style.margin = '5px';
            img.title = file;
            container.appendChild(img);
        });
    } catch (err) {
        console.error("Błąd ładowania historii zdjęć:", err);
    }
}
loadHistory();

async function loadLastPhotos() {
    try {
        const res = await fetch('/last_photos');
        const data = await res.json();
        const container = document.getElementById("lastPhotos");
        container.innerHTML = "";

        if (data.images && data.images.length > 0) {
    data.images.forEach(b64 => {
                const img = document.createElement("img");
                img.src = "data:image/jpeg;base64," + b64;
                img.style.maxWidth = "180px";
                img.style.border = "2px solid #333";
                container.appendChild(img);
            });
        } else {
            container.innerText = "Brak zdjęć.";
        }
    } catch (err) {
        console.error("Błąd ładowania zdjęć:", err);
        document.getElementById("lastPhotos").innerText = "Błąd ładowania zdjęć.";
    }
}

async function sendCameraCommand(selectId) {
    const command = document.getElementById(selectId).value;
    try {
        const res = await fetch("/camera_command", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ command })
        });
        const data = await res.json();
        document.getElementById("cameraStatus").innerText = data.status || data.error;
    } catch (err) {
        console.error("Błąd sterowania kamerą:", err);
        document.getElementById("cameraStatus").innerText = "❌ Błąd sterowania kamerą.";
    }
}
</script>
</div>

<script>
    async function fetchStatus() {
        try {
            const res = await fetch('/status');
            const data = await res.json();
            document.getElementById('prompt').innerText = 'Prompt: ' + data.prompt;
            document.getElementById('response').innerText = 'Response: ' + data.response;
            if (data.image) {
                document.getElementById('photo').src = 'data:image/jpeg;base64,' + data.image;
            }
        } catch (err) {
            console.error(err);
        }
    }

    async function sendTakePhoto() {
        try {
            const res = await fetch("/take_photo", { method: "POST" });
            const data = await res.json();
            document.getElementById("photoStatus").innerText = data.status || data.error;
        } catch (err) {
            console.error(err);
            document.getElementById("photoStatus").innerText = "❌ Błąd przy robieniu zdjęcia.";
        }
    }

async function measureDistance() {
    try {
        const res = await fetch("/get_distance_sensor");
        const data = await res.json();
        if (data.distance !== undefined) {
            document.getElementById("distanceResult").innerText = `Odległość: ${data.distance.toFixed(1)} cm`;
        } else {
            document.getElementById("distanceResult").innerText = `Błąd: ${data.error}`;
        }
    } catch (err) {
        console.error("Błąd odczytu odległości:", err);
        document.getElementById("distanceResult").innerText = "❌ Błąd odczytu odległości.";
    }
}

    document.getElementById("manualForm").addEventListener("submit", async function (e) {
        e.preventDefault();
        const direction = document.getElementById("direction").value;
        const duration = parseFloat(document.getElementById("duration").value);

        try {
            const res = await fetch("/manual_command", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ direction, duration })
            });
            const data = await res.json();
            document.getElementById("manualStatus").innerText = data.status || data.error;
        } catch (err) {
            console.error(err);
            document.getElementById("manualStatus").innerText = "Błąd wysyłania komendy.";
        }
    });

    setInterval(fetchStatus, 1000);
    fetchStatus();
</script>


</body>
</html>
''')

if __name__ == '__main__':
    logger.info(f"Server running on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)