from flask import render_template_string

def register_ui(app):
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
        }
    }

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
        }
    });

    setInterval(fetchStatus, 1000);
    fetchStatus();
</script>
</body>
</html>
        ''')