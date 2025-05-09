import socket
import requests
import time
import glob
import RPi.GPIO as GPIO
import os
import re
import cv2
import base64
import json  # <- potrzebne do parsowania JSON-a

SCLK = 8
DIO = 9

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(SCLK, GPIO.OUT)
GPIO.setup(DIO, GPIO.OUT)

AI_SERVER_URL = "http://192.168.137.1:12345/infer"  # Adres serwera AI (głównego komputera)
PI_CONTROL_HOST = "192.168.137.147"  # Komunikacja lokalna z MainControl
PI_CONTROL_PORT = 5053
history = []

face_fun = (
0xe0, 0x80, 0x8e, 0x8e, 0x8e, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x8e, 0x8e, 0x8e, 0x80, 0xe0
)

face_danger = (
0x00, 0x00, 0x0a, 0x04, 0xea, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xa0, 0xea, 0x04, 0x0a, 0x00, 0x00
)

face_explore = (
0x00, 0x00, 0x0e, 0x2e, 0x4e, 0x80, 0x40, 0x20, 0x20, 0x40, 0x80, 0x4e, 0x2e, 0x0e, 0x00, 0x00
)

def nop():
    time.sleep(0.00003)

def start():
    GPIO.output(SCLK, 0)
    nop()
    GPIO.output(SCLK, 1)
    nop()
    GPIO.output(DIO, 1)
    nop()
    GPIO.output(DIO, 0)
    nop()

def end():
    GPIO.output(SCLK, 0)
    nop()
    GPIO.output(DIO, 0)
    nop()
    GPIO.output(SCLK, 1)
    nop()
    GPIO.output(DIO, 1)
    nop()

def send_data(byte):
    for _ in range(8):
        GPIO.output(SCLK, 0)
        if byte & 0x01:
            GPIO.output(DIO, 1)
        else:
            GPIO.output(DIO, 0)
        nop()
        GPIO.output(SCLK, 1)
        nop()
        byte >>= 1
    GPIO.output(SCLK, 0)

def matrix_display(data):
    start()
    send_data(0xC0)  # start address
    for value in data:
        send_data(value)
    end()
    start()
    send_data(0x8A)  # display control: brightness
    end()

def extract_json_from_text(text):
    try:
        match = re.search(r"\{.*\}", text)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print("Nie udało się wydobyć JSON z tekstu:", e)
    return None

def take_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    filename = None
    if ret:
        filename = f"photo_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Zapisano zdjęcie jako {filename}")
    else:
        print("Nie udało się zrobić zdjęcia.")
    cap.release()
    return filename

def get_latest_photo():
    photos = sorted(glob.glob("photo_*.jpg"), key=os.path.getmtime, reverse=True)
    if not photos:
        print("Brak zdjęć photo_*.jpg")
        return None
    return photos[0]

def send_photo_to_ai(path):
    try:
        if not os.path.exists(path):
            print("Plik nie istnieje:", path)
            return None

        with open(path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")


        prompt = "You are an autonomous robot control system. Your main obiective is to explore and have fun.\n"
        if history:
            prompt += "Here are the last 5 actions taken by the robot. Use this to make better decisions:\n"
            for h in history:
                prompt += f"- {h['r']} ➤ {h['m']}, t={h['t']}, d={h['d']}\n"
        prompt += (
        "\nBased on the image, decide whether the robot should turn, stop, or go forward. The image you get always shows your latest location"
        "Choose between: F (forward), B (backward), L (rotate left), R (rotate right), S (stop). "
        "JSON format:\n"
        "- m: movement command (F, B, L, R, S)\n"
        "- s: speed 0–100\n"
        "- t: turn angle in degrees\n"
        "- d: duration in seconds from 0-4.\n"
        "Try to avoid obstacles, but don't let them stop you from having fun. You should not repeat the same commands multiple times. If the robot has stopped many times in a row and there are no visible threats, it is better to move forward or turn left or right to explore the surroundings. Start with one sentence describing what you see and your decision, then respond only in JSON format!"
        )

        payload = {
            "image": image_base64,
            "prompt": prompt
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(AI_SERVER_URL, json=payload, headers=headers, timeout=10)

        print(f"Wysłano do AI ({path}, {os.path.getsize(path)} bajtów), status:", response.status_code)

        if response.status_code == 200:
            try:
                raw_json = response.json()
                print("AI raw response:", raw_json)

                if isinstance(raw_json, str):
                    parsed = json.loads(raw_json)
                elif "response" in raw_json:
                    parsed = extract_json_from_text(raw_json["response"])
                else:
                    parsed = raw_json

                print("AI parsed response:", parsed)
                return parsed

            except Exception as parse_err:
                print("Błąd parsowania odpowiedzi AI:", parse_err)

        else:
            print("Błąd AI:", response.status_code, response.text)

    except Exception as e:
        print("Wyjątek przy połączeniu z AI:", e)
    return None

def send_command_to_maincontrol(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((PI_CONTROL_HOST, PI_CONTROL_PORT))
            s.sendall(command.encode() + b"\n")
            print(f"Wysłano do MainControl: {command}")
    except Exception as e:
        print("Błąd połączenia z MainControl:", e)

def main():
    global history
    print("Startuję lokalny AI driver (Pi → AI → MainControl)")
    while True:
        photo = take_photo()
        if not photo:
            time.sleep(3)
            continue

        latest = get_latest_photo()
        if not latest:
            time.sleep(3)
            continue

        result = send_photo_to_ai(latest)
        if result and isinstance(result, dict):
            m = result.get("m", None)
            t = result.get("t", 0)
            d = result.get("d", 1.5)
            r = result.get("r", "")

            # Wyświetlanie odpowiedniego wyrazu twarzy
            r_lower = r.lower()
            if "danger" in r_lower:
                matrix_display(face_danger)
            elif "fun" in r_lower:
                matrix_display(face_fun)
            elif "explore" in r_lower:
                matrix_display(face_explore)
            else:
                matrix_display(face_fun)  # Domyślnie radość

            # Ogranicz czas trwania d
            if d > 4:
                print(f"AI zwróciło za długi czas d={d}, przycinam do 4s.")
                d = 4
            elif d < 0:
                print(f"AI zwróciło ujemny czas d={d}, ustawiam na 0.")
                d = 0
            
            # Zapisz do historii
            history.append({
                "r": r,
                "m": m,
                "t": t,
                "d": d
            })
            if len(history) > 5:
                history = history[-5:]
            # Jeśli 3 ostatnie ruchy są identyczne, resetuj historię
            if len(history) >= 3:
                last_moves = [h["m"] for h in history[-3:]]
                if all(m == last_moves[0] for m in last_moves):
                    print(f"Powtarzający się ruch: {last_moves[0]} – reset historii")
                    history = []

            print(f"AI mówi: {r}")
            print(f"➡️ m: {m}, t: {t}, d: {d}")

            # Mapowanie m → komendy
            command_map = {
                "F": "DirForward",
                "B": "DirBack",
                "L": "DirLeft",
                "R": "DirRight",
                "S": "DirStop"
            }

            # Wybór komendy
            if m:
                command = command_map.get(m, "DirStop")
            elif t > 0:
                command = "DirLeft" if t < 180 else "DirRight"
            else:
                command = "DirStop"

            send_command_to_maincontrol(command)
            time.sleep(d)
            send_command_to_maincontrol("DirStop")
        else:
            print("AI nie zwróciło poprawnej komendy — zatrzymuję pojazd.")
            send_command_to_maincontrol("DirStop")

        time.sleep(3)

if __name__ == "__main__":
    main()