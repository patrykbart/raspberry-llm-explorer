import socket
import time
import RPi.GPIO as GPIO

TRIG = 14
ECHO = 4
PORT = 5055
HOST = ''  # Listen on all interfaces

def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start_time = time.time()
    stop_time = time.time()

    timeout = time.time() + 2.0  # 50 ms timeout
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
        if time.time() > timeout:
            return -1

    timeout = time.time() + 2.0
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()
        if time.time() > timeout:
            return -1

    elapsed = stop_time - start_time
    distance = (elapsed * 34300) / 2
    return round(distance, 2)

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, False)
time.sleep(2)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"Serwer odległości nasłuchuje na porcie {PORT}...")

    while True:
        conn, addr = s.accept()
        with conn:
            print(f"Połączenie z {addr}")
            data = conn.recv(1024).decode().strip()
            if data == 'GetDistance':
                distance = get_distance()
                print(f"➡️  Wysyłam: {distance} cm")
                conn.sendall(f"{distance}".encode())
            else:
                print(f"Nieznana komenda: {data}")
                conn.sendall(b"ERROR")