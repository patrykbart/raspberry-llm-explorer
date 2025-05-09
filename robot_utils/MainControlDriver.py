import socket
import time
import cv2
from OledModule.OLED import OLED
import RPi.GPIO as GPIO
GPIO.setwarnings(False)

servoPin1 = 5
servoPin2 = 6
servoPin3 = 7
angle1 = 90
angle2 = 90

GPIO.setup(servoPin1, GPIO.OUT)
GPIO.setup(servoPin2, GPIO.OUT)
GPIO.setup(servoPin3, GPIO.OUT)

def servoPulse(servoPin, myangle):
    pulsewidth = (myangle*11) + 500
    GPIO.output(servoPin,GPIO.HIGH)
    time.sleep(pulsewidth/1000000.0)
    GPIO.output(servoPin,GPIO.LOW)
    time.sleep(20.0/1000 - pulsewidth/1000000.0)

L_IN1 = 20
L_IN2 = 21
L_PWM1 = 0
L_IN3 = 22
L_IN4 = 23
L_PWM2 = 1

R_IN1 = 24
R_IN2 = 25
R_PWM1 = 12

R_IN3 = 26
R_IN4 = 27
R_PWM2 = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(L_IN1,GPIO.OUT)
GPIO.setup(L_IN2,GPIO.OUT)
GPIO.setup(L_PWM1,GPIO.OUT)
GPIO.setup(L_IN3,GPIO.OUT)
GPIO.setup(L_IN4,GPIO.OUT)
GPIO.setup(L_PWM2,GPIO.OUT)
GPIO.setup(R_IN1,GPIO.OUT)
GPIO.setup(R_IN2,GPIO.OUT)
GPIO.setup(R_PWM1,GPIO.OUT)
GPIO.setup(R_IN3,GPIO.OUT)
GPIO.setup(R_IN4,GPIO.OUT)
GPIO.setup(R_PWM2,GPIO.OUT)

GPIO.output(L_IN1,GPIO.LOW)
GPIO.output(L_IN2,GPIO.LOW)
GPIO.output(L_IN3,GPIO.LOW)
GPIO.output(L_IN4,GPIO.LOW)
GPIO.output(R_IN1,GPIO.LOW)
GPIO.output(R_IN2,GPIO.LOW)
GPIO.output(R_IN3,GPIO.LOW)
GPIO.output(R_IN4,GPIO.LOW)

pwm_R1 = GPIO.PWM(R_PWM1,100)
pwm_R2 = GPIO.PWM(R_PWM2,100)
pwm_L1 = GPIO.PWM(L_PWM1,100)
pwm_L2 = GPIO.PWM(L_PWM2,100)

pwm_R1.start(0)
pwm_L1.start(0)
pwm_R2.start(0)
pwm_L2.start(0)

def ahead():
    GPIO.output(L_IN1,GPIO.LOW)
    GPIO.output(L_IN2,GPIO.HIGH)
    pwm_L1.ChangeDutyCycle(80)
    GPIO.output(L_IN3,GPIO.HIGH)
    GPIO.output(L_IN4,GPIO.LOW)
    pwm_L2.ChangeDutyCycle(80)
    GPIO.output(R_IN1,GPIO.HIGH)
    GPIO.output(R_IN2,GPIO.LOW)
    pwm_R1.ChangeDutyCycle(80)
    GPIO.output(R_IN3,GPIO.LOW)
    GPIO.output(R_IN4,GPIO.HIGH)
    pwm_R2.ChangeDutyCycle(80)

def left():
    GPIO.output(L_IN1,GPIO.HIGH)
    GPIO.output(L_IN2,GPIO.LOW)
    pwm_L1.ChangeDutyCycle(80)
    GPIO.output(L_IN3,GPIO.LOW)
    GPIO.output(L_IN4,GPIO.HIGH)
    pwm_L2.ChangeDutyCycle(80)
    GPIO.output(R_IN1,GPIO.HIGH)
    GPIO.output(R_IN2,GPIO.LOW)
    pwm_R1.ChangeDutyCycle(80)
    GPIO.output(R_IN3,GPIO.LOW)
    GPIO.output(R_IN4,GPIO.HIGH)
    pwm_R2.ChangeDutyCycle(80)

def right():
    GPIO.output(L_IN1,GPIO.LOW)
    GPIO.output(L_IN2,GPIO.HIGH)
    pwm_L1.ChangeDutyCycle(80)
    GPIO.output(L_IN3,GPIO.HIGH)
    GPIO.output(L_IN4,GPIO.LOW)
    pwm_L2.ChangeDutyCycle(80)
    GPIO.output(R_IN1,GPIO.LOW)
    GPIO.output(R_IN2,GPIO.HIGH)
    pwm_R1.ChangeDutyCycle(80)
    GPIO.output(R_IN3,GPIO.HIGH)
    GPIO.output(R_IN4,GPIO.LOW)
    pwm_R2.ChangeDutyCycle(80)

def rear():
    GPIO.output(L_IN1,GPIO.HIGH)
    GPIO.output(L_IN2,GPIO.LOW)
    pwm_L1.ChangeDutyCycle(80)
    GPIO.output(L_IN3,GPIO.LOW)
    GPIO.output(L_IN4,GPIO.HIGH)
    pwm_L2.ChangeDutyCycle(80)
    GPIO.output(R_IN1,GPIO.LOW)
    GPIO.output(R_IN2,GPIO.HIGH)
    pwm_R1.ChangeDutyCycle(80)
    GPIO.output(R_IN3,GPIO.HIGH)
    GPIO.output(R_IN4,GPIO.LOW)
    pwm_R2.ChangeDutyCycle(80)

def stop():
    pwm_L1.ChangeDutyCycle(0)
    pwm_L2.ChangeDutyCycle(0)
    pwm_R1.ChangeDutyCycle(0)
    pwm_R2.ChangeDutyCycle(0)

def clear():
    GPIO.cleanup()

def getLocalIp():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        time.sleep(0.1)
    finally:
        s.close()
    return ip

def cameraAction(command):
    global angle1, angle2
    if command == 'CamUp':
        angle1 -= 1
        servoPulse(servoPin2, angle1)
        angle1 = max(angle1, 0)
    elif command == 'CamDown':
        angle1 += 1
        servoPulse(servoPin2, angle1)
        angle1 = min(angle1, 180)
    elif command == 'CamLeft':
        angle2 += 1
        servoPulse(servoPin3, angle2)
        angle2 = min(angle2, 180)
    elif command == 'CamRight':
        angle2 -= 1
        servoPulse(servoPin3, angle2)
        angle2 = max(angle2, 0)
    elif command == 'PanLeft':
        angle3 = 100  # lekki obrót w lewo
        servoPulse(servoPin1, angle3)
    elif command == 'PanRight':
        angle3 = 80   # lekki obrót w prawo
        servoPulse(servoPin1, angle3)

def motorAction(command):
    if command == 'DirForward':
        print("go")
        ahead()
    elif command == 'DirBack':
        print("back")
        rear()
    elif command == 'DirLeft':
        print("left")
        left()
    elif command == 'DirRight':
        print("right")
        right()
    elif command == 'DirStop':
        print("stop")
        stop()

def setCameraAction(command):
    return command if command in ['CamUp', 'CamDown', 'CamLeft', 'CamRight', 'PanLeft', 'PanRight'] else 'CamStop'

def takePhotoAction():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        filename = f"photo_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Zdjęcie zapisane jako {filename}")
    else:
        print("❌ Nie udało się zrobić zdjęcia")
    cap.release()

def main():
    oled = OLED()
    oled.setup()
    ks = 'keyestudio'
    host = '192.168.137.147'
    port = 5053
    oled.writeArea1(ks)
    oled.writeArea3('State:')
    oled.writeArea4(' Disconnect')
    print(f'localhost ip : {host}')
    time.sleep(2)

    tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpServer.bind((host, port))
    tcpServer.setblocking(0)
    tcpServer.listen(5)

    global cameraActionState
    cameraActionState = 'CamStop'

    oled.writeArea1(host)
    time.sleep(2)

    while True:
        try:
            time.sleep(0.001)
            (client, addr) = tcpServer.accept()
            print('accept the client!')
            oled.writeArea4(' Connect')
            client.setblocking(0)
            while True:
                time.sleep(0.001)
                cameraAction(cameraActionState)
                try:
                    data = client.recv(1024)
                    if not data:
                        print('client is closed')
                        oled.writeArea4(' Disconnect')
                        break
                    command = data.decode().strip()
                    if command == "TakePhoto":
                        takePhotoAction()
                        client.sendall(b"OK\n")
                    else:
                        motorAction(command)
                        cameraActionState = setCameraAction(command)
                except socket.error:
                    continue
                except KeyboardInterrupt:
                    raise
        except socket.error:
            pass
        except KeyboardInterrupt:
            tcpServer.close()
            oled.clear()
            print("close")
        except Exception as e:
            print("Exception:", e)
            tcpServer.close()
            oled.clear()

if __name__ == "__main__":
    main()
