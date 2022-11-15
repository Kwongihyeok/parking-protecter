import RPi.GPIO as GPIO
import time

buzzer =18
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer, GPIO.OUT)
GPIO.setwarnings(False)

pwm = GPIO.PWM(buzzer, 1.0)
pwm.start(50.0)

for cnt in range(0.3):
    pwm.ChangeFrequency(262)
    time.sleep(1.0)
    pwm.ChangeFrequency(294)
    time.sleep(1.0)
    pwm.ChangeFrequency(330)
    time.sleep(1.0)

pwm.ChangeDutyCycle(0.0)

pwm.stop()
GPIO.cleanup()