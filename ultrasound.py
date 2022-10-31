import RPi.GPIO as GPIO #파이썬의 GPIO 모듈 불러온다.
import time # 파이썬에서 time 모듈을 불러온다.


GPIO.setmode(GPIO.BCM) #GPIO의 라즈베리파이 GPIO번호로 사용

TRIG = 10 # 핀번호 설정
ECHO = 9  # 핀번호 설정

GPIO.setup(TRIG, GPIO.OUT) # 초음파 출력
GPIO.setup(ECHO, GPIO.IN)  # 초음파 입력

# try내에 반복문을 이용해 송수신 설정 및 거리 측정
try:
    while True:

        GPIO.output(TRIG, False)
        time.sleep(0.5)

        GPIO.output(TRIG,True)
        time.sleep(0.00001)
        GPIO.output(TRIG,False)

        while GPIO.input(ECHO)==0:
        start = time.time()

        while GPIO.input(ECHO)==1:
        end = time.time()

        check_time = end - start
        distance = check_time * 17000
        print("거리 : %.1f cm" % distance)
        time.sleep(0.4)

except KeyboardInterrupt:
    print("거리 측정 완료")
    GPIO.cleanup()
