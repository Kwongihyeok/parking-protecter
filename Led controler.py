from gpiozero import LED
from time import sleep

ledset = [LED(17), LED(27), LED(22)]
mode = [0, 0, 0]

def LEDToggle(n):
    if mode[n] == 0:
        ledset[n].on()
    else:
        ledset[n].off()

    mode[n] = not mode[n]  # on/off 확인용도

while True:
    n = int(input())  # 문자로 입력받기 때문에 정수로 바꿔주기
    LEDToggle(n - 1)  # 0번 index부터 시작하니 -1 해주기
