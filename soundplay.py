import pygame
import time

pygame.mixer.init()
p = pygame.mixer.Sound("sam.wav")
pygame.mixer.music.set_volume(0.5)

while True:
    p.play()
    time.sleep(10)
