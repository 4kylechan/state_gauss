import pygame
import time

try:
    pygame.mixer.init()
    print("mixer ok")
except Exception as e:
    print("mixer init hatasi:", e)
    raise

try:
    pygame.mixer.music.load("sounds/sag_alt.wav")
    print("dosya yuklendi")
except Exception as e:
    print("dosya yukleme hatasi:", e)
    raise

pygame.mixer.music.play()
print("ses cal komutu gitti")

time.sleep(3)
print("test bitti")
