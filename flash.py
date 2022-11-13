import time
import math
from colorsys import hsv_to_rgb
from unicornhatmini import UnicornHATMini
import pygame

fl = UnicornHATMini()
fl.set_brightness(0.05)
a = 0
pygame.mixer.init()
pygame.mixer.load("testAudio.wav")

while True:
	pygame.mixer.music.play()
	for x in range(17):
		for y in range(7):
			fl.set_pixel(x,y,255,255,255)

	a += 1
	fl.show()
	time.sleep(1.5)
	fl.clear()
	fl.show()
	time.sleep(1.5)
	print(a)

