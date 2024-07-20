import pygame
import time

# Initialize the mixer module in pygame
pygame.mixer.init()

# Load the mp3 file
pygame.mixer.music.load("audio.mp3")

# Start playing the music
pygame.mixer.music.play()

# Print numbers from 0 to 10
for i in range(11):
    print(i)
    time.sleep(1)  # Wait for 1 second

# Stop the music after the loop ends
pygame.mixer.music.stop()
