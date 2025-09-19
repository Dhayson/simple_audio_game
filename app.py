import pygame
import os
import random
pygame.init()

# Audio Teachable Machine model setup
# Import libraries
import numpy as np

# Import tensorflow and keras libraries
import tensorflow as tf
import sounddevice as sd
import pyaudio
import resample
import datetime

# Seleciona o dispositivo
# Execute details.py para verificar o index do dispositivo que deseja usar
sd.default.device = 3

def get_microphone_sample_rate(device_index=None):
    """
    Retrieves the default sample rate for a given microphone device.
    If no device_index is provided, it attempts to get the default input device's sample rate.
    """
    p = pyaudio.PyAudio()
    try:
        if device_index is None:
            # Try to get the default input device info
            info = p.get_default_input_device_info()
            device_index = info['index']
        else:
            info = p.get_device_info_by_index(device_index)

        # The 'defaultSampleRate' key contains the default sample rate
        sample_rate = int(info['defaultSampleRate'])
        print("sample rate is ", sample_rate)
        return sample_rate
    except Exception as e:
        print(f"Error getting microphone sample rate: {e}")
        return None
    finally:
        p.terminate()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels.
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()
   
def classify_sample(audio_data, samplerate):
    # Ensure the audio is mono and resample if necessary
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0] # Take one channel
    if samplerate != 44032: # Assuming shape[1] is the expected sample rate
        # You might need to resample the audio here using libraries like librosa
        audio_data = resample.resample(audio_data, samplerate, 44032)

    # Pad or truncate the audio to the expected length (1 second)
    expected_length = 44032
    if len(audio_data) < expected_length:
        audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))
    elif len(audio_data) > expected_length:
        audio_data = audio_data[:expected_length]

    # Normalize audio data if required by your model
    # Teachable Machine models expect float32 input in a specific range (-1 to 1)
    input_data = audio_data.astype(np.float32)
    input_data = (input_data - 0.5)*2
    input_data = np.expand_dims(input_data, axis=0) # Add batch dimension

    # Set the tensor.
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference.
    interpreter.invoke()

    # Get the output tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Process the output (e.g., get the most likely class).
    predicted_class_index = np.argmax(output_data)
    predicted_label = labels[predicted_class_index]
    confidence = output_data[0][predicted_class_index]

    print(f"Predicted class: {predicted_label} with confidence: {confidence:.2f}")
    return predicted_class_index

audio_rate = get_microphone_sample_rate()
# Coleta 7 samples por segundo
sample_time_in_seconds = 0.14285714285 # 1/7


def listen_audio():
    # Record audio from the default microphone
    my_recording = sd.rec(int(sample_time_in_seconds * audio_rate), samplerate=audio_rate, channels=1, dtype='float32')
    return my_recording


# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect().scale_by(0.6)
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput, voiceInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if (userInput[pygame.K_UP] or voiceInput[2]) and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput[pygame.K_DOWN] and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput[pygame.K_DOWN]):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect().scale_by(0.6)
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect().scale_by(0.6)
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < - self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect().scale_by(0.6)
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 250
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1


def main():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 15
    x_pos_bg = 0
    y_pos_bg = 380
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0

    def score():
        global points, game_speed
        points += 1
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Points: " + str(points), True, (0, 0, 0))
        textRect = text.get_rect().scale_by(0.6)
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed

    voices = []
    idx = 0
    for i in range(0, 6):
        voices.append(listen_audio())
        sd.wait()
    voices.append(listen_audio())
    reference_datetime = datetime.datetime.now()
    while run:
        current_datetime = datetime.datetime.now()
        voiceInput = [False]*3
        if (current_datetime - reference_datetime).total_seconds() >= sample_time_in_seconds:
            # sd.wait()
            voice = np.concatenate(voices[idx:idx+6])
            voiceInput[classify_sample(voice, audio_rate)] = True
            reference_datetime = current_datetime
            voices.append(listen_audio())
            idx += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        SCREEN.fill((255, 255, 255))
        # print(userInput)

        player.draw(SCREEN)
        
        userInput = pygame.key.get_pressed()
        player.update(userInput, voiceInput)

        if len(obstacles) == 0:
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 2) == 2:
                obstacles.append(Bird(BIRD))

        for obstacle in obstacles:
            obstacle.draw(SCREEN)
            obstacle.update()
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(2000)
                death_count += 1
                menu(death_count)

        background()

        cloud.draw(SCREEN)
        cloud.update()

        score()

        clock.tick(30)
        pygame.display.update()


def menu(death_count):
    global points
    run = True
    while run:
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 30)

        if death_count == 0:
            text = font.render("Press any Key to Start", True, (0, 0, 0))
        elif death_count > 0:
            text = font.render("Press any Key to Restart", True, (0, 0, 0))
            score = font.render("Your Score: " + str(points), True, (0, 0, 0))
            scoreRect = score.get_rect().scale_by(0.6)
            scoreRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
            SCREEN.blit(score, scoreRect)
        textRect = text.get_rect().scale_by(0.6)
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        SCREEN.blit(text, textRect)
        SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False
            if event.type == pygame.KEYDOWN:
                main()


menu(death_count=0)
