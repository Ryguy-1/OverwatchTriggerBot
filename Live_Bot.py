import cv2
from mss import mss
import numpy as np
import threading
import mouse
from numpy.lib.npyio import load
import pickle
import time
import pyautogui


class Live_Bot:
    def __init__(self, resolution = (1920, 1080), screen_resolution={'left': 0, 'top': 0, 'width': 1920, 'height': 1080}):
        self.resolution = resolution
        self.screen_resolution = screen_resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]
        self.model = load_model()
        self.scalar = load_scalar()

        # Reload Wait Timer
        self.last_fired_time = time.time()
        # # Cree
        self.fire_rate = 0.4
        # Widow
        # self.fire_rate = 1.3

        # Last 5 Guesses (0 for no click, 1 for click)
        self.buffer = [0, 0, 0, 0, 0]
        self.threshold = 0.8
        self.threshold_tracking_release = 0
        self.threshold_tracking_press = 0.7

        # Capturing
        self.capturing = True
        # self.capture_thread = threading.Thread(target=self.capture_loop)
        # self.capture_thread.start()
        self.capture_loop()

    def start_capture(self):
        self.capturing = True

    def end_capture(self):
        self.capturing = False

    def apply_transforms(self, image):
        # Resize Image
        image = image[round(self.y_res*2/5):round(self.y_res*3/5), round(self.x_res*2/5):round(self.x_res*3/5)]
        image = image[:, :, :3] # Drop Alpha Channel
        # b, g, image = cv2.split(image)
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        # (b, g, image) = cv2.split(image)
        # Flatten Image
        image = image.flatten()
        # Reshape Image
        image = image.reshape(1, -1)
        # Scalar
        # image = self.scalar.transform(image)

        # Return Image
        return image

    def get_screen_frame(self):
        with mss() as sct:
            # Get Screen Frame
            frame = np.array(sct.grab(self.screen_resolution))
        return frame

    def predict_and_click(self, frame):
        # Predict
        prediction = self.model.predict(frame)
        # Click Mouse if Click is True
        if prediction[0] == 0:
            # Update certainty array (shifts last 4 over right one and adds new value to beginning)
            self.buffer = [1] + self.buffer[:-1]
            # Check to See if 1.5 seconds has elapsed
            if time.time() - self.last_fired_time > self.fire_rate:
                # print("Click")
                # If sure, fire
                print(np.array(self.buffer))
                
                if np.average(np.array(self.buffer)) >= self.threshold:
                    # Widow only
                    # if mouse.is_pressed('right'):
                    #     mouse.click(button='left')
                    #     self.last_fired_time = time.time()
                    # Other Characters
                    mouse.click(button='left')
                    self.last_fired_time = time.time()

             # Tracking
            # self.buffer = [1] + self.buffer[:-1]
            # if np.average(np.array(self.buffer)) >= self.threshold_tracking_press:
            #     pyautogui.mouseDown()
            #     print("Press")
        else:       
            self.buffer = [0] + self.buffer[:-1]
            print(np.array(self.buffer))
            print("No Click")

            # # Tracking
            # self.buffer = [0] + self.buffer[:-1]
            # if np.average(np.array(self.buffer)) <= self.threshold_tracking_release:
            #     pyautogui.mouseUp()
            #     print("Release")

    def capture_loop(self):
        # Get Current Screen Frame
        while True:
            if self.capturing:
                # Get Screen Frame
                frame = self.get_screen_frame()
                # Apply Transforms
                frame = self.apply_transforms(frame)
                # Predict and Click Mouse
                self.predict_and_click(frame)

                # # FOR WIDOW ONLY
                # if mouse.is_pressed('right') and time.time() - self.last_fired_time > self.fire_rate:
                #     self.last_fired_time = time.time()

            # print(self.mouse_left_is_pressed())
            cv2.waitKey(20)

def load_model():
    with open('model_4.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

def load_scalar():
    with open('scalar_6.pickle', 'rb') as f:
        scalar = pickle.load(f)
    return scalar
