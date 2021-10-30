import cv2
from mss import mss
import numpy as np
import threading
import mouse
from numpy.lib.npyio import load
import pickle
import time


class Live_Bot:
    def __init__(self, resolution = (1920, 1080), screen_resolution={'left': 0, 'top': 0, 'width': 1920, 'height': 1080}):
        self.resolution = resolution
        self.screen_resolution = screen_resolution
        self.x_res = resolution[0]
        self.y_res = resolution[1]
        self.model = load_model()

        # Capturing
        self.capturing = True
        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.start()

    def start_capture(self):
        self.capturing = True

    def end_capture(self):
        self.capturing = False

    def apply_transforms(self, image):
        # Resize Image
        image = image[round(self.y_res*2/5):round(self.y_res*3/5), round(self.x_res*2/5):round(self.x_res*3/5)]
        image = image[:, :, :3] # Drop Alpha Channel
        # (b, g, image) = cv2.split(image)
        # Flatten Image
        image = image.flatten()
        # Reshape Image
        image = image.reshape(1, -1)
        
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
            print("Click")
            mouse.click(button='left')
        else:
            print("No Click")

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
            # print(self.mouse_left_is_pressed())
            cv2.waitKey(100)

def load_model():
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model
