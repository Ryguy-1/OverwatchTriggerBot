import cv2
from mss import mss
import numpy as np
import threading
import mouse


class ScreenRecord:
    def __init__(self, screen_resolution={'left': 0, 'top': 0, 'width': 1920, 'height': 1080}, save_data='Game_Data_3/', frame_delay=100):
        # Save Data Path
        self.save_data = save_data
        # Screen Resolution
        self.screen_resolution = screen_resolution
        # Set Frame Delay
        self.frame_delay = frame_delay
        # Is Recording Currently
        self.recording = False
        # Clicks / Non_Clicks
        self.clicks = 0
        self.non_clicks = 0
        # Start Record Thread
        self.record_thread = threading.Thread(target=self.record_loop)
        self.record_thread.start()

    def mouse_left_is_pressed(self):
        return mouse.is_pressed(mouse.LEFT)

    def start_recording(self):
        # Start Recording
        self.recording = True
    
    def stop_recording(self):
        # Stop Recording
        self.recording = False
    
    def get_screen_frame(self):
        with mss() as sct:
            # Get Screen Frame
            frame = np.array(sct.grab(self.screen_resolution))
        return frame[:, :, :3] # Drop Alpha Channel

    def save_frame(self, frame):
        is_pressed = False
        # Check if Left Click Pressed
        if self.mouse_left_is_pressed():
            # Left Click Pressed
            self.clicks += 1
            is_pressed = True
        else:
            # Left Click Not Pressed
            self.non_clicks += 1
        # Save Frame
        if is_pressed:
            cv2.imwrite(f'{self.save_data}/clicked/{self.clicks}.png', frame)
        else:
            cv2.imwrite(f'{self.save_data}/not_clicked/{self.non_clicks}.png', frame)

    def record_loop(self):
        # Get Current Screen Frame
        while True:
            if self.recording:
                # Get Screen Frame
                frame = self.get_screen_frame()
                # Save Frame
                self.save_frame(frame)
            # print(self.mouse_left_is_pressed())
            cv2.waitKey(100)
