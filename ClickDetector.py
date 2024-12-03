from Utilities import Utilities

class ClickDetector:
    def __init__(self):
        self.prev_touching = False
        self.release_time = None
        self.MIN_CLICK_TIME_THRESHOLD = 0.2  # Minimum time (seconds) between release and next touch for a click
        self.MAX_CLICK_TIME_THRESHOLD = 0.5  # Maximum time (seconds) between release and next touch for a click
        self.utilities = Utilities()

    def detect_touch(self, landmarks):
        finger_dist = self.utilities.get_finger_distance(landmarks[4], landmarks[8])
        ref_dist = self.utilities.get_reference_distance(landmarks[0], landmarks[5])
        normalized_dist = finger_dist / ref_dist

        TOUCH_THRESHOLD = 0.25  # Adjust based on testing
        return normalized_dist < TOUCH_THRESHOLD

    def detect_click(self, landmarks):
        """
        Detect click based on touch -> release -> quick touch pattern
        Returns: bool indicating if click should happen
        """
        import time
        current_time = time.time()
        is_touching = self.detect_touch(landmarks)

        click_detected = False

        if is_touching and not self.prev_touching:  # Touch just started
            if self.release_time and (
                    self.MIN_CLICK_TIME_THRESHOLD < (current_time - self.release_time) < self.MAX_CLICK_TIME_THRESHOLD):
                click_detected = True
            self.release_time = None

        elif not is_touching and self.prev_touching:  # Just released
            self.release_time = current_time

        self.prev_touching = is_touching
        return click_detected, is_touching