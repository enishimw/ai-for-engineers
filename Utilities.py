import pyautogui
import numpy as np


class Utilities:
    def __int__(self):
        self.screen_width, self.screen_height = pyautogui.size()

    def get_finger_distance(self, thumb_tip, index_tip):
        return np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

    def get_reference_distance(self, wrist, index_mcp):
        return np.sqrt((wrist.x - index_mcp.x) ** 2 + (wrist.y - index_mcp.y) ** 2)

    def get_cursor_position(self, thumb_tip, index_tip):
        # Use the midpoint between thumb and index finger as cursor position
        cursor_x = (thumb_tip.x + index_tip.x) / 2
        cursor_y = (thumb_tip.y + index_tip.y) / 2

        # Increase sensitivity - smaller movements cover more screen space
        sensitivity = 4  # Increase this value for more sensitivity

        # Center the input range around 0.5 and apply sensitivity
        cursor_x = 0.5 + (cursor_x - 0.5) * sensitivity
        cursor_y = 0.5 + (cursor_y - 0.5) * sensitivity

        # Clamp values between 0 and 1
        cursor_x = max(0, min(1, cursor_x))
        cursor_y = max(0, min(1, cursor_y))

        # Map to screen coordinates
        screen_x = int(cursor_x * self.screen_width)
        screen_y = int(cursor_y * self.screen_height)

        return screen_x, screen_y

    def smooth_movement(self, current_pos, target_pos, smoothing_factor=0.5):
        """Apply smoothing to cursor movement to reduce jitter"""
        x = current_pos[0] + (target_pos[0] - current_pos[0]) * smoothing_factor
        y = current_pos[1] + (target_pos[1] - current_pos[1]) * smoothing_factor

        return (int(x), int(y))
