import cv2
import mediapipe as mp
import pyautogui
import time

from SignLanguageDetector import SignLanguageDetector
from ClickDetector import ClickDetector
from Utilities import Utilities

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_drawing = mp.solutions.drawing_utils

pyautogui.FAILSAFE = False


# Main Loop
def main():
    click_detector = ClickDetector()
    sign_detector = SignLanguageDetector(model_path="asl_model.pth")
    utilities = Utilities()
    cap = cv2.VideoCapture(0)

    current_word = []
    last_letter = None
    last_letter_time = time.time()
    letter_cooldown = 1.0  # Seconds between letters

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # mirror image
        image = cv2.flip(image, 1)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(image)

        # Convert back to BGR for display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # hand check
                handedness = results.multi_handedness[results.multi_hand_landmarks.index(landmarks)].classification[0].label

                mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

                if handedness == "Right":
                    # Check for click gesture
                    click_detected, is_touching = click_detector.detect_click(landmarks.landmark)

                    if click_detected:
                        print("clicked")
                        pyautogui.click()

                    # Move cursor when touching
                    if is_touching:
                        target_x, target_y = utilities.get_cursor_position(landmarks.landmark[4], landmarks.landmark[8])
                        current_pos = pyautogui.position()
                        smooth_x, smooth_y = utilities.smooth_movement(current_pos, (target_x, target_y))
                        pyautogui.moveTo(smooth_x, smooth_y)
                elif handedness == "Left":  # right keyboard
                    predicted_letter = sign_detector.predict_letter(landmarks.landmark)
                    if predicted_letter:
                        current_time = time.time()
                        if (last_letter != predicted_letter and
                                current_time - last_letter_time > letter_cooldown):
                            current_word.append(predicted_letter)
                            last_letter = predicted_letter
                            print(f"Predicted letter", predicted_letter)
                            last_letter_time = current_time

        cv2.imshow("Gesture Recognition", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
