"""
Hand Control: control your laptop with hand gestures using only the webcam.

Gestures:
  Thumbs up            -> Volume up   (repeats while held)
  Thumbs down          -> Volume down (repeats while held)
  Point right (index)  -> Right arrow
  Point left (index)   -> Left arrow
  Pinch thumb + index  -> Space bar   (one press per pinch)

Press 'q' in the preview window to quit.
"""

import math
import time

import cv2
import mediapipe as mp
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# ---------- Settings ----------
CAMERA_INDEX = 0
STABLE_FRAMES = 3          # frames a gesture must stay the same before it counts
VOLUME_INTERVAL = 0.12     # seconds between volume steps while holding
ARROW_INTERVAL = 0.6       # seconds between arrow presses while holding
PINCH_THRESHOLD = 0.28     # thumb-index distance relative to hand size
# ------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_extended(lm, tip, pip):
    """A finger is extended if its tip is farther from the wrist than its PIP joint.
    Works for any hand orientation."""
    return dist(lm[0], lm[tip]) > dist(lm[0], lm[pip]) * 1.1


def detect_gesture(lm):
    """Return one of: 'pinch', 'thumb_up', 'thumb_down', 'point_right',
    'point_left', or None."""
    size = dist(lm[0], lm[9])  # wrist -> middle finger base = hand size
    if size == 0:
        return None

    index = finger_extended(lm, 8, 6)
    middle = finger_extended(lm, 12, 10)
    ring = finger_extended(lm, 16, 14)
    pinky = finger_extended(lm, 20, 18)

    # 1) Pinch: thumb tip close to index tip
    if dist(lm[4], lm[8]) / size < PINCH_THRESHOLD:
        return "pinch"

    others_curled = not (middle or ring or pinky)

    # 2) Thumbs up / down: all four fingers curled, thumb pointing vertically
    if not index and others_curled:
        dx = lm[4].x - lm[2].x
        dy = lm[4].y - lm[2].y
        if abs(dy) > abs(dx) and abs(dy) / size > 0.5:
            return "thumb_up" if dy < 0 else "thumb_down"

    # 3) Pointing: only index finger extended, direction horizontal
    if index and others_curled:
        dx = lm[8].x - lm[5].x
        dy = lm[8].y - lm[5].y
        if abs(dx) > abs(dy) and abs(dx) / size > 0.6:
            return "point_right" if dx > 0 else "point_left"

    return None


LABELS = {
    "pinch": "Space",
    "thumb_up": "Volume Up",
    "thumb_down": "Volume Down",
    "point_right": "Right Arrow",
    "point_left": "Left Arrow",
}


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open the webcam.")
        return

    last_candidate = None
    stable_count = 0
    active = None            # confirmed gesture
    last_fire = 0.0
    pinch_latched = False    # makes pinch fire once per pinch

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror so left/right feel natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            candidate = None
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                candidate = detect_gesture(hand.landmark)

            # Debounce: require the same gesture for a few frames
            if candidate == last_candidate:
                stable_count += 1
            else:
                last_candidate = candidate
                stable_count = 1

            if stable_count >= STABLE_FRAMES:
                active = candidate

            if active != "pinch":
                pinch_latched = False

            now = time.time()
            if active == "pinch":
                if not pinch_latched:
                    pyautogui.press("space")
                    pinch_latched = True
            elif active in ("thumb_up", "thumb_down"):
                if now - last_fire >= VOLUME_INTERVAL:
                    pyautogui.press("volumeup" if active == "thumb_up" else "volumedown")
                    last_fire = now
            elif active in ("point_right", "point_left"):
                if now - last_fire >= ARROW_INTERVAL:
                    pyautogui.press("right" if active == "point_right" else "left")
                    last_fire = now

            text = LABELS.get(active, "No gesture")
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
            cv2.imshow("Hand Control (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
