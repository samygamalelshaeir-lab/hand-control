"""
Hand Control: control your laptop with hand gestures using only the webcam.

Gestures:
  Thumbs up            -> Volume up   (repeats while held)
  Thumbs down          -> Volume down (repeats while held)
  Point right (index)  -> Right arrow
  Point left (index)   -> Left arrow
  Pinch thumb + index  -> Space bar   (one press per pinch)

IMPORTANT: arrow keys and space are sent to whichever window has focus.
Click on the window you want to control (video, slides, ...) so it has
focus, and keep the camera window visible beside it.

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
PINCH_THRESHOLD = 0.35     # thumb-index distance relative to hand size
POINT_MIN_LENGTH = 0.5     # index length (relative to hand size) to count as pointing
# ------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def detect_gesture(lm, w, h):
    """Return (gesture, pinch_ratio). Uses pixel coordinates so horizontal
    and vertical distances are measured equally."""

    def pt(i):
        return lm[i].x * w, lm[i].y * h

    def d(a, b):
        ax, ay = pt(a)
        bx, by = pt(b)
        return math.hypot(ax - bx, ay - by)

    size = d(0, 9)  # wrist -> middle finger base = hand size
    if size == 0:
        return None, 0.0

    def extended(tip, pip):
        return d(0, tip) > d(0, pip) * 1.15

    index = extended(8, 6)
    middle = extended(12, 10)
    ring = extended(16, 14)
    pinky = extended(20, 18)
    others_curled = not (middle or ring or pinky)

    pinch_ratio = d(4, 8) / size

    # 1) Pinch
    if pinch_ratio < PINCH_THRESHOLD:
        return "pinch", pinch_ratio

    # 2) Thumbs up / down
    if not index and others_curled:
        tx, ty = pt(4)
        bx, by = pt(2)
        dx, dy = tx - bx, ty - by
        if abs(dy) > abs(dx) and abs(dy) / size > 0.5:
            return ("thumb_up" if dy < 0 else "thumb_down"), pinch_ratio

    # 3) Pointing left / right
    if index and others_curled:
        tx, ty = pt(8)
        bx, by = pt(5)
        dx, dy = tx - bx, ty - by
        if abs(dx) > abs(dy) and abs(dx) / size > POINT_MIN_LENGTH:
            return ("point_right" if dx > 0 else "point_left"), pinch_ratio

    return None, pinch_ratio


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
    active = None
    last_fire = 0.0
    pinch_latched = False
    last_sent = ""
    last_sent_time = 0.0

    def send(key):
        nonlocal last_sent, last_sent_time
        pyautogui.press(key)
        last_sent = key
        last_sent_time = time.time()

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            candidate = None
            pinch_ratio = 0.0
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                candidate, pinch_ratio = detect_gesture(hand.landmark, w, h)

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
                    send("space")
                    pinch_latched = True
            elif active in ("thumb_up", "thumb_down"):
                if now - last_fire >= VOLUME_INTERVAL:
                    send("volumeup" if active == "thumb_up" else "volumedown")
                    last_fire = now
            elif active in ("point_right", "point_left"):
                if now - last_fire >= ARROW_INTERVAL:
                    send("right" if active == "point_right" else "left")
                    last_fire = now

            cv2.putText(frame, LABELS.get(active, "No gesture"), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"pinch ratio: {pinch_ratio:.2f}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            if now - last_sent_time < 1.0:
                cv2.putText(frame, f"sent: {last_sent}", (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow("Hand Control (press q to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
