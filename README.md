# Hand Control

Control your laptop with hand gestures, using only the webcam.

Built with Python, [MediaPipe](https://developers.google.com/mediapipe) (hand tracking), OpenCV (camera) and pyautogui (keyboard control).

## Gestures

| Gesture | What it does |
|---|---|
| 👍 Thumbs up | Volume up (keeps going while you hold it) |
| 👎 Thumbs down | Volume down (keeps going while you hold it) |
| ☝️ Point right with your index finger | Right arrow |
| ☝️ Point left with your index finger | Left arrow |
| 🤏 Pinch thumb and index together | Space bar (one press per pinch) |

Press `q` in the camera window to quit.

## Quick start (Windows, no Python needed)

1. Go to the [Releases](../../releases) page.
2. Download `HandControl.exe`.
3. Double-click it. The first launch takes 10-30 seconds.

> Windows may show a "protected your PC" warning because the app is not code-signed. Click **More info → Run anyway**.

## Run from source

Requires Python 3.9 - 3.11.

```bash
   git clone https://github.com/samygamalelshaeir-lab/hand-control.git
   cd hand-control
   pip install -r requirements.txt
   python hand_control.py
```

## How it works

1. MediaPipe detects 21 landmarks on your hand in every webcam frame.
2. `detect_gesture()` compares distances between landmarks (for example, a finger is extended when its tip is farther from the wrist than its middle joint). Distances are divided by hand size, so it works at any distance from the camera.
3. A gesture must stay stable for a few frames before it triggers (debounce), which avoids accidental presses.
4. pyautogui sends the matching key press.

## Settings

At the top of `hand_control.py`:

| Setting | Meaning |
|---|---|
| `PINCH_THRESHOLD` | How close thumb and index must be to count as a pinch |
| `STABLE_FRAMES` | Frames a gesture must hold before it triggers |
| `VOLUME_INTERVAL` | Seconds between volume steps while holding |
| `ARROW_INTERVAL` | Seconds between arrow presses while holding |

## Build the .exe yourself

```bash
pip install pyinstaller
python -m PyInstaller --onefile --name HandControl --collect-all mediapipe hand_control.py
```

The result is in `dist/HandControl.exe`.

## Notes

- Volume keys work on Windows. On macOS, pyautogui cannot press volume keys.
- Close other apps that use the camera (Zoom, browser tabs) before running.

## License

MIT
