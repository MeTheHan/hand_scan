# 🤖 Hand Scanner Pro - Hand Gesture Video Trigger

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8%2B-orange)
![AI Powered](https://img.shields.io/badge/AI-Powered-red)

**Channel: MTechno - @mtechnow**

</div>

## 🎯 What Does It Do?
- Captures your hand via webcam
- You bring your hand close to the hand template on screen
- When you place your hand over the template, video plays automatically
- Like magic! 🪄

## 🚀 Quick Start

### 1. Install Requirements:
```bash
pip install opencv-python mediapipe numpy
```

### 2. Prepare Files:
- `hand_template.png` (hand image with transparent background)
- `vid.mp4` (video to be played)

### 3. Run the Code:
```python
import cv2
import mediapipe as mp
import numpy as np

# Load hand template
template = cv2.imread('hand_template.png', cv2.IMREAD_UNCHANGED)
template_h, template_w = template.shape[:2]

# Initialize camera
cap = cv2.VideoCapture(0) 

# Mediapipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# Function for transparent image overlay
def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    alpha_overlay = img_overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1], c] = (
            alpha_overlay * img_overlay[:, :, c] +
            alpha_background * img[y:y+img_overlay.shape[0], x:x+img_overlay.shape[1], c]
        )

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontal for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Calculate center position for template
    center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2

    # Overlay hand template
    overlay_image_alpha(frame, template, (center_x, center_y))

    # WATERMARK - Bottom right corner
    text = "YT:@mtechnow"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = w - text_size[0] - 10
    text_y = h - 10
    
    cv2.rectangle(frame, 
                 (text_x - 5, text_y - text_size[1] - 5),
                 (text_x + text_size[0] + 5, text_y + 5),
                 (0, 0, 0), -1)
    
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    # Process hand detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            if (center_x < x_min < center_x + template_w and
                center_y < y_min < center_y + template_h and
                center_x < x_max < center_x + template_w and
                center_y < y_max < center_y + template_h):
                video = cv2.VideoCapture('vid.mp4')
                while video.isOpened():
                    ret_vid, frame_vid = video.read()
                    if not ret_vid:
                        break
                    cv2.imshow('Hand Scanner', frame_vid)
                    if cv2.waitKey(30) & 0xFF == 27:
                        break
                video.release()

    cv2.imshow('Hand Scanner', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

## ⚡ How to Use:
1. Run the code
2. Show your hand to the camera
3. Bring your hand close to the hand template on screen
4. Video plays automatically!

## 🛠️ Features:
- **AI Hand Detection** - Powered by MediaPipe
- **Real-time Processing** - Smooth performance
- **Touchless Interaction** - No physical contact needed
- **Customizable** - Easy to modify templates and videos

## 🎯 Tips:
- `hand_template.png` should be transparent PNG
- `vid.mp4` can be any video you want
- Press `ESC` to exit

## 📁 Project Structure:
```
hand-scanner/
├── hand_scanner.py
├── hand_template.png
└── vid.mp4
```

## 🔧 Customization:
- Change watermark text: `text = "Your Text Here"`
- Modify template: `hand_template.png`
- Change trigger video: `vid.mp4`
- Adjust detection sensitivity: `min_detection_confidence=0.7`

## 📞 Contact:
**YouTube: @mtechnow**

---

**Copy - Paste - Run! 🚀**

*Code is ready to use, just run and enjoy!*

## ⭐ Support:
If you like this project, don't forget to give it a star! ⭐

---

**Perfect for:**
- Interactive displays
- Educational projects
- Gesture control systems
- AI and computer vision learning

*No license restrictions - Feel free to use and modify!*
