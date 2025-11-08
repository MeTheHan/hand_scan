import cv2
import mediapipe as mp
import numpy as np

# Load gambar telapak tangan 
template = cv2.imread('hand_template.png', cv2.IMREAD_UNCHANGED)
template_h, template_w = template.shape[:2]

# Inisialisasi kamera
cap = cv2.VideoCapture(0) 

# Mediapipe untuk deteksi tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# Fungsi untuk overlay gambar transparan
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

    # Hitung posisi tengah layar untuk gambar template
    center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2

    # Overlay template tangan
    overlay_image_alpha(frame, template, (center_x, center_y))

    # REKLAM FİLİGRANI - Sağ alt köşeye küçük YT kullanıcı adı
    text = "YT:@mtechnow"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    
    # Text boyutunu hesapla
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = w - text_size[0] - 10  # Sağdan 10 pixel içeride
    text_y = h - 10  # Alttan 10 pixel yukarı
    
    # Arka plan için küçük bir kutu (opsiyonel)
    cv2.rectangle(frame, 
                 (text_x - 5, text_y - text_size[1] - 5),
                 (text_x + text_size[0] + 5, text_y + 5),
                 (0, 0, 0), -1)
    
    # Beyaz text ekle
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    # Proses deteksi tangan
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Dapatkan bounding box dari landmark
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            # Bounding
            if (center_x < x_min < center_x + template_w and
                center_y < y_min < center_y + template_h and
                center_x < x_max < center_x + template_w and
                center_y < y_max < center_y + template_h):
                # Bounding box berada dalam area template
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
