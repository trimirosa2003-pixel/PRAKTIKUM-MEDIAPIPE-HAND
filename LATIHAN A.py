import cv2
import mediapipe as mp

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Convert BGR ke RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Proses deteksi tangan
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Gambar landmark tangan
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Klasifikasi kanan / kiri
            hand_label = results.multi_handedness[idx].classification[0].label

            if hand_label == "Right":
                cv2.putText(img, "KANAN", (200, 50),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            elif hand_label == "Left":
                cv2.putText(img, "KIRI", (200, 50),
                            cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("Deteksi Tangan", img)

    if cv2.waitKey(1) & 0xFF == 27:  # tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
