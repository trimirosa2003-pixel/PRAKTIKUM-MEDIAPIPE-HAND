import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

while True:
    ret, img = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            label = handedness.classification[0].label  # Left / Right

            lm = hand_landmarks.landmark
            thumb_tip_x = lm[4].x
            thumb_mcp_x = lm[2].x

            posisi = ""

            if label == "Right":
                # Logika tangan kanan
                if thumb_tip_x < thumb_mcp_x:
                    posisi = "TELAPAK (DEPAN)"
                else:
                    posisi = "PUNGGUNG (BELAKANG)"

            else:
                # Logika tangan kiri
                if thumb_tip_x > thumb_mcp_x:
                    posisi = "TELAPAK (DEPAN)"
                else:
                    posisi = "PUNGGUNG (BELAKANG)"

            cv2.putText(img, posisi, (50, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("Deteksi Depan / Belakang", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
