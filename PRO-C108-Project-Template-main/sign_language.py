import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4
finger_fold_status = [False] * len(finger_tips)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = [lm for id, lm in enumerate(hand_landmark.landmark)]

            for tip in finger_tips:
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                if lm_list[tip].x < lm_list[tip - 3].x:
                    finger_fold_status[finger_tips.index(tip)] = True
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                else:
                    finger_fold_status[finger_tips.index(tip)] = False

            if all(finger_fold_status):
                if lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip - 2].y:
                    print("LIKE")
                    cv2.putText(img, "LIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                elif lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip - 2].y:
                    print("DISLIKE")
                    cv2.putText(img, "DISLIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("hand tracking", img)
    key = cv2.waitKey(1)
    if key == 32:  # Spacebar key
        cv2.destroyAllWindows()
        break
