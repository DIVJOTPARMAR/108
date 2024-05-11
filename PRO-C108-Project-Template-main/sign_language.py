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
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Get the x and y positions of the fingertips
            fingertips_pos = [lm_list[i].y * h for i in finger_tips]
            thumb_tip_pos = lm_list[thumb_tip].y * h

            # Check if the finger is folded or not
            for i in range(len(finger_tips)):
                if fingertips_pos[i] < thumb_tip_pos:
                    finger_fold_status[i] = True
                else:
                    finger_fold_status[i] = False

            # Draw circles around the fingertips
            for i in range(len(finger_tips)):
                cv2.circle(img, (int(lm_list[finger_tips[i]].x * w), int(lm_list[finger_tips[i]].y * h)), 10, (255, 0, 0), -1)

                # Check if the finger is folded or not
                if finger_fold_status[i]:
                    cv2.circle(img, (int(lm_list[finger_tips[i]].x * w), int(lm_list[finger_tips[i]].y * h)), 10, (0, 255, 0), -1)

            # Check if all fingers are folded
            if all(finger_fold_status):
                # Check if the thumb is raised up or down
                if lm_list[thumb_tip].y * h < lm_list[thumb_tip - 1].y * h:
                    print("LIKE")
                    cv2.putText(img, "LIKE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print("DISLIKE")
                    cv2.putText(img, "DISLIKE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("hand tracking", img)
    key = cv2.waitKey(1)
    if key == 32: # Spacebar key
        cv2.destroyAllWindows()
        break