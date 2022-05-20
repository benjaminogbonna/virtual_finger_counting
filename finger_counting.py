import os
import cv2
import mediapipe as mp
import time
import hand_tracking_module as htm

cap = cv2.VideoCapture(0)
finger_folder = 'fingers'
my_list = os.listdir(finger_folder)

overlay_list = []
prev_time = 0

for image_path in my_list:
    image = cv2.imread(f'{finger_folder}/{image_path}')
    overlay_list.append(image)

# print(len(overlay_list))
detector = htm.HandDetector(detection_con=.8)
tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    land_marks = detector.find_position(img, draw=False)

    if len(land_marks) > 0:
        fingers = []
        # thumb
        if land_marks[tip_ids[0]][1] > land_marks[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # for the other four fingers
        for id in range(1, 5):
            if land_marks[tip_ids[id]][2] < land_marks[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        total_fingers = fingers.count(1)
        # print(total_fingers)

        h, w, c = overlay_list[total_fingers-1].shape
        img[0:h, 0:w] = overlay_list[total_fingers-1]
        # cv2.rectangle(img, (10, 100), (80, 10), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (10, 250), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 10)
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time

    cv2.putText(img, str(int(fps)), (550, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Stopped at 4:05:46
