
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------
# Sign Language Detection System
# Developed using MediaPipe & OpenCV
# Team Project | Contributor: Aadyasha Giri
# ----------------------------------------

import csv
import copy
import argparse
import itertools
from collections import Counter, deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.8)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)

    return parser.parse_args()


def main():
    print("Starting Sign Language Detection by Aadyasha...")

    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_labels = [row[0] for row in csv.reader(f)]

    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_labels = [row[0] for row in csv.reader(f)]

    fps_calc = CvFpsCalc(buffer_len=10)

    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_history = deque(maxlen=history_length)

    mode = 0

    while True:
        fps = fps_calc.get()

        key = cv.waitKey(10)
        if key == 27:
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_landmarks = pre_process_landmark(landmark_list)
                pre_history = pre_process_point_history(debug_image, point_history)

                logging_csv(number, mode, pre_landmarks, pre_history)

                hand_sign_id = keypoint_classifier(pre_landmarks)

                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_id = 0
                if len(pre_history) == history_length * 2:
                    finger_id = point_history_classifier(pre_history)

                finger_history.append(finger_id)
                most_common = Counter(finger_history).most_common()

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_labels[hand_sign_id],
                    point_labels[most_common[0][0]]
                )

        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        
        cv.imshow("Aadyasha's Sign Language Project", debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == 110:
        mode = 0
    if key == 107:
        mode = 1
    if key == 104:
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    h, w = image.shape[:2]
    points = np.array([[min(int(l.x * w), w-1), min(int(l.y * h), h-1)] for l in landmarks.landmark])
    x, y, bw, bh = cv.boundingRect(points)
    return [x, y, x + bw, y + bh]


def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[min(int(l.x * w), w-1), min(int(l.y * h), h-1)] for l in landmarks.landmark]


def pre_process_landmark(landmarks):
    temp = copy.deepcopy(landmarks)
    base_x, base_y = temp[0]

    temp = [[x - base_x, y - base_y] for x, y in temp]
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat))
    return [v / max_val for v in flat]


def pre_process_point_history(image, history):
    h, w = image.shape[:2]
    temp = copy.deepcopy(history)
    base_x, base_y = temp[0]

    temp = [[(x - base_x)/w, (y - base_y)/h] for x, y in temp]
    return list(itertools.chain.from_iterable(temp))


def logging_csv(number, mode, landmark_list, history_list):
    if mode == 1 and 0 <= number <= 9:
        with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
            csv.writer(f).writerow([number, *landmark_list])
    if mode == 2 and 0 <= number <= 9:
        with open('model/point_history_classifier/point_history.csv', 'a', newline="") as f:
            csv.writer(f).writerow([number, *history_list])


def draw_info(image, fps, mode, number):
    cv.putText(image, f"FPS:{fps}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

   
    cv.putText(image, "Aadyasha's Project", (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return image


if __name__ == '__main__':
    main()