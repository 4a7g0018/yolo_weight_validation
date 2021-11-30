import random
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from k_means import *
from polygon_drawing import *
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
from validation_index import *
import json
import glob
from Validation import Validation
import statistics
import time

CROP_SIZE = [3, 3]
TH = 5e-4
DRAW_BOX = True
BOX_ANS = True
DEAW_POLYGON = True
POLYGON_ANS = True

FOLD = 'C'
WEIGHTS_PATH = 'weight/{}/yolo-obj_final.weights'.format(FOLD)
# WEIGHTS_PATH = 'weight/{}/yolo-obj_6000.weights'.format(FOLD)
CONFIG_PATH = 'cfg/yolo-obj.cfg'
LABLE_PATH = 'names/obj.names'
LABLES = None

with open(LABLE_PATH, 'rt') as f:
    LABLES = f.read().rstrip('\n').split('\n')
NET = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)


def crop_image(image, crop_size_x, crop_size_y):
    mask_H = int(image.shape[0] / crop_size_y)
    mask_W = int(image.shape[1] / crop_size_x)
    mask = np.zeros((mask_H, mask_W, 3), dtype="float32")

    i = 0
    format_box_list = []
    format_box_center_list = []
    format_box_w_h_list = []
    format_confidences_list = []
    for crop_y in range(crop_size_y):
        for crop_x in range(crop_size_x):
            i += 1
            start_h = mask_H * crop_y
            end_h = mask_H * (crop_y + 1)
            start_w = mask_W * crop_x
            end_w = mask_W * (crop_x + 1)
            for h in range(start_h, end_h):
                for w in range(start_w, end_w):
                    mask[h - crop_y * mask_H, w - crop_x * mask_W] = image[h, w]

            # 送第一次裁切前先送整張
            if crop_y == 0 and crop_x == 0:
                format_box, format_box_center, format_box_w_h, format_confidences = verify_image(image, 0, 0)
                format_box_list += (format_box)
                format_box_center_list += format_box_center
                format_box_w_h_list += format_box_w_h
                format_confidences_list += format_confidences

            format_box, format_box_center, format_box_w_h, format_confidences = verify_image(mask, start_w, start_h)
            format_box_list += (format_box)
            format_box_center_list += format_box_center
            format_box_w_h_list += format_box_w_h
            format_confidences_list += format_confidences

    return format_box_list, format_box_center_list, format_box_w_h_list, format_confidences_list


def verify_image(image, x_range, y_range):
    (H, W) = image.shape[:2]
    ln = NET.getLayerNames()
    ln = [ln[i - 1] for i in NET.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    NET.setInput(blob)
    start = time.time()
    layer_output = NET.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    boxes = []
    confidences = []
    class_ids = []
    for output in layer_output:
        for detection in output:
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > TH:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_X, center_Y, wight, height) = box.astype('int')
                x = int(center_X - (wight / 2))
                y = int(center_Y - (height / 2))
                if wight < 100:
                    wight = 100
                if height < 100:
                    height = 100

                boxes.append([x + x_range, y + y_range, int(wight), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    format_box = []
    format_box_center = []
    format_box_w_h = []
    format_confidences = []
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, TH, TH)
    if len(idxs) > 0:
        for i in idxs.flatten():
            format_box.append(boxes[i])
            format_box_center.append([boxes[i][0], boxes[i][1]])
            format_box_w_h.append([boxes[i][2], boxes[i][3]])
            format_confidences.append(confidences[i])

    return format_box, format_box_center, format_box_w_h, format_confidences


def map_drawing_verify_box_to_image(image, class_map, format_confidences_list):
    color_list = []
    rand = random.Random(1337)
    for i in range(len(class_map)):
        color_list.append(np.array([rand.random() * 256, rand.random() * 256, rand.random() * 256]))
    numbe = 0
    for box_id in class_map:
        center = class_map[box_id]['center']
        size = class_map[box_id]['size']
        color = np.random.random(300).reshape((100, 3))

        for number in range(len(center)):
            numbe += 1
            (x, y) = (center[number][0], center[number][1])
            (w, h) = (size[number][0], size[number][1])
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color_list[box_id], 3)
            # alpha = float(format_confidences_list[number]) * 10
            alpha = 1
            text = '%.4f' % format_confidences_list[number]
            cv2.putText(overlay, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_list[box_id], 2)
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # cv2.imwrite('ttt2.png', image)
    return image


def map_drawing_verify_polygon_to_image(image, class_map, alpha=0.3):
    polygon_list = []

    for box_id in class_map:
        center = class_map[box_id]['center']
        size = class_map[box_id]['size']
        confidences = class_map[box_id]['confidences']
        points = []
        for number in range(len(center)):
            (x, y) = (center[number][0], center[number][1])
            (w, h) = (size[number][0], size[number][1])
            points.append([x, y])
            points.append([x, y + h])
            points.append([x + w, y])
            points.append([x + w, y + h])

        output = calculate_contour(points)
        polygon_list.append(output)

        confidences = statistics.median(confidences)

        color = [255, 0, 0]
        if confidences >= 0.1:
            alpha = 0.45
            color = [0, 60, 255]
        elif confidences >= 0.01:
            alpha = 0.4
            color = [0, 145, 255]
        elif confidences >= 0.001:
            alpha = 0.35
            color = [0, 220, 255]
        elif confidences >= 0.0001:
            alpha = 0.25
            color = [0, 250, 140]
        else:
            alpha = 1
            color = [255, 255, 255]

        overlay = image.copy()
        cv2.fillPoly(overlay, pts=[output], color=color)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image, polygon_list


def drawing_ans_to_image(ans_name, image, alpha=0.5):
    color = [255, 0, 0]
    (H, W) = image.shape[:2]
    outsize_h = 0
    outsize_w = 0
    if H != 2688:
        outsize_h = int((2688 - H) / 2)

    if W != 2688:
        outsize_w = int((2688 - W) / 2)

    with open('ANS/{}'.format(ans_name)) as json_data:
        json_content = json.load(json_data)
        for points in json_content['shapes']:
            if points['shape_type'] == 'polygon':
                polygon = np.array(points['points'], dtype=np.int32)
                if (outsize_h != 0):
                    for i in range(len(polygon)):
                        polygon[i][1] = polygon[i][1] - outsize_h

                if (outsize_w != 0):
                    for i in range(len(polygon)):
                        polygon[i][0] = polygon[i][0] - outsize_w

                overlay = image.copy()
                cv2.fillPoly(overlay, pts=[polygon], color=color)
                image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    json_data.close()
    # cv2.imwrite('ttt4.png', image)
    return image


def ComparePoint(a, b):
    if a[0] < b[0] or a[1] < b[1]:
        return True
    return False


def calculate_contour(points):
    hull = 0
    for i in range(1, len(points)):
        if ComparePoint(points[i], points[hull]):
            hull = i

    def is_left(a, b, c):
        u1 = b[0] - a[0]
        v1 = b[1] - a[1]
        u2 = c[0] - a[0]
        v2 = c[1] - a[1]
        return u1 * v2 - v1 * u2 < 0

    out_indices = [0 for _ in range(len(points))]
    end_point = -1
    i = 0

    while end_point != out_indices[0]:
        out_indices[i] = hull
        i = i + 1
        end_point = 0
        for j in range(1, len(points)):
            if hull == end_point or is_left(points[hull], points[end_point], points[j]):
                end_point = j

        hull = end_point

    contour = []
    for j in range(i):
        contour.append(points[out_indices[j]])

    return np.array(contour)


def main():
    data = 'val'
    all_names = []
    for image_name in glob.iglob('resource/{}/{}/*.jpg'.format(FOLD, data)):
        name = image_name.replace("resource/{}/{}\\".format(FOLD, data), "")
        all_names.append(name[:7])

    precision = 0
    recall = 0
    F1_score = 0
    tp_t = 0
    fp_t = 0
    fn_t = 0
    all_names = set(all_names)
    for name in all_names:
        start = time.time()

        name = name[:7]
        print(name)
        image = cv2.imread('resource/{}/{}/{}0.jpg'.format(FOLD, data, name))
        json_name = '{}.json'.format(name)

        format_box_list, format_box_center_list, format_box_w_h_list, format_confidences_list = crop_image(image,
                                                                                                           CROP_SIZE[0],
                                                                                                           CROP_SIZE[1])
        class_map = k_main(format_box_center_list, format_box_w_h_list, format_confidences_list)

        if DRAW_BOX:
            drawing_box_image = map_drawing_verify_box_to_image(image, class_map, format_confidences_list)
            cv2.imwrite('output/box_{}.png'.format(name), drawing_box_image)
            if BOX_ANS:
                add_ans_image = drawing_ans_to_image(json_name, drawing_box_image)
                cv2.imwrite('output/box_{}.png'.format(name), add_ans_image)

        if DEAW_POLYGON:
            drawing_polygon_image, polygon_list = map_drawing_verify_polygon_to_image(image, class_map)
            cv2.imwrite('output/polygon_{}.png'.format(name), drawing_polygon_image)
            if POLYGON_ANS:
                add_ans_image = drawing_ans_to_image(json_name, drawing_polygon_image)
                cv2.imwrite('output/polygon_{}.png'.format(name), add_ans_image)

        v = Validation(add_ans_image, json_name, polygon_list)
        v.draw_coincidence_rate_to_image('output/polygon_{}'.format(name))
        v.draw_confusion_matrix('output/polygon_{}'.format(name))

        coincidence_rate_map = v.get_coincidence_rate()
        get_confusion_matrix = v.get_confusion_matrix()
        confusion_matrix = v.get_confusion_matrix()
        end = time.time()
        print(confusion_matrix, "------time : ", end - start)
        print()

        precision += confusion_matrix['precision']
        recall += confusion_matrix['recall']
        F1_score += confusion_matrix['F1_score']
        tp_t += confusion_matrix['tp']
        fp_t += confusion_matrix['fp']
        fn_t += confusion_matrix['fn']

    print("________________________________________________________________________")
    print("------------------------------total-------------------------------------")

    print('tp :', tp_t)
    print('fp :', fp_t)
    print('fn :', fn_t)
    print('precision :', precision / len(all_names))
    print('recall :', recall / len(all_names))
    print('F1_score :', F1_score / len(all_names))


if __name__ == '__main__':
    main()
