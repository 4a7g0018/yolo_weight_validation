import cv2
import numpy as np
import json
from shapely.geometry import Polygon


class Validation:
    def __init__(self, image, json_name, polygon_list):
        """
        參數需求:
            name            --圖片名稱(含路徑)。
            json_name       --該圖片搭配的 JSON 名稱(含路徑)，計算正解用。
            image           --圖片 array。
            polygon_list    --分群後的多邊形陣列。

        -------------------------------
        GET -可取得資源:

            get_confusion_matrix()   --取得混淆舉證。
            get_coincidence_rate()   --取得預判與正解的覆蓋率與繪製座標
                                        ( 可以使用 draw_coincidence_rate_to_image() 來繪製)。

        -------------------------------
        DRAW -可繪製資源(因為會自動存圖片所以需要給要儲存的圖片名稱):


            draw_coincidence_rate_to_image(save_name)   --繪製正解的覆蓋率在圖片上，會自動透過save_name來存圖片。
            draw_confusion_matrix(save_name)            --繪製recall、precision、F1_source、probability、color_box
                                                            在圖片上，會自動透過save_name來存圖片。
        """

        self.image = image
        self.json_name = json_name
        self.polygon_list = polygon_list

        self.polygon_ans_content_list = []
        self.polygon_ans_frame_list = []
        self.polygon_content_list = []
        self.coincidence_rate_map = {}
        self.confusion_matrix = {}

        self.drawing_ans_to_iou_image()
        self.map_drawing_class_polygon()
        self.calculate_coincidence_rate()
        self.calculate_confusion_matrix()

    def drawing_ans_to_iou_image(self):

        color = [255, 255, 255]
        (H, W) = self.image.shape[:2]

        outsize_h = 0
        outsize_w = 0
        if H != 2688:
            outsize_h = int((2688 - H) / 2)

        if W != 2688:
            outsize_w = int((2688 - W) / 2)
        polygon_ans_content_list = []
        polygon_ans_frame_list = []
        with open('ANS/{}'.format(self.json_name)) as json_data:
            json_content = json.load(json_data)
            for points in json_content['shapes']:
                if points['shape_type'] == 'polygon':
                    mask = np.zeros(self.image.shape)

                    # if (int(outsize_h) != 0 or int(outsize_w) != 0):
                    #     for i in range(len(polygon)):
                    #         polygon[i][0] -= outsize_w
                    #         polygon[i][1] -= outsize_h
                    if (outsize_h != 0):
                        for i in range(len(polygon)):
                            polygon[i][1] = polygon[i][1] - outsize_h

                    if (outsize_w != 0):
                        for i in range(len(polygon)):
                            polygon[i][0] = polygon[i][0] - outsize_w

                    content_point = Polygon(polygon).convex_hull
                    polygon_ans_content_list.append(content_point)
                    polygon_ans_frame_list.append(polygon)

        self.polygon_ans_content_list = polygon_ans_content_list
        self.polygon_ans_frame_list = polygon_ans_frame_list

    def map_drawing_class_polygon(self):
        image_number = 0

        for box in self.polygon_list:
            self.polygon_content_list.append(content_point)

    def calculate_coincidence_rate(self):

        has_check_ans = np.zeros(len(self.polygon_ans_content_list))
        for ans_number in range(len(self.polygon_ans_content_list)):

            # coincidence_rate init
            coincidence_rate_center = {}
            coincidence_rate_center['area'] = 0.

            try:
                coincidence_rate_center['show_point']
            except:
                coincidence_rate_center['show_point'] = min(np.ndarray.tolist(self.polygon_ans_frame_list[ans_number]))

            self.coincidence_rate_map[ans_number] = coincidence_rate_center

            for class_number in range(len(self.polygon_content_list)):

                # 判斷預測多邊形面積是否與正解相交
                if self.polygon_ans_content_list[ans_number].intersects(self.polygon_content_list[class_number]):

                    # 確認該點正解是否有被計算過，如果有責選面積覆蓋率大的
                    if has_check_ans[ans_number] != 0:
                        inter_area = self.polygon_ans_content_list[ans_number].intersection(
                            self.polygon_content_list[class_number]).area
                        if new_inter_area > old_inter_area:
                            coincidence_rate_center['area'] = new_inter_area
                    else:
                        # 計算相交面積與顯示位置
                        has_check_ans[ans_number] = 1
                        inter_area = self.polygon_ans_content_list[ans_number].intersection(
                            self.polygon_content_list[class_number]).area
                        coincidence_rate_center['area'] = round(
                            inter_area / self.polygon_ans_content_list[ans_number].area * 100, 2)
                        coincidence_rate_center['show_point'] = min(
                            np.ndarray.tolist(self.polygon_ans_frame_list[ans_number]))
                        self.coincidence_rate_map[ans_number] = coincidence_rate_center

    def calculate_confusion_matrix(self):
        confusion_matrix = {}
        tp = 0
        fp = 0
        fn = 0
        has_check_ans = np.zeros(len(self.polygon_ans_content_list))
        for class_number in range(len(self.polygon_content_list)):

            successful_predictions_quantity = 0

            for ans_number in range(len(self.polygon_ans_content_list)):

                # 如果預判與正重合
                if self.polygon_ans_content_list[ans_number].intersects(self.polygon_content_list[class_number]):

                    # 重合面積
                    inter_area = self.polygon_ans_content_list[ans_number].intersection(
                        self.polygon_content_list[class_number]).area
                    if inter_area / self.polygon_ans_content_list[ans_number].area * 100 > 30 and has_check_ans[
                        ans_number] != 1:
                        has_check_ans[ans_number] = 1
                        successful_predictions_quantity += 1
                        tp += 1

            if successful_predictions_quantity == 0:
                fp += 1

        fn = len(self.polygon_ans_content_list) - tp

        confusion_matrix['tp'] = tp
        confusion_matrix['fp'] = fp
        confusion_matrix['fn'] = fn

        recall = round(tp / (tp + fn), 3)
        precision = round(tp / (tp + fp), 3)

        if recall == 0 and precision == 0:
            f1_sorce = 0
        else:
            f1_sorce = round(2 * (recall * precision) / (recall + precision), 3)

        confusion_matrix['recall'] = recall
        confusion_matrix['precision'] = precision
        confusion_matrix['F1_score'] = f1_sorce

        self.confusion_matrix = confusion_matrix

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_coincidence_rate(self):
        return self.coincidence_rate_map

    def draw_coincidence_rate_to_image(self, save_name):
        for coincidence_rate_number in self.coincidence_rate_map:

            center = self.coincidence_rate_map[coincidence_rate_number]
            text = '{}%'.format(round(center['area'], 0))

            if center['show_point'] != 0:
                x, y = center['show_point']
                self.image = cv2.putText(self.image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 0], 3)

        cv2.imwrite('{}.png'.format(save_name), self.image)

    def draw_confusion_matrix(self, image_name):
        alpha = 0.5

        text_recall = 'recall : {}%'.format(round(self.confusion_matrix['recall'] * 100, 6))
        text_precision = 'precision : {}%'.format(round(self.confusion_matrix['precision'] * 100, 6))
        text_f1_sorce = 'F1_score : {}%'.format(round(self.confusion_matrix['F1_score'] * 100, 6))

        # 圖片加border
        self.image = cv2.copyMakeBorder(self.image, 400, 0, 0, 0, cv2.BORDER_CONSTANT)

        # text_recall,text_precision,text_f1_sorce
        overlay = self.image.copy()
        cv2.rectangle(overlay, (0, 300), (640, 0), (0, 0, 0), -1)
        self.image = cv2.addWeighted(overlay, alpha, self.image, 1 - alpha, 0)
        cv2.putText(self.image, text_recall, (10, 0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, [255, 255, 0], 5)
        cv2.putText(self.image, text_precision, (10, 0 + 150), cv2.FONT_HERSHEY_SIMPLEX, 1.8, [255, 255, 0], 5)
        cv2.putText(self.image, text_f1_sorce, (10, 0 + 250), cv2.FONT_HERSHEY_SIMPLEX, 1.8, [255, 255, 0], 5)

        # color box
        alpha = 1
        text_probability = 'probability :'
        text_high = 'High'
        text_medium = 'Medium'
        text_low = 'Low'
        text_very_low = 'Very Low'

        box1 = self.image.copy()

        x_start = 10
        y_start = 380
        cv2.rectangle(box1, (x_start + 400, y_start + 10), (x_start + 450, y_start - 50), (0, 60, 255), -1)
        cv2.rectangle(box1, (x_start + 670, y_start + 10), (x_start + 720, y_start - 50), (0, 145, 255), -1)
        cv2.rectangle(box1, (x_start + 1020, y_start + 10), (x_start + 1070, y_start - 50), (0, 220, 255), -1)
        cv2.rectangle(box1, (x_start + 1320, y_start + 10), (x_start + 1370, y_start - 50), (0, 250, 140), -1)
        self.image = cv2.addWeighted(box1, alpha, self.image, 1 - alpha, 0)

        cv2.putText(self.image, text_probability, (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 1.8, [255, 255, 255], 5)
        cv2.putText(self.image, text_high, (x_start + 460, y_start), cv2.FONT_HERSHEY_SIMPLEX, 1.8, [255, 255, 255], 5)
        cv2.putText(self.image, text_medium, (x_start + 730, y_start), cv2.FONT_HERSHEY_SIMPLEX, 1.8, [255, 255, 255],
                    5)
        cv2.putText(self.image, text_low, (x_start + 1080, y_start), cv2.FONT_HERSHEY_SIMPLEX, 1.8, [255, 255, 255], 5)
        cv2.putText(self.image, text_very_low, (x_start + 1380, y_start), cv2.FONT_HERSHEY_SIMPLEX, 1.8,
                    [255, 255, 255], 5)

        cv2.imwrite('{}.png'.format(image_name), self.image)
