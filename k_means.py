import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def k_main(boxes_list, format_box_w_h_list,format_confidences_list):
    '''將方框進行 k means 分群再以map回傳'''
    test_blobs_x = boxes_list
    k_range = range(2, int(len(test_blobs_x) / 3))

    distortions = []
    scores = []
    for i in k_range:
        k_means = KMeans(n_clusters=i).fit(test_blobs_x)
        distortions.append(k_means.inertia_)
        scores.append(silhouette_score(test_blobs_x, k_means.predict(test_blobs_x)))


    class_map = {}
    box_center_map = {}

    box_center = []
    box_size = []


    for class_id in range(max(new_item)):
        confidences = []
        for id in range(len(new_item)):
            if new_item[id]==class_id:
                box_center += [boxes_cneter[id]]
                box_size += [format_box_w_h_list[id]]
                confidences+=[format_confidences_list[id]]
        box_center_map['center'] = box_center
        box_center_map['size'] = box_size
        box_center_map['confidences']=confidences
        class_map[class_id] = box_center_map

        box_center_map = {}
        box_center = []
        box_size = []

    return class_map
