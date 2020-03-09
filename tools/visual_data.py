import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import os
from PIL import Image
import json
import numpy as np


def show_boxes(im, bboxs, segs, img, color):

    # Display in largest to smallest order to reduce occlusion
    # min_area = 99999
    # bbox_min = [0,0,0,0]
    # for det in dets:
    #     bbox = det[:4]
    #     area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    #     if area < min_area:
    #         min_area = area
    #         bbox_min = bbox.copy()
    # ax.add_patch(
    #     plt.Rectangle((bbox_min[0], bbox_min[1]),
    #                     bbox_min[2] - bbox_min[0],
    #                     bbox_min[3] - bbox_min[1],
    #                     fill=False, edgecolor=color,
    #                     linewidth=2))
    for det in bboxs:
        bbox = np.array(det[:4]).astype(int)
        cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255,0,0), 1)
    # for det in segs:
    #     # bbox = det[:4]
    #     # cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
    #     det = np.array(det)
    #     det = det.reshape((-1,1,2))
    #     im = cv2.polylines(im,[det],True,color, 2)
    # cv2.imwrite('train_draw/'+img, im)
    if im.shape[0] > 1000:
        im = cv2.resize(im, (int(0.5*im.shape[1]), int(0.5*im.shape[0])))
    cv2.imshow('img', im)
    cv2.waitKey(0)

    return im


root_dir = 'data/train/image/'
# root_dir = 'submit/test_detection1'
images = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
label_file = "data/train/annotations/train.json"
labels = json.load(open(label_file, 'r'))
imgpath2id = {}
for img in labels["images"]:
    imgpath2id[img["file_name"]] = img["id"]

bboxs_imgid = {}
segs_imgid = {}
for anno in labels["annotations"]:
    if anno["image_id"] not in bboxs_imgid.keys():
        bboxs_imgid[anno["image_id"]] = []
        segs_imgid[anno["image_id"]] = []
    bboxs_imgid[anno["image_id"]].append(anno["bbox"])
    # segs_imgid[anno["image_id"]].append(anno["minAreaRect"])
# print('201908302_1d4495a0b2ae68070201908300504594OK-2.jpg' in imgpath2id)
# print(imgpath2id['201908302_1d4495a0b2ae68070201908300504594OK-2.jpg'])
# print(bboxs_imgid[imgpath2id['201908302_1d4495a0b2ae68070201908300504594OK-2.jpg']])
import cv2
for img in images:
    if 'vertical_flips' in img:
        continue
    # if '-' not in img:
    #     continue
    im = cv2.imread(img)
    assert im is not None
    # im = show_boxes(im, bboxs_imgid[imgpath2id[os.path.basename(img)]], segs_imgid[imgpath2id[os.path.basename(img)]], os.path.basename(img), color = (0, 0, 255))
    try:
        print(img)
        im = show_boxes(im, bboxs_imgid[imgpath2id[os.path.basename(img)]], segs_imgid, os.path.basename(img), color = (0, 0, 255))
    except:
        print(img+'err!!!')