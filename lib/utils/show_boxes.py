# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------

# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------
import cv2
import matplotlib.pyplot as plt
from random import random as rand
def show_boxes(im, dets, classes, scale = 1.0):  
    outputs = []  
    for cls_idx, cls_name in enumerate(classes):
        if cls_name in ['bicycle','car','motorcycle','bus', 'train', 'truck']:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2) # x1, y1, x2, y2 
            if cls_dets.shape[1] == 5:
                score = det[-1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(im, '{:s} {:.3f}'.format(cls_name, score), (bbox[0], bbox[1]), font, 0.3, color, 1)
                outputs.append([cls_name, score, bbox[0], bbox[1], bbox[2], bbox[3]])
    
    # shrink im
    #im = cv2.resize(im, (int(im.shape[1]*0.5), int(im.shape[0]*0.5)))

    return im, outputs


"""
import matplotlib.pyplot as plt
from random import random as rand
def show_boxes(im, dets, classes, scale = 1.0):
    plt.cla()
    plt.axis("off")
    plt.imshow(im)
    for cls_idx, cls_name in enumerate(classes):
        cls_dets = dets[cls_idx]
        for det in cls_dets:
            bbox = det[:4] * scale
            color = (rand(), rand(), rand())
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                  bbox[2] - bbox[0],
                                  bbox[3] - bbox[1], fill=False,
                                  edgecolor=color, linewidth=2.5)
            plt.gca().add_patch(rect)

            if cls_dets.shape[1] == 5:
                score = det[-1]
                plt.gca().text(bbox[0], bbox[1],
                               '{:s} {:.3f}'.format(cls_name, score),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    plt.show()
    return im
    
"""

