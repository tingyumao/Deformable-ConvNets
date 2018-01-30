import cPickle
import cv2
import os
import json
import numpy as np

from imdb import IMDB

# coco api
"""
from .pycocotools.coco import COCO
from .pycocotools.cocoeval import COCOeval
from .pycocotools import mask as COCOmask
from utils.mask_coco2voc import mask_coco2voc
from utils.mask_voc2coco import mask_voc2coco
from utils.tictoc import tic, toc
from bbox.bbox_transform import clip_boxes
import multiprocessing as mp
"""


class detrac(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None, mask_size=-1, binary_thresh=None):
        """
        fill basic information to initialize imdb
        :param image_set: train2014, val2014, test2015
        :param root_path: 'data', will write 'rpn_data', 'cache'
        :param data_path: 'data/coco'
        """
        super(detrac, self).__init__('detrac', image_set, root_path, data_path, result_path)
        self.root_path = root_path
        self.data_path = data_path

        # deal with class names
        cats = ["car", "bus", "van", "others"]#[cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        #self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        #self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls])
        #                                    for cls in self.classes[1:]])

        # load image file names
        self.name = "detrac"
        #self.image_set_index = self._load_image_set_index()
        #self.num_images = len(self.image_set_index)
        #print 'num_images', self.num_images
        #self.mask_size = mask_size
        #self.binary_thresh = binary_thresh

        # deal with data name
        #view_map = {'minival2014': 'val2014',
        #            'valminusminival2014': 'val2014',
        #            'test-dev2015': 'test2015'}
        #self.data_name = view_map[image_set] if image_set in view_map else image_set

    def _load_detrac_annotation(self, detrac_roi):
        # detrac_roi: image_file, width, height, ignore_region, boxes
        image_file = os.path.join(self.root_path, detrac_roi[0])
        width = detrac_roi[1]#960
        height = detrac_roi[2]#540
        ignore_region = detrac_roi[3]
        boxes = [bbox[-4:] for bbox in detrac_roi[4]]
        num_box = len(boxes)
        
        boxes = np.stack(boxes, axis=1).transpose().astype(np.uint16) # bbox: [car_id, car_type, x1, y1, x2, y2]

        boxes[:,0] = np.clip(boxes[:,0], 0, width-1)
        boxes[:,2] = np.clip(boxes[:,2], 0, width-1)

        boxes[:,1] = np.clip(boxes[:,1], 0, height-1)
        boxes[:,3] = np.clip(boxes[:,3], 0, height-1)

        assert (boxes[:, 2] >= boxes[:, 0]).all()
        assert (boxes[:, 3] >= boxes[:, 1]).all()
        #print("hhh")

        gt_classes = np.asarray([self._class_to_ind[bbox[1]] for bbox in detrac_roi[4]], dtype=np.int32)
        overlaps = np.zeros((num_box, self.num_classes), dtype=np.float32)
        for i, bbox in enumerate(detrac_roi[4]):
            overlaps[i, self._class_to_ind[bbox[1]]] = 1.0

        roi_rec = {'image': image_file,
                   'height': height,
                   'width': width,
                   'boxes': boxes,
                   'gt_classes': gt_classes,
                   'gt_overlaps': overlaps,
                   'max_classes': overlaps.argmax(axis=1),
                   'max_overlaps': overlaps.max(axis=1),
                   'flipped': False}

        return roi_rec

    def gt_roidb(self):
        pkl_file = os.path.join(self.root_path, 'detect_gt.pkl')
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} original gt roidb loaded from {}'.format(self.name, pkl_file)
            #return roidb

        gt_roidb = [self._load_detrac_annotation(roi) for roi in roidb]
        self.num_images = len(gt_roidb)
        print("number of images: {}".format(self.num_images))

        #gt_roidb = [self._load_coco_annotation(index) for index in self.image_set_index]
        #with open(cache_file, 'wb') as fid:
        #    cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        #print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb
