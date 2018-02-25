# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------

from tqdm import *

import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2

import imageio
import cPickle as pickle

from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fpn/cfgs/resnet_v1_101_detrac_trainval_fpn_dcn_end2end_ohem.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--rfcn_only', help='whether use R-FCN only (w/o Deformable ConvNets)', default=False, action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()

def main():
    # get symbol
    pprint.pprint(config)
    #config.symbol = "resnet_v1_101_fpn_dcn_rcnn"  if not args.rfcn_only else "resnet_v1_101_fpn_rcnn"
    config.symbol = "resnet_v1_101_fpn_dcn_rcnn"
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 5
    classes = ["car", "bus", "van", "others"]

    # load demo videos
    im_path = '../../aic2018/track1/images/'
    image_names = [x for x in os.listdir('../../aic2018/track1/images/') if (x.endswith(".jpg") and (x.startswith("9_1") or x.startswith("9_1"))) and not x.endswith("_bbox.jpg")]
    data = []
    for idx, im_name in enumerate(image_names[:1]):
        if idx == 0:
            assert os.path.exists(im_path + im_name), ('%s does not exist'.format(im_path + im_name))
            im = cv2.imread(im_path + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            target_size = config.SCALES[0][0]
            max_size = config.SCALES[0][1]
            im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            im_tensor = transform(im, config.network.PIXEL_MEANS)
            im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
            data.append({'data': im_tensor, 'im_info': im_info})
        else:
            data.append({'data': None, 'im_info': None})
    
    print(data)
    
    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[0][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    # what does provide_data and provide_label work for?
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[0])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    ## load parameters
    arg_params, aux_params = load_param(cur_path + '/../model/' + 'fpn_detrac', 1, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)
  
    print("successfully load model")
    
    # find all videos
    video_path = "../../tmp"
    video_files = [ x for x in os.listdir(video_path) if x.endswith(".mp4")]
    save_path = "../../tmp/output"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    print("processing {} videos...".format(len(video_files)))
    pbar = tqdm(total=len(video_files))
    for vf in video_files:
        vid = imageio.get_reader(os.path.join(video_path, vf),'ffmpeg')
        vout = []
        for frame_idx, im in enumerate(vid):
            #im = cv2.imread(im_path + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            target_size = config.SCALES[0][0]
            max_size = config.SCALES[0][1]
            im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
            im_tensor = transform(im, config.network.PIXEL_MEANS)
            im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

            data_idx = [{"data": im_tensor, "im_info": im_info}]
            data_idx = [[mx.nd.array(data_idx[i][name]) for name in data_names] for i in xrange(len(data_idx))]
            data_batch = mx.io.DataBatch(data=[data_idx[0]], label=[], pad=0, index=idx,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, data_idx[0])]],
                                         provide_label=[None])

            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

            tic()
            scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
            boxes = boxes[0].astype('f')
            scores = scores[0].astype('f')
            dets_nms = []
            for j in range(1, scores.shape[1]):
                cls_scores = scores[:, j, np.newaxis]
                cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                cls_dets = cls_dets[keep, :]
                cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
                dets_nms.append(cls_dets)
            
            print 'testing {} the {} th frame at {:.4f}s, detections {}'.format(vf, frame_idx, toc(), len(dets_nms))
            # save results
            #im = cv2.imread(im_path + im_name)
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #im_bbox = show_boxes(im, dets_nms, classes, 1)
            #cv2.imwrite(im_path + im_name.replace(".jpg", "_bbox.jpg"), im_bbox)
            save_im, outputs = show_boxes(im, dets_nms, classes, 1)
            #cv2.imwrite(os.path.join(save_path, "{}_{}.jpg".format(vf.replace(".mp4", ""), str(frame_idx).zfill(5))), save_im)
            
            for out in outputs:
                vout.append([frame_idx] + out)
        
        # save the whole video detection into pickle file
        with open(os.path.join(save_path, vf.replace(".mp4", ".pkl")), "wb") as f:
            pickle.dump(vout, f, protocol=2)
        pbar.update(1)
        
    pbar.close()    
    print 'done'

if __name__ == '__main__':
    main()
