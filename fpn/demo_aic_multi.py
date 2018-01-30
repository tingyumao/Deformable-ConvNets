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
import skvideo.io
import cPickle as pickle

from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fpn/cfgs/fpn_demo.yaml')

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
    config.symbol = "resnet_v1_101_fpn_dcn_rcnn"  if not args.rfcn_only else "resnet_v1_101_fpn_rcnn"
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 81
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    # test
    # find all videos
    video_path = "../../aic2018/track1/track1_videos"
    video_files = sorted([ x for x in os.listdir(video_path) if x.endswith(".mp4") and x.startswith("Loc1_4")])
    print(video_files)
    save_path = "../../aic2018/track1/output"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    print("processing {} videos...".format(len(video_files)))
    pbar = tqdm(total=len(video_files))
    for vf in video_files:
        vid = imageio.get_reader(os.path.join(video_path, vf),'ffmpeg')
        data = []
        for idx, im in enumerate(vid):
            if idx == 0:
                #assert os.path.exists(im_path + im_name), ('%s does not exist'.format(im_path + im_name))
                #im = cv2.imread(im_path + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                target_size = config.SCALES[0][0]
                max_size = config.SCALES[0][1]
                im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
                im_tensor = transform(im, config.network.PIXEL_MEANS)
                im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
                data.append({'data': im_tensor, 'im_info': im_info})
            else:
                break
                #data.append({'data': None, 'im_info': None})
        
        # get predictor
        data_names = ['data', 'im_info']
        label_names = []
        data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
        provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
        provide_label = [None for i in xrange(len(data))]

        print("hhhhh")
        print(provide_data, provide_label)
        print("hhhhh")  

        arg_params, aux_params = load_param(cur_path + '/../model/demo_model/' + ('fpn_dcn_coco' if not args.rfcn_only else 'fpn_coco'), 0, process=True)

        #print(type(arg_params), type(aux_params))

        predictor = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        nms = gpu_nms_wrapper(config.TEST.NMS, 0)

        print("successfully load model")
        
        vout = []
        # write to video
        writer = skvideo.io.FFmpegWriter(os.path.join(save_path, vf.replace(".mp4","_multi.mp4")), outputdict={'-vcodec': 'libx264', '-b': '300000000'})
        
        # define multi-scale detection
        multi_scales = [[0,0,1920,1080], [500, 200, 400, 200], [900, 200, 400, 200]]
        
        for frame_idx, im in enumerate(vid):
            #im = cv2.imread(im_path + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            im_original = im.copy()
            
            def multi_detect_input(im):
                
                #print(im.shape)
                
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
                return data_batch, scales

            tic()
            dets_nms = []
            num_dets = 0
            for scale_idx, each_scale in enumerate(multi_scales):
                offx, offy, w, h = each_scale
                data_batch, scales = multi_detect_input(im_original[offy:offy+h, offx:offx+w])
                #tmp_img = im_original[offy:offy+h, offx:offx+w]
                #cv2.imwrite(os.path.join(save_path, "{}_{}_{}.jpg".format(vf.replace(".mp4", ""), str(frame_idx).zfill(5), str(scale_idx))), tmp_img)
            
                scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
                boxes = boxes[0].astype('f')
                scores = scores[0].astype('f')
                each_scale_dets = 0
                for j in range(1, scores.shape[1]):
                    cls_scores = scores[:, j, np.newaxis]
                    cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
                    cls_dets = np.hstack((cls_boxes, cls_scores))
                    keep = nms(cls_dets)
                    cls_dets = cls_dets[keep, :]
                    cls_dets = cls_dets[cls_dets[:, -1] > 0.65, :] #num_dets: x1, y1, x2, y2, score
                    cls_dets[:, 0] += offx
                    cls_dets[:, 1] += offy
                    cls_dets[:, 2] += offx
                    cls_dets[:, 3] += offy
                    if scale_idx == 0:
                        dets_nms.append(cls_dets)
                    else:
                        tmp = dets_nms[j-1]
                        dets_nms[j-1] = np.concatenate((tmp, cls_dets), axis=0)
                    num_dets += cls_dets.shape[0]
                    each_scale_dets += cls_dets.shape[0]
                print("num dets at {} scale: {}".format(scale_idx, each_scale_dets))
            
            print 'testing {} the {} th frame at {:.4f}s, detections {}'.format(vf, frame_idx, toc(), num_dets)
            # save results
            #im = cv2.imread(im_path + im_name)
            #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            #im_bbox = show_boxes(im, dets_nms, classes, 1)
            #cv2.imwrite(im_path + im_name.replace(".jpg", "_bbox.jpg"), im_bbox)
            save_im, outputs = show_boxes(im_original, dets_nms, classes, 1)
            #cv2.imwrite(os.path.join(save_path, "{}_{}.jpg".format(vf.replace(".mp4", ""), str(frame_idx).zfill(5))), save_im)
            writer.writeFrame(save_im)
            
            for out in outputs:
                vout.append([frame_idx] + out)
        
        # save the whole video detection into pickle file
        writer.close()
        with open(os.path.join(save_path, vf.replace(".mp4", "_multi.pkl")), "wb") as f:
            pickle.dump(vout, f, protocol=2)
        pbar.update(1)
        
    pbar.close()    
    print 'done'

if __name__ == '__main__':
    main()
