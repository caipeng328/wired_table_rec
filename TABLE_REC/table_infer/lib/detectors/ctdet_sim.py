from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
from external.shapelyNMS import pnms
from models.decode import corner_decode,ctdet_4ps_decode
from table_utils.image import get_affine_transform
from table_utils.post_process import ctdet_post_process,ctdet_4ps_post_process,ctdet_4ps_post_process_upper_left
from table_utils.post_process import ctdet_corner_post_process
from .base_detector_sim import BaseDetector_sim
from PIL import Image
from matplotlib import cm


class CtdetDetector_sim(BaseDetector_sim):
  def __init__(self, opt):
    super(CtdetDetector_sim, self).__init__(opt)
    if self.use_onnx:
      self.input_name = self.model.get_inputs()[0].name

  def process(self, images, origin, use_onnx = True):

    if not use_onnx:
      t = time.time()
      print('not use onnx')
      outputs = self.model(images)
      output = outputs[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if self.opt.reg_offset else None
      st = output['st']
      ax = output['ax']
      cr = output['cr']
      print(f'model_inference--->>>{time.time() - t}')
    else:
      t = time.time()
      output = self.model.run(None, {self.input_name: images})
      hm = 1 / (1 + np.exp(-np.array(output[0])))
      st = np.array(output[1])
      wh = np.array(output[2])
      ax = np.array(output[3])
      cr = np.array(output[4])
      reg = np.array(output[5]) if self.opt.reg_offset else None
      print(f'model_inference--->>>{time.time() - t}')
      forward_time = time.time()

    scores, inds, ys, xs, st_reg, corner_dict = corner_decode(hm[:,1:2,:,:], st, reg, K=int(self.opt.MK))
    dets, keep, logi, cr = ctdet_4ps_decode(hm[:,0:1,:,:], wh, ax, cr, corner_dict, reg=reg, K=self.opt.K, wiz_rev = True)

    corner_output = np.concatenate((np.transpose(xs),np.transpose(ys),np.array(st_reg),np.transpose(scores)), axis=2)
    
    return output, output, dets, corner_output, forward_time, logi, cr, keep #, overlayed_map

  def post_process(self, dets, meta, corner_st, scale=1):
    # dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    #return dets is list and what in dets is dict. key of dict is classes, value of dict is [bbox,score]
    if self.opt.upper_left:
      dets = ctdet_4ps_post_process_upper_left(
          dets.copy(), [meta['c']], [meta['s']],
          meta['out_height'], meta['out_width'], self.opt.num_classes)
    else:
      dets = ctdet_4ps_post_process(
          dets.copy(), [meta['c']], [meta['s']],
          meta['out_height'], meta['out_width'], self.opt.num_classes)
    corner_st = ctdet_corner_post_process(
        corner_st.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 9)
      dets[0][j][:, :8] /= scale
    return dets[0],corner_st[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)

      if len(self.scales) > 1 or self.opt.nms:
         #soft_nms(results[j], Nt=0.5, method=2)
         results[j] = pnms(results[j],self.opt.thresh_min,self.opt.thresh_conf)
    scores = np.hstack(
      [results[j][:, 8] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 8] >= thresh)
        results[j] = results[j][keep_inds]
    return results
