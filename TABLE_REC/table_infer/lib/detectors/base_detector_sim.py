from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import onnxruntime as ort
import time
from table_utils.image import get_affine_transform, get_affine_transform_upper_left

class BaseDetector_sim(object):
  def __init__(self, opt, use_onnx = True):
    self.use_onnx = use_onnx
    if use_onnx:
      self.model = ort.InferenceSession(opt.load_model, providers=['CPUExecutionProvider'])
    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = opt.K
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) #+ 1
      inp_width = (new_width | self.opt.pad) #+ 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)
    if self.opt.upper_left:
      c = np.array([0, 0], dtype=np.float32)
      s = max(height, width) * 1.0
      trans_input = get_affine_transform_upper_left(c, s, 0, [inp_width, inp_height])
    else:
      trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    meta = {'c': c, 's': s,
            'input_height': inp_height,
            'input_width' : inp_width,
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def resize(self,image):
    h,w,_ = image.shape
    scale = 1024/(max(w,h)+1e-4)
    image = cv2.resize(image,(int(w*scale),int(h*scale)))
    image = cv2.copyMakeBorder(image,0,1024 - int(h*scale), 0, 1024 - int(w*scale),cv2.BORDER_CONSTANT, value=[0,0,0])
    return image,scale

  def infer(self, opt, image, meta=None):

    images, meta = self.pre_process(image, 1.0, meta)

    outputs, output, dets, corner_st_reg, forward_time, logi, cr, keep = self.process(images, image, use_onnx = self.use_onnx)
    dets, corner_st_reg = self.post_process(dets, meta, corner_st_reg, 1.0)
    detections = [dets]
    results = self.merge_outputs(detections)
    return {'4ps':results}
