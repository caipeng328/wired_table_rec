import os
from tqdm import tqdm
import cv2
import numpy as np
import shlex
import time
import os.path as osp
import sys
import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

from .lib.opts import opts
from .lib.detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']
cmd_str = '--task ctdet_sim --dataset table --demo ../input_images/test --demo_name demo_wired --debug 1 --arch resfpnhalf_34 --K 500 --MK 1000 --tsfm_layers 3 --stacking_layers 3 --gpus -1 --wiz_detect --wiz_4ps --wiz_stacking --wiz_pairloss --convert_onnx 0 --vis_thresh_corner 0.3 --vis_thresh 0.20 --scores_thresh 0.2 --nms --demo_dir ../visualization_wired_test/'
args_list = shlex.split(cmd_str)


class TableLineRecognitionV3:
    def __init__(self, model_path):
      self.opt = opts(model_path).init(args_list)
      self.opt.debug = max(self.opt.debug, 1)
      Detector = detector_factory[self.opt.task]
      self.detector = Detector(self.opt)

    def __call__(self, image):
      ret = self.detector.infer(self.opt, image)['4ps']
      bboxes = ret[1]
      res = []
      for bbox in bboxes:
        if bbox[8] > self.opt.vis_thresh: 
          res.append(bbox[:8])
      return res

def shrink_quadrilateral(points):  
    points = points.reshape(4,2)
    center = points.mean(axis=0)  

    shrunk_points = np.zeros_like(points)  

    for i in range(points.shape[0]):  
        vector_to_center = points[i] - center  
        shrunk_vector =1 * vector_to_center 
        shrunk_points[i] = center + shrunk_vector  

    return shrunk_points.reshape(-1)

def vis(image, bboxes): 
  for bbox in bboxes:
    bbox = np.array(bbox, dtype=np.int32)
    bbox = shrink_quadrilateral(bbox)
    cv2.line(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
    cv2.line(image,(bbox[2],bbox[3]),(bbox[4],bbox[5]),(0,0,255),2)
    cv2.line(image,(bbox[4],bbox[5]),(bbox[6],bbox[7]),(0,0,255),2)
    cv2.line(image,(bbox[6],bbox[7]),(bbox[0],bbox[1]),(0,0,255),2) 

if __name__ == '__main__':
  model, opt = Init_model()
  image_names = []
  image_root_path = 'input_images'
  save_root_path = 'save_image'
  image_ext = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff']
  ls = os.listdir(image_root_path)
  for file_name in sorted(ls):
      ext = file_name[file_name.rfind('.') + 1:].lower()
      if ext in image_ext:
          image_names.append(os.path.join(image_root_path, file_name))

  for i in tqdm(range(len(image_names))):
      image_name = image_names[i]
      image = cv2.imread(image_name)
      t = time.time()
      ret = inference(opt, model, image)
      print(f'{image_name}:{time.time() - t}')
      vis(image, ret)
      cv2.imwrite(f'{save_root_path}/{i}.jpg', image)


