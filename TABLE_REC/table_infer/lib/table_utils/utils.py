from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_umich_gaussian_wh, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.adjacency import adjacency

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        if self.val != 0:
            self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

