from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import numpy as np


def _h_dist_feat(output, width):
  feat =  (output[:,:,0] + output[:,:,1])/(2*(width+1))
  return feat

def _v_dist_feat(output, height):
  feat =  (output[:,:,2] + output[:,:,3])/(2*(height + 1))
  return feat


def _gather_feat(feat, ind, mask=None):

    dim = feat.shape[2]
    ind = np.expand_dims(ind, axis=2) 
    ind = np.tile(ind, (1, 1, dim)) 
    feat = np.take_along_axis(feat, ind, axis=1) 
    if mask is not None:
        mask = np.expand_dims(mask, axis=2)
        mask = np.tile(mask, (1, 1, dim))

        feat = feat[mask].reshape(-1, dim)

    return feat


def _flatten_and_gather_feat(output, ind):
  dim = output.size(3)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  output = output.contiguous().view(output.size(0), -1, output.size(3))
  output1 = output.gather(1, ind)

  return output1

def _tranpose_and_gather_feat(feat, ind):

    feat = np.transpose(feat, (0, 2, 3, 1)) 
    batch, height, width, channels = feat.shape
    feat = feat.reshape(batch, height * width, channels)
    feat = _gather_feat(feat, ind)
    return feat





