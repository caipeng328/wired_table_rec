from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .ctdet import CtdetDetector
from .ctdet_sim import CtdetDetector_sim
detector_factory = {
  'ctdet': CtdetDetector_sim,
  'ctdet_mid': CtdetDetector_sim,
  'ctdet_small': CtdetDetector_sim,
  'ctdet_sim' : CtdetDetector_sim
}
