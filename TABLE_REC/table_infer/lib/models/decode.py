from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .utils import _gather_feat, _tranpose_and_gather_feat
import numpy as np 
import shapely
import time
from shapely.geometry import Polygon, Point


import numpy as np
from scipy.ndimage import maximum_filter


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = maximum_filter(heat, size=kernel, mode='constant')
    keep = (hmax == heat).astype(np.float32)
    return heat * keep, keep

# def _nms(heat, kernel=3):
#     pad = (kernel - 1) // 2

#     hmax = nn.functional.max_pool2d(
#         heat, (kernel, kernel), stride=1, padding=pad)
#     keep = (hmax == heat).float()
#     return heat * keep,keep




import numpy as np

def _topk(scores, K=40):

    batch, cat, height, width = scores.shape
    scores_flat = scores.reshape(batch, cat, -1)
    topk_inds_cat = np.argsort(-scores_flat, axis=2)[:, :, :K]
    topk_scores = np.take_along_axis(scores_flat, topk_inds_cat, axis=2) 
    topk_inds = topk_inds_cat % (height * width)
    topk_ys = (topk_inds // width).astype(np.float32)  
    topk_xs = (topk_inds % width).astype(np.float32) 
    topk_scores_flat = topk_scores.reshape(batch, -1)  
    topk_inds_flat = topk_inds_cat.reshape(batch, -1)  
    topk_ys_flat = topk_ys.reshape(batch, -1, 1) 
    topk_xs_flat = topk_xs.reshape(batch, -1, 1)  
    topk_inds_global = np.argsort(-topk_scores_flat, axis=1)[:, :K] 
    topk_scores = _gather_feat(topk_scores_flat.reshape(batch, -1, 1), topk_inds_global).squeeze(2)
    topk_clses = (topk_inds_global // K).astype(np.int32)
    topk_inds = _gather_feat(topk_inds_flat.reshape(batch, -1, 1), topk_inds_global).squeeze(2)
    topk_ys = _gather_feat(topk_ys_flat, topk_inds_global).squeeze(2)
    topk_xs = _gather_feat(topk_xs_flat, topk_inds_global).squeeze(2)

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs




def corner_decode(mk, st_reg, mk_reg=None, K=400):

  batch, cat, height, width = mk.shape
  mk,keep = _nms(mk)
  scores, inds, clses, ys, xs = _topk(mk, K=K)

  if mk_reg is not None:
    reg = _tranpose_and_gather_feat(mk_reg, inds)
    reg = reg.reshape(batch, K, 2)
    xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
  else:
    xs = xs.reshape(batch, K, 1) + 0.5
    ys = ys.reshape(batch, K, 1) + 0.5
  scores = scores.reshape(batch, K, 1)
  st_Reg = _tranpose_and_gather_feat(st_reg, inds)
  bboxes = np.concatenate([xs - st_Reg[..., 0:1], 
                          ys - st_Reg[..., 1:2],
                          xs - st_Reg[..., 2:3], 
                          ys - st_Reg[..., 3:4],
                          xs - st_Reg[..., 4:5],
                          ys - st_Reg[..., 5:6],
                          xs - st_Reg[..., 6:7],
                          ys - st_Reg[..., 7:8]], axis=2)
  corner_dict = {'scores': scores, 'inds': inds, 'ys': ys, 'xs': xs, 'gboxes': bboxes}
  return scores, inds, ys, xs, bboxes, corner_dict

def ctdet_4ps_decode(heat, wh, ax, cr, corner_dict=None, reg=None, cat_spec_wh=False, K=100, wiz_rev = False):

    batch, cat, height, width = heat.shape
    heat,keep = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    if reg is not None:
      reg = _tranpose_and_gather_feat(reg, inds)
      reg = reg.reshape(batch, K, 2)
      xs = xs.reshape(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.reshape(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.reshape(batch, K, 1) + 0.5
      ys = ys.reshape(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    # assert False, 'now'
    
    if cat_spec_wh:
      wh = wh.reshape(batch, K, cat, 8)
      clses_ind = clses.reshape(batch, K, 1, 1)
      clses_ind = np.tile(clses_ind, (1, 1, 1, 8)).astype(int)  # Expand and convert to int
      wh = np.take_along_axis(wh, clses_ind, axis=2).reshape(batch, K, 8)
    else:
      wh = wh.reshape(batch, K, 8)

    clses = clses.reshape(batch, K, 1).astype(float)
    scores = scores.reshape(batch, K, 1)



    bboxes = np.concatenate([xs - wh[..., 0:1], 
                            ys - wh[..., 1:2],
                            xs - wh[..., 2:3], 
                            ys - wh[..., 3:4],
                            xs - wh[..., 4:5],
                            ys - wh[..., 5:6],
                            xs - wh[..., 6:7],
                            ys - wh[..., 7:8]], axis=2)
    # assert False, 'now'
    rev_time_s1 = time.time()
    if wiz_rev :
      bboxes_rev = np.copy(bboxes)
      bboxes_cpu = np.copy(bboxes)  

      gboxes = corner_dict['gboxes']
      gboxes_cpu = np.copy(gboxes) 

      num_bboxes = bboxes.shape[1]
      num_gboxes = gboxes.shape[1]

      corner_xs = corner_dict['xs'] 
      corner_ys = corner_dict['ys'] 
      corner_scores = corner_dict['scores']
      
      for i in range(num_bboxes):
        if scores[0, i, 0] >= 0.2:
            count = 0
            for j in range(num_gboxes):
                if corner_scores[0, j, 0] >= 0.3:

                    bbox = bboxes_cpu[0, i, :]
                    gbox = gboxes_cpu[0, j, :]
                    if is_group_faster_faster(bbox, gbox): 
                        cr_x = corner_xs[0, j, 0]
                        cr_y = corner_ys[0, j, 0]

                        ind4ps = find4ps(bbox, cr_x, cr_y)
                        if np.allclose(bboxes_rev[0, i, 2*ind4ps], bboxes[0, i, 2*ind4ps]) and np.allclose(bboxes_rev[0, i, 2*ind4ps+1], bboxes[0, i, 2*ind4ps+1]):
                            count += 1
                            bboxes_rev[0, i, 2*ind4ps] = cr_x
                            bboxes_rev[0, i, 2*ind4ps + 1] = cr_y
                        else:
                            origin_x = bboxes[0, i, 2*ind4ps]
                            origin_y = bboxes[0, i, 2*ind4ps+1]

                            old_x = bboxes_rev[0, i, 2*ind4ps]
                            old_y = bboxes_rev[0, i, 2*ind4ps+1]
                            if dist(origin_x, origin_y, old_x, old_y) >= dist(origin_x, origin_y, cr_x, cr_y):
                                count += 1
                                bboxes_rev[0, i, 2*ind4ps] = cr_x
                                bboxes_rev[0, i, 2*ind4ps + 1] = cr_y
                            else:
                                continue
                    else:
                        continue
                else:
                    break        
            if count <= 2:
                scores[0, i, 0] *= 0.4
        else:
            break

    # assert False, 'now'

    if wiz_rev:
      cc_match = np.concatenate([
          (bboxes_rev[:, :, 0:1] + width * np.round(bboxes_rev[:, :, 1:2])),
          (bboxes_rev[:, :, 2:3] + width * np.round(bboxes_rev[:, :, 3:4])),
          (bboxes_rev[:, :, 4:5] + width * np.round(bboxes_rev[:, :, 5:6])),
          (bboxes_rev[:, :, 6:7] + width * np.round(bboxes_rev[:, :, 7:8]))
      ], axis=2)
    else:    
      cc_match = np.concatenate([
          (xs - wh[..., 0:1]) + width * np.round(ys - wh[..., 1:2]),
          (xs - wh[..., 2:3]) + width * np.round(ys - wh[..., 3:4]),
          (xs - wh[..., 4:5]) + width * np.round(ys - wh[..., 5:6]),
          (xs - wh[..., 6:7]) + width * np.round(ys - wh[..., 7:8])
      ], axis=2)

    cc_match = np.round(cc_match).astype(np.int64)

    if wiz_rev:
        detections = np.concatenate([bboxes_rev, scores, clses], axis=2)
        sorted_ind = np.argsort(scores, axis=1)[:, :, ::-1]  # Sorting in descending order
        sorted_inds = sorted_ind.repeat(detections.shape[2], axis=-1)
        # sorted_inds = np.expand_dims(sorted_ind, axis=-1).repeat(detections.shape[2], axis=-1)

        print(detections.shape, sorted_inds.shape)

        detections = np.take_along_axis(detections, sorted_inds, axis=1)
    else:
        detections = np.concatenate([bboxes, scores, clses], axis=2)


    return detections, keep, None, None




def find4ps(bbox, x, y):
    xs = np.array([bbox[0], bbox[2], bbox[4], bbox[6]])
    ys = np.array([bbox[1], bbox[3], bbox[5], bbox[7]])

    dx = xs - x
    dy = ys - y

    l = dx**2 + dy**2
    return np.argmin(l)  # Use numpy's argmin



def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    l = dx**2 + dy**2
    return l


def is_group_faster_faster(bbox, gbox):
    bbox = bbox.reshape(4, 2)
    gbox = gbox.reshape(4, 2)
  
    # Calculate the bounding box min and max values using numpy
    bbox_xmin, bbox_xmax = bbox[:, 0].min(), bbox[:, 0].max()
    bbox_ymin, bbox_ymax = bbox[:, 1].min(), bbox[:, 1].max()
    gbox_xmin, gbox_xmax = gbox[:, 0].min(), gbox[:, 0].max()
    gbox_ymin, gbox_ymax = gbox[:, 1].min(), gbox[:, 1].max()

    # Check for overlap between bbox and gbox
    if bbox_xmin > gbox_xmax or gbox_xmin > bbox_xmax or bbox_ymin > gbox_ymax or gbox_ymin > bbox_ymax:
        return False
    else:
        # Create a polygon for the bbox
        bpoly = Polygon(bbox)

        flag = 0
        for i in range(4):
            p = Point(gbox[i])
            if p.within(bpoly):
                flag = 1
                break
        if flag == 0:
            return False
        else:
            return True

# def is_group_faster_faster(bbox, gbox):
#     bbox = bbox.view(4,2)
#     gbox = gbox.view(4,2)
  
#     bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax = bbox[:,0].min(), bbox[:,0].max(), bbox[:,1].min(), bbox[:,1].max()#min(bbox_xs), max(bbox_xs), min(bbox_ys), max(bbox_ys)
#     gbox_xmin, gbox_xmax, gbox_ymin, gbox_ymax = gbox[:,0].min(), gbox[:,0].max(), gbox[:,1].min(), gbox[:,1].max()

#     if bbox_xmin > gbox_xmax or gbox_xmin > bbox_xmax or bbox_ymin > gbox_ymax or gbox_ymin > bbox_ymax:
#         return False
#     else:
#         bpoly = Polygon(bbox)

#         flag = 0
#         for i in range(4):
#             p = Point(gbox[i])
#             if p.within(bpoly):
#                 flag = 1
#                 break
#         if flag == 0:
#             return False
#         else :
#             return True


