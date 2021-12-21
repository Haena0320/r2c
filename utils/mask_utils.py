import numpy as np
import matplotlib
from matplotlib import path
matplotlib.use("agg")


def _spaced_points(low, high, n): # 각 이미지에 교차점 리스트 생성 (가로 리스트, 세로 리스트)
    padding = (high - low) / (n * 2)
    return np.linspace(low + padding, high - padding, num=n)


# 이미지를 14, 14 로 분할했을 때, 해당 오브젝트가 존재하는 범위에 14, 14 교차점이 존재하는지(1), 존재하지 않는지(0) 체크하는 마스크
def make_mask(mask_size, box, polygons_list): # mask_size : 14, box : [ box_coordinates], polygons_list :[14,14]
    mask = np.zeros((mask_size, mask_size), dtype=bool) # (14 , 14)
    xy = np.meshgrid(_spaced_points(box[0], box[2], n=mask_size), # x1~x2, 14 등분
                     _spaced_points(box[1], box[3], n=mask_size)) # y1~y2, 14 등분
    # xy : [(14, 14), (14, 14)]
    xy_flat = np.stack(xy, 2).reshape((-1, 2)) # 좌표 리스트 [[0.5,0.5], [0.5, 1], [0.5, 1.5] ...]
    for polygon in polygons_list:
        polygon_path = path.Path(polygon)
        mask |= polygon_path.contains_points(xy_flat).reshape((mask_size, mask_size))
    return mask.astype(np.float32) #[nobj, 14, 14]




# mask = np.zeros((7,7), dtype=bool)
# xy = np.meshgrid(_spaced_points(0, 10, 10),_spaced_points(0, 10, 10))
# xy_flat = np.stack(xy, 2).reshape((-1, 2))