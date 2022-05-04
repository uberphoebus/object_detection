# nms : cleaning up bboxes
# 1. bboxes < probability threshold
# 2. while bboxes: (for each class)
#       take out largest prob
#       remove all other boxes with IoU > threshold

import torch
import iou_module as im # implemented iou

def nms(
    bboxes,
    iou_threshold,
    threshold,
    box_format='corners',
):
    # bboxes = [[1, 0.9, x1, y1, x2, y2], [], []]
    # [class, proba, x1, y1, x2, y2]
    
    assert type(bboxes) == list
    
    # bboxes < probability threshold
    bboxes = [box for box in bboxes if box[1] > threshold]
    
    # sort to get highest prob
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] # compare class
            or im.intersection_over_union(
                torch.tensor(chosen_box[2:]), # [x1, y1, x2, y2]
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold # iou < iou threshold
        ]
        
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms