# object localization = what(cls) + where(bbox)
# object detection = what(cls) + where(bbox) multiple

# output1 : [*classes, x1, y1, x2, y2]
# output2 : [*classes, x, y, W, H]

# sliding window
# move by slide -> reach ob bbox
# a lot of computation
# many bbox for same ob -> NMS

# RCNN, Fast RCNN, Faster RCNN, Yolo