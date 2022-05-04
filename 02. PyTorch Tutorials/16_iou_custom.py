import torch
import cv2
import numpy as np

# IoU = intersection / union (IoU > 0.7)

class IoU:
    def __init__(self, path, boxes_preds, boxes_labels, box_format='midpoint'):
        self.path = path
        self.boxes_preds = boxes_preds
        self.boxes_labels = boxes_labels
        self.box_format = box_format
        self.box1_x1 = None
        self.box1_y1 = None
        self.box1_x2 = None
        self.box1_y2 = None
        self.box2_x1 = None
        self.box2_y1 = None
        self.box2_x2 = None
        self.box2_y2 = None
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
    
    def intersection_over_union(self):
        """
        Calculates intersection over union
        Parameters:
            boxes_preds (tensor): Predictoins of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x, y, w, h) or (x1, y1, x2, y2)
        
        Returns:
            tensor: Coords of intersection
        """
        
        # boxes_preds shape is (N, 4) where N is the num of bboxes
        # boxes_labels shape is (N, 4) where N is the num of bboxes
        if self.box_format == 'midpoint':
            self.box1_x1 = self.boxes_preds[..., 0:1] - self.boxes_preds[..., 2:3] / 2
            self.box1_y1 = self.boxes_preds[..., 1:2] - self.boxes_preds[..., 3:4] / 2
            self.box1_x2 = self.boxes_preds[..., 0:1] - self.boxes_preds[..., 2:3] / 2
            self.box1_y2 = self.boxes_preds[..., 1:2] - self.boxes_preds[..., 3:4] / 2
            self.box2_x1 = self.boxes_labels[..., 0:1] - self.boxes_labels[..., 2:3] / 2
            self.box2_y1 = self.boxes_labels[..., 1:2] - self.boxes_labels[..., 3:4] / 2
            self.box2_x2 = self.boxes_labels[..., 0:1] - self.boxes_labels[..., 2:3] / 2
            self.box2_y2 = self.boxes_labels[..., 1:2] - self.boxes_labels[..., 3:4] / 2
        
        if self.box_format == 'corners':
            # slice to keep tensor shape (N, 1)
            self.box1_x1 = self.boxes_preds[..., 0:1]
            self.box1_y1 = self.boxes_preds[..., 1:2]
            self.box1_x2 = self.boxes_preds[..., 2:3]
            self.box1_y2 = self.boxes_preds[..., 3:4]
            self.box2_x1 = self.boxes_labels[..., 0:1]
            self.box2_y1 = self.boxes_labels[..., 1:2]
            self.box2_x2 = self.boxes_labels[..., 2:3]
            self.box2_y2 = self.boxes_labels[..., 3:4]
        
        # corner points of the intersection
        self.x1 = torch.max(self.box1_x1, self.box2_x1)
        self.y1 = torch.max(self.box1_y1, self.box2_y1)
        self.x2 = torch.max(self.box1_x2, self.box2_x2)
        self.y2 = torch.max(self.box1_y2, self.box2_y2)
        
        # # intersection = W x H
        # # .clamp(0) is for when they do not intersect (torch tensor)
        self.intersection = (self.x2 - self.x1).clamp(0) * (self.y2 - self.y1).clamp(0)
        
        # area
        box1_area = abs((self.box1_x2 - self.box1_x1) * (self.box1_y2 - self.box1_y1))
        box2_area = abs((self.box2_x2 - self.box2_x1) * (self.box2_y2 - self.box2_y1))
        
        return self.intersection / (box1_area + box2_area - self.intersection + 1e-6)
    
    def draw_iou(self):
        
        DOWNSIZE_RATE = 0.3
        COLOR_PRED = (0, 255, 0)
        COLOR_LABEL = (0, 0, 255)
        COLOR_IOU = (0, 234, 255)
        THICKNESS = 3
        ALPHA = 0.2
        
        # call image
        img = cv2.imread(self.path)
        img = cv2.resize(img, None, fx=DOWNSIZE_RATE, fy=DOWNSIZE_RATE)
        shapes = np.zeros_like(img, np.uint8)
        
        # calculate coords
        self.intersection_over_union()
        
        # draw shapes
        cv2.rectangle(img, 
                      (100, 50), 
                      (500, 550), 
                      COLOR_PRED, THICKNESS)
        cv2.rectangle(img, 
                      (150, 100), 
                      (550, 600), 
                      COLOR_LABEL, THICKNESS)
        cv2.rectangle(shapes, 
                      (int(self.x1), int(self.x2)), 
                      (int(self.y1), 550), 
                      COLOR_IOU, cv2.FILLED)
        
        mask_img = img.copy()
        mask = shapes.astype(bool)
        mask_img[mask] = cv2.addWeighted(img, ALPHA, shapes, 1 - ALPHA, 0)[mask]
        
        cv2.imshow('IoU', mask_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    
    # 좌표값 문제 없음
    boxes_preds = torch.Tensor([300, 300, 400, 500])
    boxes_labels = torch.Tensor([350, 350, 400, 500])
    
    iou = IoU(path='./cat.jpg', boxes_preds=boxes_preds, boxes_labels=boxes_labels, box_format='midpoint')
    iou.draw_iou()