import time
import cv2
import torch
import torchvision
import numpy as np


def letterbox(
        im, 
        new_shape=(640, 640), 
        color=(114, 114, 114), 
        auto=True, # for pytorch models auto = True
        scale_fill=False,
        scale_up=True, 
        stride=32):
    """
    Resize and pad the input image to make it compatible with the network's 
    input size while preserving the original aspect ratio 
    of the image. It is important because object detection models, like YOLO, 
    typically require images of a fixed size as input.
    """
    # Input Image Aspect Ratio: The method first calculates the aspect ratio
    # (width/height) of the original input image.
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    #Resize or Pad: Depending on the aspect ratio, the method either 
    # resizes the image to fit the network's input size while maintaining
    # the aspect ratio or pads the image to reach the desired size.
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # wh padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        # width, height ratios
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Add border: When padding is applied, a common practice is to fill the 
    # extra space with a predefined color, often black or gray, to distinguish 
    # it from the actual content of the image.
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  
    return im, ratio, (dw, dh)


def xywh2xyxy(xywh):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] 
    # where xy1=top-left, xy2=bottom-right
    xyxy = xywh.clone() if isinstance(xywh, torch.Tensor) else np.copy(xywh)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2  # top left x
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2  # top left y
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2  # bottom right x
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2  # bottom right y
    return xyxy


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = box1.unsqueeze(1).chunk(2, 2)
    (b1, b2) = box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0):
        """
        Non-Maximum Suppression (NMS) on inference results 
        to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image 
            [x1, y1, x2, y2, conf, cls]
        """
        prediction = torch.from_numpy(np.array(prediction[None]))

        # Checks
        assert 0 <= conf_thres <= 1, \
            f'Invalid Confidence threshold {conf_thres}, \
            valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, \
            f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
         # YOLOv5 model in validation model, output = (inference_out, loss_out)
        if isinstance(prediction, (list, tuple)): 
            prediction = prediction[0]  # select only inference output

        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints - width-height
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4]=0  
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            # center_x, center_y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])  
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None],
                    j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) >
                                                                conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == 
                       torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
             # sort by confidence and remove excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]] 

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + c, x[:, 4]  
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections

            # Merge NMS (boxes merged using weighted mean)
            if merge and (1 < n < 3E3):  
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                # merged boxes
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / \
                    weights.sum(1, keepdim=True)  
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            # if mps:
            #     output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output
