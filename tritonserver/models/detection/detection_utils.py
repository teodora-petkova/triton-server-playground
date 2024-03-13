import numpy as np
import yaml

from tritonserver.models.detection.utils import (
    letterbox,
    non_max_suppression,
    scale_boxes,
)

class DetectionUtils:

    @staticmethod
    def get_labels(filepath: str):
        labels = []
        with open(filepath, 'r', encoding='utf-8') as file_pointer:
            data_yaml = yaml.safe_load(file_pointer)
        labels = data_yaml['names']
        return labels
    
    
    @staticmethod
    def preprocess(
        img: np.array,
        input_width=640,
        input_height=640,
        np_dtype=np.float32,
        stride=32):

        img = letterbox(
            img, max(input_width, input_height), stride=stride, auto=False)[0]
        img = img.transpose((2, 0, 1))#[::-1]  # HWC to CHW
        # note: expected RGB, no BGR to RGB    
        img = np.ascontiguousarray(img)
        img = img.astype(np_dtype)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.reshape([1, *img.shape])
        return img
    
    
    @staticmethod
    def _get_result_as_json_dict(bboxes, scores, labels):
        detections = []
       
        for bbox, score, label in zip(bboxes, scores, labels):

            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            detection = {
                "confidence": score * 100,
                "coordinates": {
                    "height": int(abs(x2 - x1)),
                    "width": abs(y2 - y1),
                    "xmax": max(x1, x2),
                    "xmin": min(x1, x2),
                    "ymax": max(y1, y2),
                    "ymin": min(y1, y2)
                },
                'name': {
                    'en': label
                }
            }

            detections.append(detection)#, dtype=np.object_))

        return {
            'result':{
                'detections': detections
            },
            "status": {
                "text": "",
                "type": "success"
            }
        }
    

    @staticmethod
    def postprocess(
        raw_output, 
        origin_image_hw,
        model_input_hw=(640, 640),
        conf_thresh=0.25,
        iou_thresh=0.45,
        labels=[]):

        boxes_ = non_max_suppression(
            raw_output,
            conf_thres = conf_thresh,
            iou_thres = iou_thresh
        )
        
        boxes = boxes_[0].numpy() 

        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classids = boxes[:, 5] if len(boxes) else np.array([])
        
        result_class_labels = [labels[int(id)] for id in result_classids]

        if len(result_boxes) > 0:
            # rescale boxes to original image size from processing size 
            # e.g. (640x640 -> 1920x1080)
            result_boxes = scale_boxes(
                (model_input_hw[0], model_input_hw[1]), result_boxes, 
                (origin_image_hw[0], origin_image_hw[1])
            )

        return result_boxes, result_scores, result_class_labels
    

    @staticmethod
    def postprocess_json(
        raw_output, 
        origin_image_hw,
        modelmodel,
        names = []):
                
        result_boxes, result_scores, result_class_labels = DetectionUtils.postprocess(
            raw_output,
            origin_image_hw,
            names,
            model_input_hw=(640, 640))

        res = DetectionUtils._get_result_as_json_dict(
            result_boxes, result_scores, result_class_labels)
    
        return res