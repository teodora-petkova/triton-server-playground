from functools import wraps
import time

import cv2


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = (end_time - start_time)*1000
        
        print(f'[DEBUG] Function {func.__name__} Took {total_time:.4f} ms')
        
        return result
    return timeit_wrapper


def draw_boxes(image, coords, scores, class_labels):
    box_color = (51, 51, 255)
    font_color = (215, 215, 215)

    line_width = max(round(sum(image.shape) / 2 * 0.0025), 2)
    font_thickness = max(line_width - 1, 1)
    draw_image = image.copy()

    for idx, tb in enumerate(coords):
        if tb[0] >= tb[2] or tb[1] >= tb[3]:
            continue
        obj_coords = list(map(int, tb[:4]))

        # bbox
        point1, point2 = (
            int(obj_coords[0]), int(obj_coords[1])), (
            int(obj_coords[2]), int(obj_coords[3]))
        
        cv2.rectangle(
            draw_image,
            point1,
            point2,
            box_color,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )

        # confidence level
        score_str = str(int(round(scores[idx], 2) * 100))
        label = f'{class_labels[idx]} {score_str}%'
        weight, height = cv2.getTextSize(
            label, 0, fontScale=2, thickness=3)[0]  # text width, height
        outside = obj_coords[1] - height - 3 >= 0  # label fits outside box

        weight, height = cv2.getTextSize(
            label, 0, fontScale=line_width / 3, thickness=font_thickness
        )[0]  # text width, height
        outside = point1[1] - height >= 3
        point2 = (point1[0] + weight, point1[1] - height - 3 if outside
            else point1[1] + height + 3)
        text_point = (point1[0], point1[1] - 2 if outside 
                        else point1[1] + height + 2) # filled
        
        cv2.rectangle(
            draw_image, point1, point2, box_color, -1, cv2.LINE_AA)

        cv2.putText(
            draw_image,
            label,
            text_point,
            0,
            line_width / 3,
            font_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )
        
    return draw_image