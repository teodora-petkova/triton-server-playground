import time
import numpy as np
from pathlib import Path

import cv2
from PIL import Image, ImageTk
from tkinter import Tk, Label

from client.triton_server_utils import(
    init_triton_grcpclient,
    detection_inference
)

from client.triton_client_utils import (
    draw_boxes,
    timeit
)

from models.detection.detection_model import DetectionModel
from models.detection.detection_utils import DetectionUtils


LABELS_TXT = './deploy/model_repository/infamous_symbols/labels.txt'
MODEL_NAME = 'infamous_symbols'
VIDEO_PATH = Path('videos') / \
    'History of the Swastika Holocaust Education.mp4' 


@timeit
def update_frame(
    root, 
    cap, 
    detector, 
    video_label, 
    frame_count, 
    interval, 
    client,
    labels):

    ret, frame = cap.read()
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + interval)
    frame_count += interval
    
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        preprocessed_image = np.array([DetectionUtils.preprocess(frame_rgb)])
        
        bboxes, scores, class_labels = detection_inference(
            preprocessed_image, frame_rgb.shape,
            detector, client, labels)

        pil_frame = Image.fromarray(frame_rgb)
        if len(bboxes) > 0:
            pred_frame = draw_boxes(frame_rgb, bboxes, scores, class_labels)
            pil_frame = Image.fromarray(pred_frame)
            
        # Display the video frame in the Tkinter label
        video_image = ImageTk.PhotoImage(image=pil_frame)
        video_label.configure(image=video_image)
        video_label.image = video_image     
        
        # Schedule the next frame update
        root.after(
            1, 
            lambda : update_frame(
                root, cap, detector, video_label, frame_count, 
                interval, client, labels
                )
            )
    else:
        root.quit()


def main():
    try:
        client = init_triton_grcpclient()
        
        # Open a video capture object
        cap = cv2.VideoCapture(str(VIDEO_PATH))

        # Create a Tkinter window
        root = Tk()
        root.title('Test Inference')

        # Create a label to display the video
        video_label = Label(root)
        video_label.pack()
        
        # Create a label to display the PIL image
        pil_label = Label(root)
        pil_label.pack()
        
        labels = DetectionUtils.get_labels(LABELS_TXT)
        detector = DetectionModel(
            name=MODEL_NAME, labels=labels, conf_thresh=0.5)
        

        fps = cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second    
        interval = int(fps) 
        frame_count = 0

        # Schedule the initial frame update
        root.after(
            1, lambda : 
            update_frame(
                root, cap, detector, video_label, 
                frame_count, interval, client, labels))

        
        # Start the Tkinter main loop
        root.mainloop()
        
        # Release the video capture object
        cap.release()
    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    main()