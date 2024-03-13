import argparse
from datetime import datetime
import io
from pathlib import Path
import random
import re
from tempfile import TemporaryDirectory
import time
from typing import BinaryIO
import uuid

from PIL import Image, ImageDraw
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
import torch
from ultralytics import YOLO
import uvicorn


DEVICE = 'cuda:0'

app = FastAPI()
torch_device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

detection_model = None


async def load_image_file(image_file: BinaryIO) -> Image:
    contents = await image_file.read()
    pil_image = Image.open(io.BytesIO(contents))
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    return pil_image


def load_detector(model_name, weights_path, device):
    print(f'Loading model at {weights_path}')

    model = YOLO('yolov8n.pt')  # Load an official Detect model

    # Get class names
    names = list(model.names.values())
    print(f'\n{model_name} class names:')
    print('\n'.join(names))

    return model


def inference(model, image: Image):

    start_time = time.time()

    raw_results = model([image])

    elapsed_time = time.time() - start_time
    print(f'[DETECTION INFERENCE] elapsed time: {elapsed_time:.6f} seconds')
        
    bboxes = raw_results[0].boxes

    detections = []

    for i, result in enumerate(bboxes.xyxy):

        xmin = round(float(result[0]))
        ymin = round(float(result[1]))
        xmax = round(float(result[2]))
        ymax = round(float(result[3]))

        conf = float(bboxes.conf[i]) * 100

        # # filter bounding boxes
        # if conf < 50:
        #     continue

        class_id = int(bboxes.cls[i])
        label = model.names[class_id]

        print(
            'detection raw results: ',
            xmin, ymin, xmax, ymax, conf, class_id, label)

        detections.append({
            'name': {
                'en': label
            },
            'coordinates': {
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'width': xmax - xmin,
                'height': ymax - ymin,
            },
            'confidence': conf
        })

    return detections


def get_colors_per_label(num_labels):
    colors_per_label = {}
    for i in range(num_labels):
        colors_per_label[i] = (
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200),
        )
    return colors_per_label


def get_image_with_drawn_bboxes(pil_image: Image, detections, labels_to_id):
   
    colors = get_colors_per_label(len(labels_to_id))

    # magic numbers for visualization
    line_size = 3
    text_offset = 10

    draw = ImageDraw.Draw(pil_image)

    for detection in detections:
        coords = detection['coordinates']
        label = detection['name']['en']
        conf = round(detection['confidence'], 2)
        box = (coords['xmin'], coords['ymin'], coords['xmax'], coords['ymax'])

        #print(labels_to_id[label])
        
        c = colors[int(labels_to_id[label])]
        draw.rectangle(box, width=line_size, outline=c)

        # text
        text_box = (
            box[0] + line_size,
            box[1] + line_size,
            box[0] + line_size + len(label) * text_offset,
            box[1] + line_size + text_offset,
        )

        draw.rectangle(text_box, fill='black')
        draw.text((text_box[0], text_box[1]), f'{label}-{conf}', fill=c)

    return pil_image


def get_unique_filename():
    uuid_str = uuid.uuid4().hex
    now = datetime.now()
    date = now.date()
    dtime = now.time()
    unique_filename = re.sub(':|\.|-', '', f'{date}{dtime}{uuid_str}')
    return unique_filename


@app.on_event('startup')
def startup_event():
    print('Startup: loading the model')

    global detection_model
    detection_model = load_detector('detection', './yolov8n.pt', torch_device)


async def get_temp_dir():
    dir = TemporaryDirectory()
    try:
        yield dir.name
    finally:
        del dir


@app.post('/detections/')
async def detections(
    image_file: UploadFile = File(...),
    tempdir = Depends(get_temp_dir)):
    try:
        pil_img = await load_image_file(image_file)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f'Failed to open the image: {e}')

    labels_to_id = {
        n: i for i, n in enumerate(list(detection_model.names.values()))}

    bboxes = inference(detection_model, pil_img)

    img_to_show = get_image_with_drawn_bboxes(pil_img, bboxes, labels_to_id)

    Path(tempdir).mkdir(parents=True, exist_ok=True)

    extension = Path(image_file.filename).suffix
    image_filepath = f'{str(Path(tempdir) / get_unique_filename())}{extension}'
    img_to_show.save(image_filepath)

    return FileResponse(image_filepath, media_type="image/jpeg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', default=8076, type=int)
    args = parser.parse_args()

    port = args.port
    uvicorn.run('api:app', host='0.0.0.0', port=port, reload=False)
