from functools import wraps
import sys
import time
import numpy as np
from pathlib import Path
import cv2

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

from PIL import Image

from tritonserver.models.classification.classification_model import (
     ClassificationModel)
from tritonserver.models.classification.classification_utils import (
    ClassificationUtils )
from tritonserver.models.detection.detection_model import DetectionModel
from tritonserver.models.detection.detection_utils import DetectionUtils

from tritonserver.client.triton_client_utils import timeit, draw_boxes

TRITON_HTTP_PORT = 18000
TRITON_GRPC_PORT = 18001

DATASET_YAML = 'tritonserver/deploy/model_repository/yolo_detection/1/coco8.yaml'

@timeit
def init_triton_grcpclient() -> grpcclient.InferenceServerClient:
    grpc_url = f'localhost:{TRITON_GRPC_PORT}'
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=grpc_url,
            verbose=False,
            ssl=False,
        )
    # trunk-ignore(pylint/C0103)
    # trunk-ignore(pylint/W0703)
    except Exception as e:
        print(f'Creation of the grcp triton inference\
                client creation failed: {str(e)}')
        sys.exit()
    return triton_client


@timeit
def classification_inference(preprocessed_images, model, client, clientmodule):

    infer_input = clientmodule.InferInput(
        model.input_name, preprocessed_images.shape, datatype='FP32')
    infer_input.set_data_from_numpy(preprocessed_images)#, binary_data=True)

    infer_output = clientmodule.InferRequestedOutput(model.output_name)

    response = client.infer(
        model_name=model.name, 
        inputs=[infer_input], 
        outputs=[infer_output])

    result = response.as_numpy(model.output_name)

    inference_result = ClassificationUtils.postprocess(result, model.labels)

    print(f'[INFO] inference: {inference_result}')

    return inference_result
    

def classification_http_inference(preprocessed_images, model, client):
    return classification_inference(preprocessed_images, model, client, httpclient)


def classification_grpc_inference(preprocessed_images, model, client):
    return classification_inference(preprocessed_images, model, client, grpcclient)


@timeit
def detection_inference(input_images, orig_shape, model, client, labels):
    inputs = []
    outputs = []

    inputs.append(
        grpcclient.InferInput(
            model.input_name, [*input_images.shape], model.fp_mode)
    )
    
    # Initialize the data
    inputs[-1].set_data_from_numpy(input_images)
    outputs.append(grpcclient.InferRequestedOutput(model.output_name))

    # Test with outputs
    response = client.infer(
        model_name=model.name, inputs=inputs, outputs=outputs
    )

    results = response.as_numpy(model.output_name)

    inference_result = DetectionUtils.postprocess(
        results, (orig_shape[0], orig_shape[1]), 
        model.input_hw, labels=labels)

    print(f'[INFO] inference: {inference_result}')

    return inference_result


def main():
    labels = ClassificationUtils.get_labels('./tritonserver/deploy/model_repository/classification_onnx/labels.txt')
    
    classification_model = ClassificationModel(name='classification', labels=labels)

    onnx_model = ClassificationModel(name='classification_onnx', labels=labels)

    tensorrt_model = ClassificationModel(name='classification_tensorrt', labels=labels)

    detector = DetectionModel(name='yolo_detection')
    labels = DetectionUtils.get_labels(DATASET_YAML)

    client = init_triton_grcpclient()

    for image_filepath in Path('./tritonserver/images/').iterdir():

        print(f'[INFO] Processing image: {image_filepath}')

        imgs = np.array([ClassificationUtils.preprocess(image_filepath)])
        
        classification_inference(imgs, classification_model, client, grpcclient)

        classification_inference(imgs, onnx_model, client, grpcclient)

        classification_inference(imgs, tensorrt_model, client, grpcclient)

        # start_time = time.time()

        # pil_img = Image.open(image_filepath)
        # img = np.asarray(pil_img)
        # #bgr_img = cv2.imread(str(image_filepath))
        # #img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
        # preprocessed_image = DetectionUtils.preprocess(img)
        # preprocessed_images = np.array([preprocessed_image])[0]
        
        # triton_bboxes, triton_scores, triton_class_labels = detection_inference(
        #     preprocessed_images, img.shape, detector, client, labels)

        # end_time = time.time() - start_time
        # print(f'[DEBUG] processing image: {end_time} seconds')
        
        # img_to_show = Image.fromarray(draw_boxes(img, triton_bboxes, triton_scores, triton_class_labels))     
        # img_to_show.save('test.jpg')



if __name__ == '__main__':
    main()
