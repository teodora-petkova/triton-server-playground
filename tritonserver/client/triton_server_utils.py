import sys
import time
import json
import numpy as np
from typing import List, Tuple


import tritonclient.grpc as grpcclient

from tritonserver.models.detection.detection_model import DetectionModel
from tritonserver.models.detection.detection_utils import DetectionUtils
from tritonserver.models.classification.classification_model import ClassificationModel
from tritonserver.models.classification.classification_utils import ClassificationUtils

from tritonserver.client.triton_client_utils import timeit

TRITON_HTTP_PORT = 18000
TRITON_GRPC_PORT = 18001


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
        print(
            f'Creation of the grcp triton inference client creation \
            failed: {str(e)}')
        sys.exit()
    return triton_client


def _detection_inference_raw_output(  
    input_image: np.array, 
    model: DetectionModel, 
    client: grpcclient.InferenceServerClient):
    inputs = []
    outputs = []

    # initialize the input data and setup model input and output names
    inputs.append(
        grpcclient.InferInput(
            model.input_name, [*input_image.shape], model.fp_mode)
    )

    inputs[-1].set_data_from_numpy(input_image)
    outputs.append(grpcclient.InferRequestedOutput(model.output_name))

    # infer on triton server
    response = client.infer(
        model_name=model.name, inputs=inputs, outputs=outputs
    )

    results = response.as_numpy(model.output_name)
    
    return results


@timeit
def detection_inference_json(
    input_image: np.array, 
    orig_shape: Tuple[int, int], 
    model: DetectionModel, 
    client: grpcclient.InferenceServerClient, 
    labels: List[str]) -> str:

    results = _detection_inference_raw_output(input_image, model, client)
    
    # post processing
    inference_result = DetectionUtils.postprocess_json(
        results, orig_shape, model, names=labels)

    print(f'[INFO] inference: {json.dumps(inference_result)}')

    return inference_result


@timeit
def detection_inference(
    input_image: np.array, 
    orig_shape: Tuple[int, int], 
    model: DetectionModel, 
    client: grpcclient.InferenceServerClient, 
    labels: List[str]) -> str:

    results = _detection_inference_raw_output(input_image, model, client)
    
    # post processing
    bboxes, confs, class_labels = DetectionUtils.postprocess(
        results, orig_shape, model, names=labels)

    print(f'[INFO] inference: {bboxes, confs, class_labels}')

    return bboxes, confs, class_labels



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


def classification_grpc_inference(preprocessed_images, model, client):
    return classification_inference(preprocessed_images, model, client, grpcclient)