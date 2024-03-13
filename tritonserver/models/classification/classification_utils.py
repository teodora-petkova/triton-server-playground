from typing import List
import numpy as np
from PIL import Image
import yaml

import torch
from torchvision import transforms

class ClassificationUtils:

    @staticmethod
    def preprocess(image_path: str) -> np.array:
        image = Image.open(image_path).convert('RGB')
        
        # image = image.resize((224, 224))
        
        # image_array = np.array(image)
        
        # # normalize pixel values
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image_array = (image_array / 255.0 - mean) / std


        # Resize to 256x256, then center-crop to 224x224 
        # (to match the resnet image size)
        transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        transformed_image = transform(image)
        image_array = np.array(transformed_image)
        
        # reshape the image array to match the model's input shape
        #image_array = np.transpose(image_array, (2, 0, 1))
        image_array = image_array.astype(np.float32)

        return image_array
    
    
    @staticmethod
    def _get_result_as_json_dict(labels, confidences):
       
        categories = []
        for label, conf in zip(labels, confidences):

            categories.append(
                {
                    "confidence": conf * 100,
                    "name": {
                        "en": label
                    }
                })


        dict_result = {
            "result": {
                "categories": categories
            },
            "status": {
                    "text": "",
                    "type": "success"
                }
            }
        
        return dict_result
    


    @staticmethod
    def postprocess(scores: np.array, labels: List[str]):
        scores = np.copy(scores)
        scores_tensor = torch.from_numpy(scores)
        
        softmax = torch.nn.Softmax(dim=-1)
        output = softmax(scores_tensor)[0].cpu().detach().numpy()

        # indices = np.flip(np.argsort(output))
        # confs = np.array(output)[indices]
        # labels = np.array(labels)[indices]

        ind = np.argmax(output)
        labels = np.array([labels[ind]])
        confs = np.array([output[ind]])
        #return label, conf

        return ClassificationUtils._get_result_as_json_dict(labels, confs)
    

    @staticmethod
    def get_labels(filepath: str):
        labels = []
        with open(filepath, 'r', encoding='utf-8') as file_pointer:
            for line in file_pointer:
                labels.append(line.strip())
        return labels
