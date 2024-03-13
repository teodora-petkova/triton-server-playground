FROM nvcr.io/nvidia/tritonserver:23.09-py3 

RUN apt-get update

RUN pip install --upgrade pip

# PyTorch & Transformers Libs
RUN pip install torch torchvision torchaudio
RUN pip install transformers ftfy scipy accelerate
RUN pip install diffusers==0.9.0
RUN pip install transformers[onnxruntime]
