FROM nvcr.io/nvidia/pytorch:23.09-py3

RUN apt-get update

# PyTorch & Transformers Libs
RUN pip install transformers ftfy scipy
RUN pip install transformers[onnxruntime]
RUN pip install diffusers==0.9.0

# RUN huggingface-cli login

# build with tag
# docker run -it --gpus all -p 8888:8888 -v ${PWD}:/mount nvcr.io/nvidia/pytorch:23.09-py3
