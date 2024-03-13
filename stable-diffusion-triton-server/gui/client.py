"""
sample prompts:
- meditating in the forest, view from the back, dark
- pos: Kermit playing on a guitar, neg: incorrect hands, incorrect feet,  incorrect mouth, incorrect eyes, incorrect nose, incorrect head
- 
"""
import argparse

import gradio as gr
import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from tritonclient.utils import np_to_triton_dtype

parser = argparse.ArgumentParser()
parser.add_argument("--triton_url", default="localhost:18004")
args = parser.parse_args()

client = grpcclient.InferenceServerClient(url=f"{args.triton_url}")


def generate(prompt, negative_prompt):
    def get_input_text(text_in, input_name):
        text_obj = np.array([text_in], dtype="object").reshape((-1, 1))
        input_text = grpcclient.InferInput(
            input_name, text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        )
        input_text.set_data_from_numpy(text_obj)
        return input_text
        
    input_text = get_input_text(prompt, 'prompt')
    negative_input_text = get_input_text(negative_prompt, 'negative_prompt')

    output_img = grpcclient.InferRequestedOutput("generated_image")

    response = client.infer(
        model_name="pipeline",
        inputs=[input_text, negative_input_text],
        outputs=[output_img]
    )
    resp_img = response.as_numpy("generated_image")
    print(resp_img.shape)
    im = Image.fromarray(np.squeeze(resp_img.astype(np.uint8)))
    return im


with gr.Blocks() as app:
    prompt = gr.Textbox(label="Prompt")
    negative_prompt = gr.Textbox(label="Negative prompt")
    submit_btn = gr.Button("Generate")
    img_output = gr.Image().style(height=512)
    submit_btn.click(fn=generate, inputs=[prompt, negative_prompt], outputs=img_output)

app.launch(server_name="0.0.0.0")
