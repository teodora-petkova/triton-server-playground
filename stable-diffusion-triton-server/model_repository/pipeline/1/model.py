import json

# sources:
# https://colab.research.google.com/drive/1_kbRZPTjnFgViPrmGcUsaszEdYa8XTpq?usp=sharing#scrollTo=pPG1gZlWO2HW
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from torch.utils.dlpack import from_dlpack, to_dlpack
from tqdm.auto import tqdm
from transformers import CLIPTokenizer


class TritonPythonModel:

    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "generated_image"
            )["data_type"]
        )

        # tokenizer
        # source: https://huggingface.co/openai/clip-vit-large-patch14/tree/main
        self.tokenizer = CLIPTokenizer.from_pretrained("/models/pipeline/1/openai/clip-vit-large-patch14")
        
        # scheduler
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        
        # unet - denoising model - retrieving the noise 
        # source: https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main/unet
        self.unet = UNet2DConditionModel.from_pretrained(
            '/models/pipeline/1/CompVis/stable-diffusion-v1-4',
            #"CompVis/stable-diffusion-v1-4",
            subfolder="unet",
            #revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True,
        ).to("cuda")


    def _get_tokenized_text(self, prompt, negative_prompt):
        # tokenize the input text
        tokenized_text = self.tokenizer(
            [prompt], 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length,
            truncation=True, 
            return_tensors='pt'
        ).input_ids

        uncond_text = ['']
        if negative_prompt != None:
            uncond_text = [negative_prompt]
        # do the same for unconditional text
        tokenized_text_uncond = self.tokenizer(
            uncond_text, #* len(prompt), # ???
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt'
        ).input_ids
        
        full_tokenized_text = np.concatenate(
        [
            tokenized_text_uncond.numpy().astype(np.int32),
            tokenized_text.numpy().astype(np.int32),
        ])

        return full_tokenized_text


    def _get_text_embeddings(self, tokenized_text_input_ids):
        
        input_ids_1 = pb_utils.Tensor(
            "input_ids", tokenized_text_input_ids
        )

        encoding_request = pb_utils.InferenceRequest(
            model_name="text_encoder",
            requested_output_names=["last_hidden_state"],
            inputs=[input_ids_1],
        )

        response = encoding_request.exec()
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        else:
            text_embeddings = pb_utils.get_output_tensor_by_name(
                response, "last_hidden_state"
            )
        text_embeddings = from_dlpack(text_embeddings.to_dlpack()).clone()
        text_embeddings = text_embeddings.to("cuda")

        return text_embeddings


    def _produce_latents(
            self,
            text_embeddings, 
            height=512, width=512,
            num_inference_steps=50,
            guidance_scale=7.5):

        # random noisy latent space / image
        torch.manual_seed(42)
        latents = torch.randn((
            text_embeddings.shape[0] // 2, 
            self.unet.in_channels, 
            height // 8, width // 8)
        ).to("cuda")

        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.sigmas[0]

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance
            # to avoid doing two forward passes.
            # sth like a normalization?
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )
            
            # predict the noise residual
            with torch.no_grad(), torch.autocast("cuda"):
                noise_pred = self.unet(
                    latent_model_input, t, 
                    encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred, self.scheduler.timesteps[i], latents
            ).prev_sample

        return latents

    
    def _decode_image_latents(self, latents):
        # reverse scaling (done during training)
        latents = 1 / 0.18215 * latents

        # vae request
        input_latent_1 = pb_utils.Tensor.from_dlpack(
            "latent_sample", to_dlpack(latents)
        )

        decoding_request = pb_utils.InferenceRequest(
            model_name="vae",
            requested_output_names=["sample"],
            inputs=[input_latent_1],
        )

        decoding_response = decoding_request.exec()
        if decoding_response.has_error():
            raise pb_utils.TritonModelException(decoding_response.error().message())
        else:
            decoded_image = pb_utils.get_output_tensor_by_name(
                decoding_response, "sample"
            )
        decoded_image = from_dlpack(decoded_image.to_dlpack()).clone()

        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
        decoded_image = decoded_image.detach().cpu().permute(0, 2, 3, 1).numpy()
        decoded_image = (decoded_image * 255).round().astype("uint8")

        return decoded_image


    def execute(self, requests):
        responses = []
        for request in requests:
            # 0. get the prompt input
            prompt = pb_utils.get_input_tensor_by_name(
                request, "prompt")
            input_text = prompt.as_numpy()[0][0].decode()

            negative_prompt = pb_utils.get_input_tensor_by_name(
                request, "negative_prompt")
            negative_input_text = negative_prompt.as_numpy()[0][0].decode()

            # 1. tokenize the text - clip vit
            tokenized_text_input_ids = self._get_tokenized_text(
                input_text, negative_input_text)

            # 2. query the text_encoding model - get the text embeddings
            text_embeddings = self._get_text_embeddings(tokenized_text_input_ids)

            # 3. run the scheduler 
            latents = self._produce_latents(text_embeddings)

            # 4. vae decoding
            decoded_image = self._decode_image_latents(latents)
           
            # sending results
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_image",
                        np.array(decoded_image, dtype=self.output_dtype),
                    )
                ]
            )
            responses.append(inference_response)
        return responses
