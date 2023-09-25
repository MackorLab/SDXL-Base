!pip install --quiet google-colab transformers diffusers accelerate torch ftfy modin[all] xformers invisible_watermark gradio==3.29.0 >/dev/null
import gradio as gr
import torch
import numpy as np
import modin.pandas as pd
from PIL import Image
import pandas as pd
from diffusers import DiffusionPipeline



device = "cuda" if torch.cuda.is_available() else "cpu"




pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
pipe = pipe.to(device)





def genie (Prompt, negative_prompt, height, width, scale, steps, seed):
    generator = torch.Generator(device=device).manual_seed(seed)


    image = pipe(Prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=steps, guidance_scale=scale, generator=generator).images[0]

    return image

gr.Interface(
    fn=genie,
    inputs=[
        gr.inputs.Textbox(label='Что вы хотите, чтобы ИИ генерировал?'),
        gr.inputs.Textbox(label='Что вы не хотите, чтобы ИИ генерировал?', default='(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, wrinkles, old face'),
        gr.Slider(512, 1024, 768, step=128, label='Высота изображения'),
        gr.Slider(512, 1024, 768, step=128, label='Ширина изображения'),
        gr.Slider(1, maximum=15, value=10, step=0.1, label='Шкала расхождения'),
        gr.Slider(1, maximum=100, value=25, step=1, label='Количество итераций'),
        gr.Slider(label="Точка старта функции", minimum=1, step=1, maximum=9999999999999999, randomize=True)
    ],


    outputs='image',
    title='DIAMONIK7777 - txt2img - SDXL - Base',
    description="<p style='text-align: center'>Будь в курсе обновлений <a href='https://vk.com/public221489796'>ПОДПИСАТЬСЯ</a></p>",
    article="<br><br><p style='text-align: center'>Генерация индивидуальной модели с собственной внешностью <a href='https://vk.com/im?sel=-221489796'>ПОДАТЬ ЗАЯВКУ</a></p><br><br><br><br><br>",


).launch(debug=True, max_threads=True, share=True, inbrowser=True)
