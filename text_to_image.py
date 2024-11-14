import torch
import uuid
import os

from diffusers import StableDiffusionPipeline



class text_to_image:
    def __init__(self):
        self.device         = "cpu"
        self.torch_dtype    = torch.float32
        self.pipe           = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = self.torch_dtype).to(self.device)
        self.a_prompt       = 'best quality, extremely detailed'
        self.n_prompt       = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                                'fewer digits, cropped, worst quality, low quality'

    def generate_image(self, text):
        image_filename = os.path.join('Images', str(uuid.uuid4())[0:8] + ".png")
        prompt = text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt = self.n_prompt).images[0]
        image.save(image_filename)
        return image_filename