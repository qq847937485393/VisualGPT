import torch

from PIL import Image as image

from config import debug_print

from transformers import BlipProcessor
from transformers import BlipForConditionalGeneration



class image_captioning:
    def __init__(self):
        debug_print("image_captioning.__init__")

        self.device         = "cpu"
        self.torch_dtype    = torch.float32
        self.processor      = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model          = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype = self.torch_dtype).to(self.device)

        return

    def generate_caption(self, image_path : str) -> str:
        inputs      = self.processor(image.open(image_path).convert("RGB"), return_tensors = "pt").to(self.device, self.torch_dtype)
        out         = self.model.generate(**inputs, max_new_tokens = 4000)
        captions    = self.processor.decode(out[0], skip_special_tokens = True)
        debug_print(f"image_captioning.generate_caption | {image_path} : {captions}")

        return captions