import torch

from transformers import BlipProcessor
from transformers import BlipForQuestionAnswering

from PIL import Image as image



class visual_analysis:
    def __init__(self):
        self.torch_dtype    = torch.float32
        self.device         = "cpu"
        self.processor      = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model          = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base", torch_dtype = self.torch_dtype).to(self.device)

    def generate_analysis(self, inputs):
        image_path, question    = inputs.split(",")
        raw_image               = image.open(image_path).convert('RGB')
        inputs                  = self.processor(raw_image, question, return_tensors="pt").to(self.device, self.torch_dtype)
        out                     = self.model.generate(**inputs)
        answer                  = self.processor.decode(out[0], skip_special_tokens=True)
        return answer