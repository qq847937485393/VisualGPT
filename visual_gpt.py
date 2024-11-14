import gradio
import os
import uuid
import numpy
import re
import config

from config import debug_print
from image_captioning import image_captioning
from text_to_image import text_to_image
from visual_analysis import visual_analysis

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_openai import OpenAI

from PIL import Image as image





class conversation_bot:
    def __init__(self):
        debug_print("conversation_bot.__init__")

        self.foundation_models = {"image_captioning": image_captioning(), "text_to_image": text_to_image(), "visual_analysis": visual_analysis()}
        debug_print("foundation models initialized")

        self.llm = OpenAI(model_name = config.open_ai.model, openai_api_key = config.open_ai.api_key, temperature = 0)
        debug_print("llm initialized")

        self.tools =    [
            Tool(   name = "Caption",
                    description = "Useful when you want to know what is inside the photo, receives image_path as input. The input to this tool should be a string, representing the image_path.",
                    func = self.foundation_models["image_captioning"].generate_caption),
            Tool(   name = "Text to image",
                    description = "Useful when you want to generate an image from a user input text and save it to a file. Like generate an image of an object or something, or generate an image that includes some objects. The input to this tool should be a string, representing the text used to generate image.",
                    func = self.foundation_models["text_to_image"].generate_image),
            Tool(   name = "Answer Question About The Image",
                    description = "Useful when you need an answer for a question based on an image. Like what is the background color of the last image, how many cats in this figure, what is in this figure. The input to this tool should be a comma seperated string of two, representing the image_path and the question",
                    func = self.foundation_models["visual_analysis"].generate_analysis)
                        ]

        self.memory = ConversationBufferMemory(memory_key = "chat_history", output_key = 'output')

        self.agent = initialize_agent(self.tools, self.llm, agent = "conversational-react-description", verbose = True, memory = self.memory, return_intermediate_steps = True, agent_kwargs={"prefix": config.visual_gpt_prefix, "format_instructions": config.visual_gpt_format_instructions, "suffix": config.visual_gpt_suffix}, )
        debug_print("agent initialized")

        return

    def txt_submit(self, text, chat_history):
        #self.agent.memory.buffer    = cut_dialogue_history(self.agent.memory.buffer, 500)
        res                         = self.agent({"input": text})
        res["output"]               = res["output"].replace("\\", "/")
        response                    = re.sub('(image/\s*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
        chat_history.append({"role": "user", "content": text})
        chat_history.append({"role": "assistant", "content": response})
        return ["", chat_history]


    def img_upload(self, img, text, chat_history):
        debug_print("conversation_bot.img_upload | resizing image")
        image_filename          = os.path.join("Images", f"{str(uuid.uuid4())[:8]}.png")
        _img                    = image.open(img.name)
        width, height           = _img.size
        ratio                   = min(512 / width, 512 / height)
        width_new, height_new   = (round(width * ratio), round(height * ratio))
        width_new               = int(numpy.round(width_new / 64.0)) * 64
        height_new              = int(numpy.round(height_new / 64.0)) * 64
        _img                     = _img.resize((width_new, height_new))
        _img                     = _img.convert("RGB")
        _img.save(image_filename, "PNG")
        debug_print(f"conversation_bot.img_upload | {image_filename} saved, {width}x{height} ---> {width_new}x{height_new}")

        caption = self.foundation_models["image_captioning"].generate_caption(image_filename)

        self.agent.memory.save_context({"input": f"Provide a figure named {image_filename}. The description is: {caption}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\"."}, {"output": "Received."})

        chat_history.append({"role": "user", "content": {"path": image_filename}})
        chat_history.append({"role": "assistant", "content": "Received."})
        return [text, chat_history]



def main():
    bot = conversation_bot()

    with gradio.Blocks() as ui:
        chatbot = gradio.Chatbot(elem_id = "chatbot", label = f"llm_model = {config.open_ai.model}", type = "messages", height = 900)
        state   = gradio.State([])

        with gradio.Row(visible = True) as input_raws:
            with gradio.Column(scale = 15):
                txt = gradio.Textbox(placeholder = "Enter a message here.")
            with gradio.Column(scale = 5, min_width = 0):
                clear = gradio.Button("Clear")
            with gradio.Column(scale = 5, min_width = 0):
                btn = gradio.UploadButton(label = "Upload Image", file_types = ["image"])

        #clear.click(lambda: [], None, chatbot)
        #clear.click(lambda: [], None, state)

        txt.submit(bot.txt_submit, [txt, chatbot], [txt, chatbot])
        txt.submit(lambda: "", None, chatbot)

        btn.upload(bot.img_upload, [btn, txt, chatbot], [txt, chatbot])

        #clear.click(test)
        #clear.click(lambda: [], None, chatbot)
        #clear.click(lambda: [], None, state)

    ui.launch(server_name = "127.0.0.1", server_port=5679)
    return



if (__name__ == "__main__"):
    main()