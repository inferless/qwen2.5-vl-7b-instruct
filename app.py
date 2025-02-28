from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import inferless
from typing import Optional
from pydantic import BaseModel, Field

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default=="What does this diagram illustrate?")
    content_url: str = Field(default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
    content_type: Optional[str] = "image"
    system_prompt: Optional[str] = "You are a helpful coding bot."
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    repetition_penalty: Optional[float] = 1.18
    max_tokens: Optional[int] = 256
    max_pixels: Optional[int] = 12845056
    top_k: Optional[int] = 40
    max_duration: Optional[int] = 60
    fps: Optional[int] = 1

@inferless.response
class ResponseObjects(BaseModel):
    generated_result: str = Field(default="Test output")

class InferlessPythonModel:
    def initialize(self):
        self.llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    def infer(self, request: RequestObjects) -> ResponseObjects:
        sampling_params = SamplingParams(temperature=request.temperature,top_p=request.top_p,repetition_penalty=request.repetition_penalty,
                                         top_k=request.top_k,max_tokens=request.max_tokens)
        if request.content_type == "image":
            content = {
                "type": "image",
                "image": request.content_url,
                "max_pixels": request.max_pixels,
            }      
        else:
            content = {
                "type": "video",
                "video": request.content_url,
                "max_duration": request.max_duration,
                "max_pixels": request.max_pixels,
                "fps": request.fps
            }
        
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": [
                content,
                {"type": "text","text": request.prompt},
             ]},
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs,video_kwargs = process_vision_info(messages,return_video_kwargs=True)
        
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        generateObject = ResponseObjects(generated_result = generated_text)
        
        return generateObject
        
    def finalize(self):
        self.llm = None
