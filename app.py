from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

class InferlessPythonModel:
  def initialize(self):
        self.llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
      
  def infer(self, inputs):
        prompt = inputs["prompt"]
        content_url = inputs["content_url"]
        content_type = inputs.get("content_type","image")
        system_prompt = inputs.get("system_prompt","You are a helpful assistant.")
        temperature = float(inputs.get("temperature",0.7))
        top_p = float(inputs.get("top_p",0.1))
        repetition_penalty = float(inputs.get("repetition_penalty",1.18))
        top_k = int(inputs.get("top_k",40))
        max_tokens = int(inputs.get("max_tokens",256))
        max_pixels = int(inputs.get("max_pixels",12845056))
        max_duration = int(inputs.get("max_duration",60))

        sampling_params = SamplingParams(temperature=temperature,top_p=top_p,repetition_penalty=repetition_penalty,
                                         top_k=top_k,max_tokens=max_tokens)
        if content_type == "image":
            content = {
                "type": "image",
                "image": content_url,
                "max_pixels": max_pixels,
            }      
        else:
            content = {
                "type": "video",
                "video": content_url,
                "max_duration": max_duration
            }  
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                content,
                {"type": "text","text": prompt},
             ]},
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return {"generated_result": generated_text}

  def finalize(self):
    self.llm = None
