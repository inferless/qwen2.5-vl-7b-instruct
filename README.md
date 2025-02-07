# Tutorial - Deploy Qwen2-VL-7B-Instruct using Inferless
[Qwen2-VL-7B-Instruct](https://huggingface.co/qwen/Qwen2-VL-7B-Instruct) is a state-of-the-art multimodal language model developed by Alibaba Cloud’s Qwen team. This model is part of the Qwen2 series and is tailored for instruction-based tasks, excelling in visual understanding and multilingual processing. It features a dense transformer architecture with 7 billion parameters, enabling it to handle complex multimodal inputs effectively. The model incorporates Naive Dynamic Resolution for processing images of arbitrary resolutions and Multimodal Rotary Position Embedding (M-ROPE) to capture 1D textual, 2D visual, and 3D video positional information, enhancing its performance across various tasks.

## TL;DR:
- Deployment of Qwen2-VL-7B-Instruct model using [vllm](https://github.com/vllm-project/vllm).
- You can expect an average latency of `5.83 seconds` for image and	`21.61 seconds` for video with generating max tokens of `256`. This setup has an average cold start time of `38.17 seconds`.
- Dependencies defined in `inferless-runtime-config.yaml`.
- GitHub/GitLab template creation with `app.py`, `inferless-runtime-config.yaml` and `inferless.yaml`.
- Model class in `app.py` with `initialize`, `infer`, and `finalize` functions.
- Custom runtime creation with necessary system and Python packages.
- Model import via GitHub with `input_schema.py` file.
- Recommended GPU: NVIDIA A100 for optimal performance.
- Custom runtime selection in advanced configuration.
- Final review and deployment on the Inferless platform.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **inferless-runtime-config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the `Add a custom model` button.

- Select `Github` as the method of upload from the Provider list and then select your Github Repository and the branch.
- Choose the type of machine, and specify the minimum and maximum number of replicas for deploying your model.
- Configure Custom Runtime ( If you have pip or apt packages), choose Volume, Secrets and set Environment variables like Inference Timeout / Container Concurrency / Scale Down Timeout
- Once you click “Continue,” click Deploy to start the model import process.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer <your_api_key>' \
    --data '{
    "inputs": [
        {
            "name": "prompt",
            "shape": [
                1
            ],
            "data": [
                "What does this diagram illustrate?"
            ],
            "datatype": "BYTES"
        },
        {
            "name": "content_url",
            "shape": [
                1
            ],
            "data": [
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            ],
            "datatype": "BYTES"
        },
        {
            "name": "content_type",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                "image"
            ],
            "datatype": "BYTES"
        },
        {
            "name": "system_prompt",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                "You are a helpful coding bot."
            ],
            "datatype": "BYTES"
        },
        {
            "name": "temperature",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                0.7
            ],
            "datatype": "FP64"
        },
        {
            "name": "top_p",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                0.1
            ],
            "datatype": "FP64"
        },
        {
            "name": "repetition_penalty",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                1.18
            ],
            "datatype": "FP64"
        },
        {
            "name": "max_tokens",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                256
            ],
            "datatype": "INT64"
        },
        {
            "name": "max_pixels",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                12845056
            ],
            "datatype": "INT64"
        },
        {
            "name": "top_k",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                40
            ],
            "datatype": "INT64"
        },
        {
            "name": "max_duration",
            "optional": true,
            "shape": [
                1
            ],
            "data": [
                60
            ],
            "datatype": "INT64"
        }
    ]
}'
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](https://docs.inferless.com/model-import/input-output-schema) for more.

```python
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
```

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting to `None`.
```python
def finalize(self):
    self.llm = None
```


For more information refer to the [Inferless docs](https://docs.inferless.com/).
