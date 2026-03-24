import argparse
import base64
import json
import math
import os
import re
import time
from datetime import datetime

import numpy as np
import openai
from airsim_wrapper import *

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="prompts/airsim_basic.txt")
parser.add_argument("--sysprompt", type=str, default="system_prompts/airsim_basic.txt")
parser.add_argument("--vision-trigger", type=str, default="!vision")
parser.add_argument("--debug-api", action="store_true")
args = parser.parse_args()

with open("config.json", "r") as f:
    config = json.load(f)

print("Initializing Qwen...")
# 请在 config.json 中增加 QWEN_API_KEY 配置你的阿里云百炼 API Key
openai.api_key = config.get("QWEN_API_KEY", config.get("OPENAI_API_KEY"))
openai.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"

with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

chat_history = [
    {
        "role": "system",
        "content": sysprompt
    },
    {
        "role": "user",
        "content": "move 10 units up"
    },
    {
        "role": "assistant",
        "content": """```python
aw.fly_to([aw.get_drone_position()[0], aw.get_drone_position()[1], aw.get_drone_position()[2]+10])
```

This code uses the `fly_to()` function to move the drone to a new position that is 10 units up from the current position. It does this by getting the current position of the drone using `get_drone_position()` and then creating a new list with the same X and Y coordinates, but with the Z coordinate increased by 10. The drone will then fly to this new position using `fly_to()`."""
    }
]

DEFAULT_VISION_PROMPT = (
    "请分析这张无人机前视图，描述关键物体、潜在障碍物和安全飞行建议；"
)
MODEL_NAME = "qwen3-vl-flash"


def get_response_model_name(completion):
    """从 API 返回对象中提取模型名。

    Args:
        completion (dict): OpenAI 兼容接口返回的响应对象。

    Returns:
        str: 模型名称；当无法解析时返回 "unknown"。
    """
    try:
        model_name = completion.get("model")
    except Exception:
        model_name = None

    if not model_name:
        try:
            model_name = completion["model"]
        except Exception:
            model_name = "unknown"
    return model_name


def ask(prompt, image_base64=None):
    """向千问模型发送请求并返回回答。

    Args:
        prompt (str): 用户输入文本。
        image_base64 (str | None): 可选的 JPEG 图片 Base64 字符串。

    Returns:
        str: 模型返回的文本内容。
    """
    user_content = prompt
    if image_base64 is not None:
        user_content = [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                },
            },
        ]

    chat_history.append(
        {
            "role": "user",
            "content": user_content,
        }
    )
    completion = openai.ChatCompletion.create(
        model="qwen3-vl-flash", # 这里换成了支持视觉的多模态千问模型
        messages=chat_history,
        temperature=0
    )
    if args.debug_api:
        image_len = len(image_base64) if image_base64 is not None else 0
        print(colors.BLUE + f"[DEBUG] request.model={MODEL_NAME}, with_image={image_base64 is not None}, image_b64_len={image_len}" + colors.ENDC)
        print(colors.BLUE + f"[DEBUG] response.model={get_response_model_name(completion)}" + colors.ENDC)

    assistant_content = completion["choices"][0]["message"]["content"]
    if assistant_content is None:
        assistant_content = ""

    chat_history.append(
        {
            "role": "assistant",
            "content": assistant_content,
        }
    )
    return assistant_content


def parse_vision_command(question):
    """解析是否触发视觉命令，并提取视觉提示词。

    Args:
        question (str): 用户输入。

    Returns:
        tuple[bool, str]:
            - 第 1 项表示是否触发视觉模式。
            - 第 2 项为发送给模型的文本提示。
    """
    trigger = args.vision_trigger
    if not question.startswith(trigger):
        return False, question

    prompt = question[len(trigger):].strip()
    if prompt == "":
        prompt = DEFAULT_VISION_PROMPT
    return True, prompt


def save_debug_image(image_base64, output_dir="vision_debug"):
    """将 Base64 图片保存为本地调试文件。

    Args:
        image_base64 (str): 图片 Base64 字符串。
        output_dir (str): 输出目录，默认值为 vision_debug。

    Returns:
        str: 已保存图片的完整路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
    return file_path


print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)


def extract_python_code(content):
    """从模型回复中提取 Python 代码块。

    Args:
        content (str): 模型回复文本。

    Returns:
        str | None: 提取出的代码字符串；若不存在代码块则返回 None。
    """
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None


class colors:  # You may need to change color settings
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


print(f"Initializing AirSim...")
aw = AirSimWrapper()
print(f"Done.")

with open(args.prompt, "r") as f:
    prompt = f.read()

ask(prompt)
print("Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")

while True:
    question = input(colors.YELLOW + "AirSim> " + colors.ENDC)

    if question == "!quit" or question == "!exit":
        break

    if question == "!clear":
        os.system("cls")
        continue

    use_vision, request_prompt = parse_vision_command(question)

    if use_vision:
        print("Capturing image from AirSim camera...")
        try:
            image_base64 = aw.get_scene_image_base64()
            saved_path = save_debug_image(image_base64)
            print(colors.GREEN + f"Saved captured image to: {saved_path}" + colors.ENDC)
            response = ask(request_prompt, image_base64=image_base64)
        except Exception as e:
            print(colors.RED + f"Vision capture failed: {e}. Falling back to text-only." + colors.ENDC)
            response = ask(request_prompt)
    else:
        response = ask(request_prompt)

    print(f"\n{response}\n")

    code = extract_python_code(response)
    if code is not None:
        print("Please wait while I run the code in AirSim...")
        exec(extract_python_code(response))
        print("Done!\n")
