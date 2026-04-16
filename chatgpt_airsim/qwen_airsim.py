import argparse
import base64
import json
import math
import os
import re
import sys
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
parser.add_argument(
    "--code-timeout",
    type=float,
    default=None,
    help="模型生成代码的执行超时秒数；<=0 表示禁用超时。",
)
args = parser.parse_args()

with open("config.json", "r") as f:
    config = json.load(f)

CODE_EXEC_TIMEOUT_SECONDS = args.code_timeout
if CODE_EXEC_TIMEOUT_SECONDS is None:
    CODE_EXEC_TIMEOUT_SECONDS = config.get("CODE_EXEC_TIMEOUT_SECONDS", 8)

try:
    CODE_EXEC_TIMEOUT_SECONDS = float(CODE_EXEC_TIMEOUT_SECONDS)
except (TypeError, ValueError):
    CODE_EXEC_TIMEOUT_SECONDS = 8

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
    "不要执行代码，请中文分析这张无人机前视图，描述关键物体、潜在障碍物和安全飞行建议；"
)
FALLBACK_MODEL_NAME = "qwen3-vl-flash"
MODEL_NAME = config.get("QWEN_MODEL", FALLBACK_MODEL_NAME)


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
    request_model = MODEL_NAME
    try:
        completion = openai.ChatCompletion.create(
            model=request_model,
            messages=chat_history,
            temperature=0
        )
    except openai.error.InvalidRequestError as e:
        err = str(e)
        unavailable = (
            "does not exist" in err
            or "do not have access" in err
            or "you do not have access" in err
        )
        if request_model != FALLBACK_MODEL_NAME and unavailable:
            print(
                colors.YELLOW
                + f"Model '{request_model}' unavailable, fallback to '{FALLBACK_MODEL_NAME}'."
                + colors.ENDC
            )
            request_model = FALLBACK_MODEL_NAME
            completion = openai.ChatCompletion.create(
                model=request_model,
                messages=chat_history,
                temperature=0
            )
        else:
            raise
    if args.debug_api:
        image_len = len(image_base64) if image_base64 is not None else 0
        print(colors.BLUE + f"[DEBUG] request.model={request_model}, with_image={image_base64 is not None}, image_b64_len={image_len}" + colors.ENDC)
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

BLOCKED_CODE_PATTERNS = [
    (re.compile(r"(^|\W)import\s+", re.IGNORECASE), "不允许导入新模块 (import)"),
    (re.compile(r"(^|\W)from\s+.+\s+import\s+", re.IGNORECASE), "不允许导入新模块 (from ... import ...)"),
    (re.compile(r"(^|\W)exec\s*\(", re.IGNORECASE), "不允许调用 exec()"),
    (re.compile(r"(^|\W)eval\s*\(", re.IGNORECASE), "不允许调用 eval()"),
    (re.compile(r"(^|\W)open\s*\(", re.IGNORECASE), "不允许直接读写本地文件 (open)"),
    (re.compile(r"(^|\W)os\s*\.\s*", re.IGNORECASE), "不允许访问 os 模块"),
    (re.compile(r"(^|\W)sys\s*\.\s*", re.IGNORECASE), "不允许访问 sys 模块"),
    (re.compile(r"(^|\W)(exit|quit)\s*\(", re.IGNORECASE), "不允许退出主程序 (exit/quit)"),
]

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


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


def validate_generated_code(code):
    """检查模型代码是否包含高风险语句。

    Args:
        code (str): 模型生成的 Python 代码。

    Returns:
        tuple[bool, str]:
            - 第 1 项表示是否通过校验。
            - 第 2 项为失败原因，成功时为空字符串。
    """
    if not code.strip():
        return False, "代码块为空。"

    if len(code) > 4000:
        return False, "代码过长，已拒绝执行。"

    for pattern, reason in BLOCKED_CODE_PATTERNS:
        if pattern.search(code):
            return False, reason

    return True, ""


def run_generated_code(code):
    """在受限环境中执行模型代码，避免主程序退出。"""
    ok, reason = validate_generated_code(code)
    if not ok:
        print(colors.RED + f"Blocked generated code: {reason}" + colors.ENDC)
        return

    sandbox_globals = {
        "__builtins__": SAFE_BUILTINS,
        "aw": aw,
        "np": np,
        "math": math,
        "time": time,
    }

    try:
        execute_generated_code_with_timeout(code, sandbox_globals, CODE_EXEC_TIMEOUT_SECONDS)
    except TimeoutError as e:
        print(colors.RED + f"Generated code timed out: {e}" + colors.ENDC)
    except BaseException as e:
        # 捕获包括 SystemExit 在内的异常，防止模型代码导致主循环退出。
        print(colors.RED + f"Generated code failed but chatbot is still running: {e}" + colors.ENDC)


def execute_generated_code_with_timeout(code, sandbox_globals, timeout_seconds):
    """以可选超时方式执行代码。

    说明：该方案可中断大多数 Python 层死循环；
    对底层阻塞调用（例如某些 C 扩展中的长时间阻塞）可能无法立即中断。
    """
    if timeout_seconds is None or timeout_seconds <= 0:
        exec(code, sandbox_globals, {})
        return

    deadline = time.time() + timeout_seconds

    def timeout_tracer(frame, event, arg):
        if time.time() > deadline:
            raise TimeoutError(f"execution exceeded {timeout_seconds:.1f}s")
        return timeout_tracer

    old_trace = sys.gettrace()
    try:
        sys.settrace(timeout_tracer)
        exec(code, sandbox_globals, {})
    finally:
        sys.settrace(old_trace)


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
        run_generated_code(code)
        print("Done!\n")
