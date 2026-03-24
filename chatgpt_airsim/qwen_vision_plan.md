# Qwen 视觉接入现状与后续计划（AirSim）

## 1. 文档目的

本文件用于同步当前代码实际实现，避免方案描述与现状不一致。

相关实现文件：
- [chatgpt_airsim/qwen_airsim.py](qwen_airsim.py)
- [chatgpt_airsim/airsim_wrapper.py](airsim_wrapper.py)

## 2. 当前已实现能力

### 2.1 触发词与路由

- 已支持前缀触发词 `!vision`（可通过 CLI 参数 `--vision-trigger` 覆盖）。
- 非视觉输入：按原路径走纯文本对话。
- 视觉输入：先抓图，再走多模态请求。

### 2.2 默认视觉提示词

当输入仅为 `!vision` 时，使用默认提示词：
- "请分析这张无人机前视图，描述关键物体、潜在障碍物和安全飞行建议；"

### 2.3 图像采集与编码

`AirSimWrapper.get_scene_image_base64()` 已实现：
- 调用 `simGetImages` 获取 Scene 图。
- 将 `image_data_uint8` 重排为 HWC 三通道数组。
- 使用 OpenCV 进行 JPEG 编码。
- 返回 base64 字符串。

当前默认参数：
- `camera_name="0"`
- `image_type=airsim.ImageType.Scene`
- `jpeg_quality=100`

注意：
- 捕获图像分辨率主要由 AirSim 运行配置决定，而不是 `qwen_airsim.py` 中的请求参数。
- 需要在 `C:\Users\<username>\Documents\AirSim\settings.json` 的 `CameraDefaults.CaptureSettings` 中修改 `ImageType: 0` 对应的 `Width` 和 `Height`。
- 修改 `settings.json` 后必须重启 AirSim 仿真进程，新的分辨率才会生效。

### 2.4 多模态请求格式

`ask(prompt, image_base64=None)` 已实现双路径：
- 无图时：`content` 为字符串。
- 有图时：`content` 为列表（text + image_url）。

图像通过 Data URL 发送：

```python
{
  "type": "image_url",
  "image_url": {
    "url": f"data:image/jpeg;base64,{image_base64}"
  }
}
```

### 2.5 失败回退与调试

- 视觉分支失败时（抓图/编码异常）自动回退为纯文本调用。
- 调试开关 `--debug-api` 已支持输出：
  - 是否携带图像。
  - `image_b64_len`。
  - 响应模型名。
- 每次视觉请求会将图片落盘到 `vision_debug/`，便于回放排查。

## 3. 与原始方案的差异

- 已完成项：
  - `!vision` 触发式多模态接入。
  - 单帧抓图与发送。
  - 失败自动回退文本路径。
- 尚未完成项：
  - `!vision <camera_name> <prompt>` 语法（当前仅支持固定相机名默认值）。
  - 失败重试与超时控制。
  - 执行前安全白名单/高风险关键词拦截（仍为提取代码后直接执行）。
  - 多帧视觉记忆与评测脚本。

## 4. 当前风险与建议

- 风险：请求体过大导致延迟提升或接口限制命中。
  - 建议：在发送前增加图片大小上限控制（先降质量，再降分辨率）。
- 风险：模型生成代码直接执行带来安全风险。
  - 建议：增加执行白名单与危险语句拦截。
- 风险：图像语义误判可能造成动作风险。
  - 建议：增加“执行前确认”开关。

## 5. 下一步落地优先级

1. 增加发送前图片大小保护（质量/分辨率自适应压缩）。
2. 增加 `!vision <camera_name> <prompt>`，支持多相机切换。
3. 增加代码执行安全门（白名单 + 拦截规则）。
