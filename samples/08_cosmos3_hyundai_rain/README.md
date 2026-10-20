# Sample08：现代实车前视雨天生成

输入为 `samples/data/hyundai/seed-5v/distorted_cameras/images/camera_front_wide`
中的时间戳 PNG。相机标定为非鱼眼 3840×2160，数据共 606 帧，时间戳间隔
50 ms，即 20 fps。脚本验证标定和帧间隔，不修改原始图像，不额外去畸变。

默认每隔一帧采样，取前 93 帧，以 1280×720、10 fps 生成 9.3 秒中雨预览。
保留原始视角、车辆、道路和画面底部已有的白色引擎盖，添加持续中雨、湿路面、
适度反射与局部轮胎水雾。雨滴更频繁可见，但保持细小、短且低对比度，
避免长亮雨线、密集雨幕和全画面叠加雨丝；
不添加镜头水滴、雨刷或积雪。

[generate_rain.py](generate_rain.py) 是本案例入口，调用
[共享天气迁移模块](../common/weather_transfer.py) 提供统一的视频打包、边缘提取、
多卡启动和左右拼接逻辑。正、负提示词模板仍由本案例独立维护：

- [正向提示词](prompts/positive_prompt.json)
- [负向提示词](prompts/negative_prompt.json)

直接编辑以上 JSON 即可调整后续生成，不需要修改 Python 或共享提示词文件。
正向模板的 `temporal_caption` 支持 `{duration}` 占位符，脚本自动填入秒数，
并根据运行参数设置 `resolution`、`duration` 和 `fps`。
准备时将实际使用的提示词保存为输出目录的 `inputs/rain_prompt.json` 和
`inputs/rain_negative_prompt.json`；仅修改输出副本会在下次准备时被覆盖。
默认采用预计算 edge + 原始 RGB blur 双控制。edge 权重仍为 1.0，
通过 `vision_path` 传入与 edge 同步的源 RGB 视频，框架在线提取 blur 条件，
参考原视频的车辆颜色、场景色彩和低频外观，不使用雪天色调处理。
默认 `--blur-weight 0.2 --blur-strength high`；这是初始调试值，并非已验证的最优值。
blur strength 仅控制条件图的模糊程度，不是将最终视频模糊。
较高的 blur 权重可能改善颜色保持，也可能限制湿路面和天气变化，需要实测。
使用 `--blur-weight 0` 恢复此前 edge-only 模式，此时不传入 `vision_path`。
双控制仍不能保证外观逐像素不变。

```bash
# 仅准备和校验输入
bash samples/08_cosmos3_hyundai_rain/run_rain.sh --prepare-only

# 生成预览；默认 GPU 0–3、CP=4、50 steps
bash samples/08_cosmos3_hyundai_rain/run_rain.sh

# 对已有结果补做拼接，不重新准备输入或运行模型
bash samples/08_cosmos3_hyundai_rain/run_rain.sh \
  --comparison-only \
  --output-dir /root/workspace/cosmos3/outputs/cosmos3_sample08_hyundai_front_rain_720p_moderate_blur02_v1

# 调整原视频外观参考权重，保存到独立目录
bash samples/08_cosmos3_hyundai_rain/run_rain.sh \
  --blur-weight 0.2 --blur-strength high \
  --output-dir /root/workspace/cosmos3/outputs/cosmos3_sample08_hyundai_front_rain_edge_blur

# 确认预览效果后，生成完整采样序列（303 帧，30.3 秒）
bash samples/08_cosmos3_hyundai_rain/run_rain.sh \
  --max-frames 303 \
  --output-dir /root/workspace/cosmos3/outputs/cosmos3_sample08_hyundai_front_rain_full
```

默认输出：
`/root/workspace/cosmos3/outputs/cosmos3_sample08_hyundai_front_rain`。
其 `inputs/` 包含 RGB/edge 视频、雨天正负提示词、推理 spec 和时间戳 manifest。
推理日志由共享启动器保存至 `/root/workspace/cosmos3/outputs/run_logs`。
需要已有 Cosmos Framework 虚拟环境、模型权重、FFmpeg、OpenCV 和 NumPy。

推理成功后自动生成 `sample08_hyundai_front_rain/comparison.mp4`，
左侧为实际采样、缩放后的原始 RGB，右侧为生成视频，保持两侧各自分辨率。
例如 720p 双画面为 2560×720，时长和帧率不变，无音轨。
拼接前检查两路尺寸、帧率、帧数一致，不通过截短或补帧掩盖不匹配。
`--prepare-only` 不生成对照视频；`--comparison-only` 只处理已有视频。
