# Sample09：现代实车前视大雨场景

使用原始前视 `camera_front_wide` 数据、edge + RGB blur 控制和自动左右拼接。
[generate_heavy_rain.py](generate_heavy_rain.py) 是本案例入口，向
[共享天气迁移模块](../common/weather_transfer.py) 提供 sample ID、默认输出目录和
提示词目录。共享模块统一输入采样、RGB 打包、edge 提取、推理参数、日志和拼接逻辑。
仍需已有 Cosmos Framework、模型权重、FFmpeg、OpenCV 和 NumPy。
默认 720p、10 fps、93 帧（9.3 秒），GPU 0–3、
CP=4、50 steps、edge=1.0、blur=0.2/high、seed=2026。

## 提示词

- [正向模板](prompts/positive_prompt.json)
- [负向模板](prompts/negative_prompt.json)

大雨包含深色湿路面、浅水膜与局部浅积水、明显的局部镜面反射、清晰可见的细密雨线、
随距离增强的雨幕、轮胎接触区水花与后向水雾。根据用户选择，启用少量
镜头水滴、滑动或聚合及局部折射模糊，不生成雨刷或大面积积水。
保留车辆身份和运动、标线位置及颜色、交通标志、相机姿态和白色引擎盖形状尺寸。
这些是生成约束，不是像素级或几何一致性保证，必须检查结果。
镜头水滴设置是本次目标效果，不表示已确认真实相机的防雨或安装结构。
反射应对应原有车辆、护栏和亮天空，随路面透视延展，并被涟漪和粗糙纹理打散；
不将整条道路变成完美镜面，也不凭空开启车灯。雨线在深色背景前明显可见，
但避免粗白条、规则粒子网格和固定屏幕叠层。

## 运行

```bash
bash samples/09_cosmos3_hyundai_heavy_rain/run_heavy_rain.sh --prepare-only
bash samples/09_cosmos3_hyundai_heavy_rain/run_heavy_rain.sh

# 已有结果只补做拼接
bash samples/09_cosmos3_hyundai_heavy_rain/run_heavy_rain.sh --comparison-only
```

默认输出目录为 `/root/workspace/cosmos3/outputs/cosmos3_sample09_hyundai_front_heavy_rain`。
`inputs/` 保存源 RGB、edge、实际正负提示词、spec 和时间戳 manifest。
`sample09_hyundai_front_heavy_rain/vision.mp4` 是成片；
同目录 `comparison.mp4` 左侧源 RGB、右侧生成结果（720p 拼接为 2560×720）。
支持与 sample08 相同的分辨率、采样、控制权重及输出目录参数。
本案例提示词与结果独立维护，不修改 sample08 的雨量或提示词。
