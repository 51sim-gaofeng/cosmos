# 共享天气迁移模块

[weather_transfer.py](weather_transfer.py) 是 sample08 和 sample09 的公共执行模块。
案例入口只提供 `sample_id`、案例目录、默认输出目录和命令说明；正负提示词仍放在
各自案例的 `prompts/` 中。

两个案例统一支持：

```text
--source-root
--output-dir
--framework-root
--model
--log-root
--resolution
--sample-step
--max-frames
--chunk-frames
--conditional-frames
--num-steps
--guidance
--control-guidance
--blur-weight
--blur-strength
--seed
--gpus
--dp-shard-size
--cp-size
--cfgp-size
--master-port
--prepare-only
--comparison-only
```

`--prepare-only` 和 `--comparison-only` 互斥。当前 `--source-root` 应包含 Hyundai
数据结构：

```text
distorted_cameras/intrinsic.json
distorted_cameras/images/camera_front_wide/*.png
```

共享模块统一代码和参数，但案例目录不再能脱离 `samples/common/` 单独复制运行。
应将 `samples/common/` 视为这两个案例的运行时依赖。
