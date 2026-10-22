# 多视角驾驶视频生成后训练与微调方案

## 1. 目标与边界

目标是在任意数量 `N >= 2` 的同步车载相机视角上生成可控的场景状态泛化视频。目标状态可为
天气、时间段（昼夜、晨昏）、太阳高度/方向、季节、道路状态或交通环境，同时增强：

- **跨视角一致性**：同一车辆、道路、标线和静态设施在相邻相机重叠区域中保持几何对应；
- **跨帧一致性**：自车运动、车辆轨迹、光照、反射、动态环境效应在时间上连续；
- **目标状态真实性**：目标天气、昼夜光照、阴影、路面状态和大气效果自然可信。

本方案不把现有的单视角 Transfer 当作严格多视角模型。当前
[Sample10](../samples/10_cosmos3_five_view_heavy_rain/README.md) 通过同步帧、相同
prompt/seed 与 edge + blur 控制实现五相机基线的风格和时间轴一致，但不消费相机标定或自车轨迹。

## 2. 推荐技术路线

### 路线 A：优先评估 Cosmos-Transfer2.5 Auto Multiview

[Cosmos-Transfer2.5 Auto Multiview](https://docs.nvidia.com/cosmos/latest/transfer2.5/auto-multiview_inference.html)
是最接近目标的公开推理路线。它要求 8 张 GPU，并以物体位置、相机标定和车辆轨迹渲染出的
world scenario 控制视频进行联合多视角生成。

适用条件：

- 可得到或构建每帧 3D 场景标注、HD Map、动态物体轨迹或 LiDAR；
- 可用相机内参、`ego_to_camera` 外参和 ego pose 生成几何控制视频；
- 接受将当前实现迁移到独立的 Transfer2.5 环境与模型。

此路线不要求将所有原始 RGB 直接作为联合输入；几何控制是跨视角一致性的主条件。

### 路线 B：Cosmos3 定制后训练

在 Cosmos3 主干上实现标定感知多视角 adapter，适合必须保留现有真实多相机 MP4
state-transfer 流程的场景。需要自行增加可变数量的视角输入 schema、位姿编码、投影注意力与训练代码。

[Cosmos3-Nano-Sim-Transfer](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Transfer)
可作为 sim-to-real 外观生成参考：它根据 depth、segmentation、edge、blur 等仿真控制生成
真实感 RGB，但模型卡没有将其声明为带相机标定输入的联合多视角模型。

## 3. 数据契约

每个训练样本是一段按时间对齐的多相机序列。令视角集合为
`V = {camera_1, ..., camera_N}`；每个样本可包含任意固定数量的已标定相机：

```text
sample_id/
  cameras/
    <camera_name_1>.mp4
    ...
    <camera_name_N>.mp4
  calibration/
    intrinsic.json
    extrinsic.json
  ego_pose.jsonl
  timestamps.json
  controls/                  # 可选：depth、seg、HD Map、3D bbox、LiDAR
  target_state/              # 可选：配对目标状态 RGB
```

硬性校验：

1. 所有视角使用同一时间原点，并裁到共同最短长度；
2. 每个样本保存每帧 timestamp、ego pose 和相机内外参版本；
3. 标定需说明坐标系、长度单位、外参方向及图像是否已去畸变；
4. 所有相机输入统一到模型的目标分辨率，但保留原始尺寸与内参缩放关系；
5. 按路线/地点/日期分割 train、validation、test，禁止相邻帧或同一路段泄漏到不同集合。

### 3.1 是否必须源状态—目标状态配对

严格配对不是启动几何后训练的前提，但它是训练内容保持式状态迁移的最高价值数据。源状态可为
晴天/白昼，目标状态可为大雨、夜晚、晨昏、不同季节或其他受控环境状态。

| 数据 | 用途 | 配对要求 |
| --- | --- | --- |
| 同步源状态多视角 + 标定 + ego pose | 多视角和时序几何预训练 | 不要求目标状态配对 |
| 同步目标状态多视角 + 标定 + ego pose | 目标状态外观域适配 | 不要求源状态配对 |
| 同路线、同 rig、源状态到目标状态配对 | 内容保持、状态迁移和最终验证 | 强烈推荐 |
| 仿真源状态/目标状态严格配对 | 重投影、深度、遮挡和轨迹监督 | 推荐 |

真实配对采集时，交通参与者会变化；必须用动态物体 mask、可见性 mask 或 3D 标注排除不可对应区域，
不能将所有像素当作晴雨一一对应的真值。

## 4. 模型设计

### 4.1 位姿与时间编码

对相机 `i` 和帧 `t`，输入：

```text
K_i                         # 相机内参
T_ego_to_camera_i           # 固定相机外参
T_world_to_ego_t            # 每帧自车位姿
timestamp_t
```

将相机 ID、归一化内参、相对外参和相对 ego motion 编码为 token 或 attention bias。训练和推理必须使用
完全一致的坐标系定义。

### 4.2 几何感知 cross-attention

不要对所有视角的完整图像 token 做无约束全连接 attention。推荐使用投影/极线约束：

1. 使用深度、LiDAR、3D bbox 或预测深度，将相机 A 的 token 反投影到 ego/world 坐标；
2. 利用 B 的 `K_B` 与外参投影到相机 B；
3. 仅对重投影邻域建立 cross-attention；
4. 用视锥、深度竞争和动态遮挡 mask 过滤不可见 token；
5. 在无重叠区域回退到每视角本地 attention。

共享的应是场景和目标状态的高层 latent；不应强制不同相机具有相同屏幕坐标的局部外观细节。

### 4.3 Ego-motion temporal attention

将 `t-k` 帧的静态场景特征通过：

```text
T_ego(t-k)_to_ego(t)
```

warp 到时刻 `t` 的参考坐标，再与当前特征做 temporal attention。动态车辆应使用 3D track/scene flow
或动态 mask，避免把自车位姿补偿错误施加给独立运动物体。

长视频继续使用 121 帧级别 chunk，但训练应在相邻 chunk 边界加入 overlap/历史条件和边界一致性损失。

## 5. 后训练策略

### 阶段 0：无训练基线

保留 Sample10 输出，记录每路：

- 原始 RGB、edge、blur control；
- 600 帧 / 20 FPS 输出；
- 每个 chunk 的耗时与随机种子；
- 重叠视场中的人工质量评分。

这组结果是所有后训练实验的对照组。

### 阶段 1：几何 adapter 预训练

使用 CARLA、Omniverse 或带完整标定/位姿的多视角数据，仅训练新增模块：

- 相机/ego pose encoder；
- 几何 cross-attention adapter；
- temporal motion adapter；
- 可选 LoRA。

冻结大部分 Cosmos 主干，避免将模型整体外观偏向仿真域。优先使用深度、分割、HD Map、3D bbox、
光流和可见性作为几何监督。

### 阶段 2：真实多视角域适配

使用真实同步多相机序列，继续训练 adapter/LoRA：

- 保持跨相机投影一致；
- 使用真实传感器的曝光、噪声、压缩和运动模式；
- 不要求严格源状态/目标状态配对，但不将不同交通状态误作像素监督。

### 阶段 3：目标状态外观适配

使用真实目标状态数据和可控 prompt，微调目标外观：

- 天气：雨线、雾、积水、路面粗糙度和大气消光；
- 时间：太阳方向、阴影、色温、曝光和人工光源；
- 季节/环境：植被、地表状态和背景大气；
- 目标状态下的动态效应，例如水花、灯光反射或风致运动。

如果有可靠源状态/目标状态配对，仅在静态且可重投影的区域增加配对重建损失；动态物体和遮挡区域使用 mask。

### 阶段 4：小规模目标 rig 校准

使用目标 rig 的真实数据做低学习率 adapter/LoRA 微调和验证。冻结主干与大部分视觉编码器，
只更新新增 attention、位姿编码和少量状态 adapter 参数。

## 6. 损失函数

总损失可写为：

```text
L = L_diffusion
  + lambda_mv * L_multiview_reprojection
  + lambda_temp * L_ego_motion_temporal
  + lambda_control * L_control_fidelity
  + lambda_state * L_state_appearance
  + lambda_boundary * L_chunk_boundary
```

建议含义：

- `L_diffusion`：每视角标准扩散/velocity 训练目标；
- `L_multiview_reprojection`：投影到相邻相机重叠区后的 feature/color/depth 一致性，只在有效可见区域计算；
- `L_ego_motion_temporal`：ego-motion warp 后的静态背景特征一致性；
- `L_control_fidelity`：输出与 depth、seg、edge、HD Map 或 bbox 控制相符；
- `L_state_appearance`：目标状态的感知/风格约束，不应破坏结构；
- `L_chunk_boundary`：相邻生成 chunk 在 overlap 区域的一致性。

初始训练应让 `lambda_mv` 和 `lambda_temp` 从小权重开始逐步升高。投影、深度或时间戳错误会导致
几何损失错误优化，必须先完成标定与轨迹验证。

## 7. 评估与准入门槛

### 7.1 自动指标

- **相机重投影一致性**：将深度支持的像素投影到相邻视角，计算有效重叠区的 feature/edge 差异；
- **ego-motion 时序一致性**：对静态区域做 pose warp 后计算 LPIPS、特征差异或 edge IoU；
- **控制保真度**：segmentation/HD Map/bbox/depth 与生成 RGB 中结构的匹配；
- **目标状态真实性**：对应目标状态评估集上的视觉质量与人工评分，例如天气效应、昼夜光照、阴影和路面状态；
- **chunk 边界**：每个 121 帧边界前后 N 帧的 flicker、轨迹和曝光突变指标。

### 7.2 人工审查

至少审查：

1. 前广角与前左右相机的重叠道路区域；
2. 后左右相机中的同一动态车辆；
3. 车道线、路缘、护栏等静态背景；
4. 与目标状态相关的局部外观，例如水花、反射、灯光或阴影；
5. 600 帧序列及所有 chunk 边界。

模型只能在相对对应任务基线同时提升几何一致性和时间稳定性，且不降低目标状态外观评分时，才进入
下一阶段或部署候选。

## 8. 资源与实施顺序

1. **数据审计**：验证标定、时间戳、ego pose、图像去畸变和重叠视场；
2. **世界控制生成**：从 3D/HD Map/LiDAR/轨迹渲染每视角 depth、seg、HD Map 或 bbox 视频；
3. **Auto Multiview 评估**：若具备 Transfer2.5 环境和权重，先用其 8 GPU 流程做联合推理基线；
4. **adapter 原型**：仅训练新增几何与时序模块，短序列验证正确性；
5. **真实域适配**：加入真实多视角源状态/目标状态数据；
6. **长序列验证**：以目标帧率和时长，对全部视角做同步评估；
7. **版本冻结**：记录模型 hash、数据版本、标定版本、prompt、seed 和全部评估报告。

## 9. 当前缺口与下一步

当前五路 MP4、内外参和 Sample10 可作为一个 `N=5` 的无训练基线，但尚缺：

- 每帧 ego pose；
- 可渲染的 HD Map、3D bbox、LiDAR 或可靠深度；
- 明确的相机时间同步与去畸变说明；
- 真实同步多视角目标状态序列或源状态/目标状态配对样本；
- Cosmos-Transfer2.5 Auto Multiview 的独立运行环境与权重。

在补齐上述任一训练栈之前，不应直接在当前 Nano 模型中插入未训练的 cross-attention 层。
