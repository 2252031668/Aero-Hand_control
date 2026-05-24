# Aero Hand Gesture Control

<div align="center">

**基于Mediapipe的Aero Hand绳驱机器人手实时手势控制系统**


</div>

---


Aero Hand Gesture Control 是一个用于控制 TetherIA Aero Hand 绳驱机器人手的图形化界面程序。该系统通过摄像头实时捕捉用户手部动作，利用 Google Mediapipe 进行手部关键点检测，并将检测到的手势映射到机器人手的7个自由度，实现自然直观的手势控制。

### 核心特点

- 🎯 **实时手势识别**：基于Mediapipe的高精度手部追踪
- 🤖 **7自由度精确控制**：完整映射拇指3自由度 + 四指弯曲
- 🎨 **彩虹骨骼可视化**：实时显示手部骨架和关节点
- 🎮 **双模式控制**：支持手势控制和手动滑块控制
- 🔄 **平滑滤波**：指数移动平均算法减少抖动
- 💻 **友好GUI界面**：基于Tkinter的跨平台图形界面

## 📁 项目结构

```
Aero-Hand_control/
│
├── aero_open_sdk/                  # 核心SDK包
│   ├── __init__.py
│   ├── aero_hand.py               # Aero Hand串口通信控制
│   ├── aero_hand_constants.py     # 常量和参数定义
│   ├── gesture_mapper.py          # ⭐ 手势映射算法（重点）
│   ├── joints_to_actuations.py    # 关节到驱动器转换模型
│   ├── actuations_to_joints.py    # 驱动器到关节转换模型
│   └── gui.py                     # 原始GUI（不含手势控制）
│
├── examples/                       # 示例代码
│   ├── get_info.py                # 获取设备信息
│   ├── joint_control.py           # 关节控制示例
│   ├── torque_control.py          # 扭矩控制示例
│   ├── perform_homing.py          # Homing示例
│   └── ...
│
│
├── gesture_control_gui.py         # ⭐ 主GUI程序（含手势控制）
├── test_camera.py                 # 摄像头测试脚本
├── requirements.txt               # Python依赖清单
├── README.md                      # 本文档
└── LICENSE                        # Apache 2.0许可证
```
---

## ✨ 功能特性

### 主要功能

✅ **实时手势追踪与控制**
- Mediapipe手部21个关键点检测
- 彩虹色骨骼可视化显示
- 30FPS流畅视频流

✅ **7自由度精确映射**
- 拇指外展（CMC关节）
- 拇指屈曲（CMC关节）
- 拇指肌腱（MCP/IP耦合）
- 食指、中指、无名指、小指弯曲

✅ **智能平滑滤波**
- 指数移动平均算法
- 可调节平滑因子
- 有效减少手部抖动

✅ **双模式控制**
- 手势自动控制模式
- 手动滑块控制模式
- 无缝切换，互不干扰

✅ **完整串口控制**
- Homing归位操作
- Set ID设备编号设置
- Trim Servo舵机微调
- 速度/扭矩参数配置
- 实时状态读取（位置/速度/电流/温度）

✅ **多摄像头支持**
- 实时切换无需重启
- 自动检测和连接
---

## 💻 系统要求

### 操作系统
- ✅ Windows 10/11
- ✅ macOS 10.14+
- ✅ Linux (Ubuntu 18.04+)

### Python环境
- Python 3.10 或更高版本
- 推荐使用虚拟环境

### 硬件要求
- USB摄像头
- Aero Hand绳驱机器人手
- USB转串口适配器（如已集成则无需）
- 建议4GB以上内存

### 软件依赖
详见 [requirements.txt](requirements.txt)

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/2252031668/Aero-Hand_control.git
cd Aero-Hand_control
```

### 2. 创建虚拟环境（推荐）
pass

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行GUI程序

```bash
python gesture_control_gui.py
```
原版程序是 gui.py
```bash
python aero_open_sdk/gui.py
```

### 5. 开始使用

1. 📷 摄像头会自动启动，显示实时画面
2. 🔌 选择串口号，点击"Connect"连接机器人手
3. ✋ 勾选"Enable Gesture Mapping"启用手势控制

---

### 高级操作

#### Homing（归位）
- 点击"Homing"按钮
- 机器人手执行归位校准
- 完成后所有舵机会回到零点

#### Set ID（设置设备编号）
- 用于多手系统中的设备区分
- 输入新ID（0-253）
- 输入电流限制（0-1023）
- 点击确认后生效

#### Trim Servo（舵机微调）
- 对单个舵机进行零点校准
- 选择舵机ID（0-6）
- 输入微调角度（-360°到+360°）
- 用于修正机械装配误差

#### Set Speed/Torque（设置速度/扭矩）
- 配置单个舵机的运动参数
- Speed范围：0-32766
- Torque范围：0-1000
- 数值越大速度/力量越大

#### Upload Firmware（固件烧录）
1. 点击"Upload Firmware"
2. 选择.bin固件文件
3. 选择烧录端口
4. 等待烧录完成（约30秒）
5. 程序会自动重连

---

## 🧠 手势映射算法详解

这是本系统的核心算法部分，详细说明了如何将Mediapipe检测到的手部关键点映射到Aero Hand的7个自由度。

### 7自由度定义

Aero Hand采用绳驱欠驱动设计，包含7个可控自由度：

```
1. thumb_abduction  - 拇指外展（CMC关节）
                      控制拇指基部的开合角度
   
2. thumb_flex       - 拇指屈曲（CMC关节）
                      控制拇指的弯曲程度
   
3. thumb_tendon     - 拇指肌腱（MCP/IP耦合）
                      控制拇指末端的弯曲
   
4. index_finger     - 食指弯曲
                      控制食指的闭合程度
   
5. middle_finger    - 中指弯曲
                      控制中指的闭合程度
   
6. ring_finger      - 无名指弯曲
                      控制无名指的闭合程度
   
7. pinky_finger     - 小指弯曲
                      控制小指的闭合程度
```

### 映射算法原理

#### 1. 手掌法向量计算

首先计算手掌平面的法向量，作为后续角度计算的参考平面：

```python
# 使用手腕、食指MCP、小指MCP三点定义手掌平面
wrist = landmarks[0]
index_mcp = landmarks[5]
pinky_mcp = landmarks[17]

v1 = index_mcp - wrist
v2 = pinky_mcp - wrist
palm_normal = cross(v1, v2)  # 叉积得到法向量
```

#### 2. 拇指3个自由度计算

**thumb_abduction（外展）**
```python
# 计算(WRIST, CMC)线段与手掌面的夹角
thumb_vec = cmc - wrist
angle = abs(90° - angle_between(thumb_vec, palm_normal))
normalized = angle / max_thumb_angle
```

**thumb_flex（屈曲）**
```python
# 计算(WRIST, CMC)与(CMC, MCP)线段的夹角
angle = angle_3d(wrist, cmc, mcp)
normalized = angle / max_thumb_angle
```

**thumb_tendon（肌腱）**
```python
# 平均MCP和IP两个关节的弯曲角度
angle1 = angle_3d(cmc, mcp, ip)
angle2 = angle_3d(mcp, ip, tip)
avg_bend = (angle1 + angle2) / 2
normalized = avg_bend / max_thumb_angle
```

#### 3. 四指弯曲度计算

对于食指、中指、无名指、小指，采用**三角度加权平均**方法：

```python
# 角度0：手指方向与手掌面的夹角
finger_vec = pip - mcp
angle0 = abs(90° - angle_between(finger_vec, palm_normal))

# 角度1：MCP-PIP-DIP夹角
angle1 = angle_3d(mcp, pip, dip)

# 角度2：PIP-DIP-TIP夹角
angle2 = angle_3d(pip, dip, tip)

# 加权平均
avg_angle = angle0 * w0 + angle1 * w1 + angle2 * w2

# 归一化到0-1
normalized = clip(avg_angle, 0, max_angle) / max_angle
```

### 可调参数说明 ⭐⭐⭐

所有映射参数都在 `aero_open_sdk/gesture_mapper.py` 文件中定义，可以直接修改以优化控制效果。

#### 📍 参数位置

打开文件：`aero_open_sdk/gesture_mapper.py`

#### 1️⃣ 平滑滤波配置（第26行）

```python
def __init__(self, enable_smoothing: bool = True, smoothing_factor: float = 0.7):
```

**参数说明：**
- `enable_smoothing`: 是否启用平滑滤波
  - `True` = 启用（推荐，减少抖动）
  - `False` = 禁用（响应更快但可能抖动）

- `smoothing_factor`: 平滑因子（0-1）
  - `0.5` = 轻度平滑，响应快
  - `0.7` = 中度平滑（默认，平衡性能和稳定性）
  - `0.9` = 重度平滑，非常稳定但有延迟

**调整建议：**
```python
# 如果想要更快的响应速度
mapper = HandGestureMapper(enable_smoothing=True, smoothing_factor=0.5)

# 如果想要更稳定的控制（适合精细操作）
mapper = HandGestureMapper(enable_smoothing=True, smoothing_factor=0.85)
```

#### 2️⃣ 角度权重配置（第41-46行）

```python
self.finger_angle_weights = {
    'angle0': 0.5,  # 与手掌面夹角权重
    'angle1': 0.5,  # MCP-PIP-DIP夹角权重
    'angle2': 0.5   # PIP-DIP-TIP夹角权重
}
```

**参数说明：**
这三个权重决定了四指弯曲度计算中三个角度的重要性。

- `angle0`: 手指整体方向相对于手掌的角度
  - 增大此值会使控制更敏感于手指的整体抬起/放下
  
- `angle1`: 近端指间关节（MCP-PIP-DIP）的弯曲
  - 增大此值会更关注手指根部的弯曲
  
- `angle2`: 远端指间关节（PIP-DIP-TIP）的弯曲
  - 增大此值会更关注手指尖端的弯曲

**调整示例：**
```python
# 如果希望更关注手指根部弯曲（适合抓握动作）
self.finger_angle_weights = {
    'angle0': 0.3,
    'angle1': 0.6,  # 增大
    'angle2': 0.3
}

# 如果希望更关注手指尖端（适合精细捏取）
self.finger_angle_weights = {
    'angle0': 0.3,
    'angle1': 0.3,
    'angle2': 0.6   # 增大
}

# 如果希望对手指整体抬起更敏感
self.finger_angle_weights = {
    'angle0': 0.7,  # 增大
    'angle1': 0.2,
    'angle2': 0.2
}
```

💡 **注意**：权重总和不一定要等于1，可以通过实验调整灵敏度。

#### 3️⃣ 最大角度配置（第49-50行）

```python
self.max_finger_angle = 90.0  # 四指最大角度
self.max_thumb_angle = 90.0   # 拇指最大角度
```

**参数说明：**
这两个值定义了角度归一化的基准。

- 减小这些值会**提高灵敏度**（较小的实际角度就能达到最大值1.0）
- 增大这些值会**降低灵敏度**（需要更大的角度才能达到最大值）

**调整示例：**
```python
# 如果想让小幅度的手指弯曲就能完全闭合机器人手
self.max_finger_angle = 60.0  # 从90降到60，灵敏度提高50%

# 如果需要很大的弯曲才能达到最大值（适合大范围控制）
self.max_finger_angle = 120.0  # 从90升到120，灵敏度降低
```

#### 4️⃣ Mediapipe检测参数（第383-388行）

在 `gesture_control_gui.py` 中可以调整Mediapipe的检测参数：

```python
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,              # 最多检测几只手
    min_detection_confidence=0.7, # 检测置信度阈值
    min_tracking_confidence=0.5   # 追踪置信度阈值
)
```

**调整建议：**
```python
# 如果检测不稳定，提高置信度
min_detection_confidence=0.8
min_tracking_confidence=0.7

# 如果想检测双手（需要修改后续逻辑）
max_num_hands=2
```

---

## ⚙️ 高级配置

### 控制频率调整

在GUI顶部的"Rate (Hz)"输入框中可以调整控制指令发送频率：

- **默认值**：50 Hz（每秒50次）
- **推荐范围**：20-100 Hz
- **最低值**：1 Hz
- **最高值**：200 Hz

**调整建议：**
- 低频（20-30 Hz）：降低串口负载，适合低速操作
- 中频（50-80 Hz）：平衡性能和资源（推荐）
- 高频（100+ Hz）：更流畅的控制，但增加串口负担

### 串口波特率配置

支持的波特率：
**921600**（默认推荐）

**注意**：必须与Aero Hand固件配置的波特率一致。

### 摄像头分辨率

当前固定为640x480，如需修改可在 `_auto_start_camera()` 和 `_start_camera_by_id()` 方法中调整：

```python
self.cv2_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 改为高清
self.cv2_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### 帧率控制

视频捕获循环中设置了30FPS的延迟：

```python
time.sleep(0.03)  # ~30 FPS
```







<div align="center">

**Made with ❤️ for Robotics Community**

⭐ 如果这个项目对你有帮助，请给我们一个Star！

</div>
