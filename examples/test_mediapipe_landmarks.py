#!/usr/bin/env python3
"""
Mediapipe手部关键点数据采集脚本
用于测试和验证手势到Aero Hand的7自由度映射算法

功能：
- 实时检测手部21个关键点
- 计算7个自由度的控制值（0-1范围）
- 显示原始角度值用于调试
- 彩虹骨骼可视化
"""

import cv2
import numpy as np
from typing import List, Tuple

from mediapipe.python.solutions import hands, drawing_utils, drawing_styles


# ==================== 核心映射算法 ====================

def calculate_3d_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三个3D点形成的夹角（顶点在p2）
    如果角度>90度，取其补角（180-angle），确保返回锐角
    
    Args:
        p1, p2, p3: 三个点的3D坐标 [x, y, z]
        
    Returns:
        angle_deg: 夹角（度数），范围0-90
    """
    v1 = p1 - p2  # p2 -> p1
    v2 = p3 - p2  # p2 -> p3
    
    # 计算余弦值
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 转换为角度
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # 如果角度>90度，取补角（因为我们关心的是弯曲程度，不是几何夹角）
    if angle_deg > 90.0:
        angle_deg = 180.0 - angle_deg
    
    return angle_deg


def calculate_palm_normal(landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    计算手掌平面的法向量
    使用手腕、食指MCP、小指MCP三点定义手掌平面
    
    Args:
        landmarks: 21个关键点列表
        
    Returns:
        normal: 单位法向量 [x, y, z]
    """
    wrist = np.array(landmarks[0])
    index_mcp = np.array(landmarks[5])
    pinky_mcp = np.array(landmarks[17])
    
    # 计算两个向量
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    
    # 计算叉积得到法向量
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    
    if norm < 1e-8:
        # 如果法向量太小，返回默认值
        return np.array([0.0, 0.0, 1.0])
    
    return normal / norm


def calculate_finger_bend_with_palm(
    wrist: np.ndarray, 
    mcp: np.ndarray, 
    pip: np.ndarray, 
    dip: np.ndarray, 
    tip: np.ndarray,
    palm_normal: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    计算手指弯曲度（方案A：使用手掌法向量）
    
    Args:
        wrist, mcp, pip, dip, tip: 手指关键点坐标
        palm_normal: 手掌平面法向量
        
    Returns:
        angle0: 手指与手掌面的夹角
        angle1: MCP-PIP-DIP夹角
        angle2: PIP-DIP-TIP夹角
        normalized: 归一化后的弯曲度（0-1）
    """
    # 角度0: 手指方向与手掌面的夹角
    finger_vec = pip - mcp
    finger_norm = np.linalg.norm(finger_vec)
    if finger_norm < 1e-8:
        angle0 = 0.0
    else:
        finger_unit = finger_vec / finger_norm
        # 计算手指向量与法向量的夹角
        cos_angle = np.dot(finger_unit, palm_normal)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_with_normal = np.degrees(np.arccos(cos_angle))
        # 转换为与平面的夹角（90度减去与法向量的夹角）
        angle0 = abs(90.0 - angle_with_normal)
    
    # 角度1: MCP-PIP-DIP
    angle1 = calculate_3d_angle(mcp, pip, dip)
    
    # 角度2: PIP-DIP-TIP
    angle2 = calculate_3d_angle(pip, dip, tip)
    
    # 赋权
    avg_angle = angle0*0.5 + angle1*0.5 + angle2*0.5  #这里合计不是1 因为实际测试不够灵敏
    
    # 裁剪到0-90度
    clipped_angle = np.clip(avg_angle, 0, 90)
    
    # 映射到0-1
    normalized = clipped_angle / 90.0
    
    return angle0, angle1, angle2, normalized


def calculate_thumb_abduction(landmarks: List[Tuple[float, float, float]], 
                              palm_normal: np.ndarray) -> Tuple[float, float]:
    """
    计算拇指CMC外展程度
    使用(WRIST, CMC)线段与手掌面的夹角
    
    Returns:
        angle: 原始角度值
        normalized: 归一化到0-1（基于0-90度）
    """
    wrist = np.array(landmarks[0])
    cmc = np.array(landmarks[1])
    
    # 计算WRIST->CMC向量
    thumb_vec = cmc - wrist
    thumb_norm = np.linalg.norm(thumb_vec)
    
    if thumb_norm < 1e-8:
        return 0.0, 0.0
    
    thumb_unit = thumb_vec / thumb_norm
    
    # 计算与法向量的夹角
    cos_angle = np.dot(thumb_unit, palm_normal)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_with_normal = np.degrees(np.arccos(cos_angle))
    
    # 转换为与手掌平面的夹角
    angle = abs(90.0 - angle_with_normal)
    
    # 裁剪到0-90度
    clipped = np.clip(angle, 0, 90)
    
    # 映射到0-1
    normalized = clipped / 90.0
    
    return angle, normalized


def calculate_thumb_flexion(landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """
    计算拇指CMC屈曲程度
    使用(WRIST, CMC)线段与(CMC, MCP)线段的夹角
    
    Returns:
        angle: 原始角度值（已转换为锐角）
        normalized: 归一化到0-1（基于0-90度）
    """
    wrist = np.array(landmarks[0])
    cmc = np.array(landmarks[1])
    mcp = np.array(landmarks[2])
    
    # 使用calculate_3d_angle函数，它会自动处理钝角转锐角
    angle = calculate_3d_angle(wrist, cmc, mcp)
    
    # 裁剪到0-90度
    clipped = np.clip(angle, 0, 90)
    
    # 映射到0-1
    normalized = clipped / 90.0
    
    return angle, normalized


def calculate_thumb_tendon(landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """
    计算拇指MCP/IP耦合弯曲
    平均(CMC,MCP)-(MCP,IP)和(MCP,IP)-(IP,TIP)两个夹角
    
    Returns:
        avg_bend: 平均弯曲角度
        normalized: 归一化到0-1（基于0-90度）
    """
    cmc = np.array(landmarks[1])
    mcp = np.array(landmarks[2])
    ip = np.array(landmarks[3])
    tip = np.array(landmarks[4])
    
    # 第一个夹角：(CMC,MCP)与(MCP,IP)
    angle1 = calculate_3d_angle(cmc, mcp, ip)
    
    # 第二个夹角：(MCP,IP)与(IP,TIP)
    angle2 = calculate_3d_angle(mcp, ip, tip)
    
    # 取平均
    avg_bend = (angle1 + angle2) / 2.0
    
    # 裁剪到0-90度
    clipped = np.clip(avg_bend, 0, 90)
    
    # 映射到0-1
    normalized = clipped / 90.0
    
    return avg_bend, normalized


def map_hand_to_aero(landmarks: List[Tuple[float, float, float]]) -> dict:
    """
    将Mediapipe手部关键点映射到Aero Hand的7个自由度
    
    Args:
        landmarks: 21个关键点列表 [(x,y,z), ...]
        
    Returns:
        result: 包含7个自由度控制值和调试信息的字典
    """
    if len(landmarks) != 21:
        return None
    
    # 计算手掌法向量
    palm_normal = calculate_palm_normal(landmarks)
    
    # === 大拇指3个自由度 ===
    thumb_abd_angle, thumb_abd = calculate_thumb_abduction(landmarks, palm_normal)
    thumb_flex_angle, thumb_flex = calculate_thumb_flexion(landmarks)
    thumb_tendon_angle, thumb_tendon = calculate_thumb_tendon(landmarks)
    
    # === 四指弯曲度 ===
    # 食指 (5-MCP, 6-PIP, 7-DIP, 8-TIP)
    idx_angles = calculate_finger_bend_with_palm(
        np.array(landmarks[0]), np.array(landmarks[5]),
        np.array(landmarks[6]), np.array(landmarks[7]),
        np.array(landmarks[8]), palm_normal
    )
    
    # 中指 (9-MCP, 10-PIP, 11-DIP, 12-TIP)
    mid_angles = calculate_finger_bend_with_palm(
        np.array(landmarks[0]), np.array(landmarks[9]),
        np.array(landmarks[10]), np.array(landmarks[11]),
        np.array(landmarks[12]), palm_normal
    )
    
    # 无名指 (13-MCP, 14-PIP, 15-DIP, 16-TIP)
    ring_angles = calculate_finger_bend_with_palm(
        np.array(landmarks[0]), np.array(landmarks[13]),
        np.array(landmarks[14]), np.array(landmarks[15]),
        np.array(landmarks[16]), palm_normal
    )
    
    # 小指 (17-MCP, 18-PIP, 19-DIP, 20-TIP)
    pinky_angles = calculate_finger_bend_with_palm(
        np.array(landmarks[0]), np.array(landmarks[17]),
        np.array(landmarks[18]), np.array(landmarks[19]),
        np.array(landmarks[20]), palm_normal
    )
    
    return {
        # 7个自由度控制值（0-1）
        'controls': {
            'thumb_abduction': thumb_abd,
            'thumb_flex': thumb_flex,
            'thumb_tendon': thumb_tendon,
            'index': idx_angles[3],
            'middle': mid_angles[3],
            'ring': ring_angles[3],
            'pinky': pinky_angles[3]
        },
        # 调试信息（原始角度）
        'debug': {
            'thumb_abd_angle': thumb_abd_angle,
            'thumb_flex_angle': thumb_flex_angle,
            'thumb_tendon_angle': thumb_tendon_angle,
            'index_angles': idx_angles[:3],  # (angle0, angle1, angle2)
            'middle_angles': mid_angles[:3],
            'ring_angles': ring_angles[:3],
            'pinky_angles': pinky_angles[:3]
        }
    }


class HandPoseTester:
    """手部姿态测试器 - 用于验证映射算法"""
    
    def __init__(self, camera_id=0):
        # Mediapipe Hands初始化
        self.mp_hands = hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = drawing_utils
        self.mp_drawing_styles = drawing_styles
        
        # 摄像头
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("=" * 80)
        print("Aero Hand 手势映射算法测试工具")
        print("=" * 80)
        print("\n使用说明:")
        print("  - 将手放在摄像头前，确保完整可见")
        print("  - 观察控制台输出的7个自由度控制值（0-1范围）")
        print("  - 同时显示原始角度值用于调试")
        print("  - 尝试各种手势，验证数值范围是否合理")
        print("  - 按 'q' 键退出程序")
        print("\n7个自由度说明:")
        print("  1. thumb_abduction - 拇指外展（0=外展, 1=收拢）")
        print("  2. thumb_flex      - 拇指屈曲（0=伸直, 1=最大屈曲）")
        print("  3. thumb_tendon    - 拇指弯曲（0=伸直, 1=握拳）")
        print("  4. index           - 食指弯曲（0=伸直, 1=握拳）")
        print("  5. middle          - 中指弯曲（0=伸直, 1=握拳）")
        print("  6. ring            - 无名指弯曲（0=伸直, 1=握拳）")
        print("  7. pinky           - 小指弯曲（0=伸直, 1=握拳）")
        print("=" * 80)
    
    def draw_rainbow_skeleton(self, image, landmarks):
        """
        绘制彩虹骨骼可视化
        
        Args:
            image: OpenCV图像
            landmarks: Mediapipe关键点列表
        """
        h, w, _ = image.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        
        # 定义五指连接关系和颜色（BGR格式）
        fingers = [
            ([1, 2, 3, 4], (0, 255, 255)),      # 拇指 - 黄色
            ([5, 6, 7, 8], (128, 0, 128)),       # 食指 - 紫色
            ([9, 10, 11, 12], (255, 255, 0)),    # 中指 - 青色
            ([13, 14, 15, 16], (0, 255, 0)),     # 无名指 - 绿色
            ([17, 18, 19, 20], (0, 0, 255))      # 小指 - 红色
        ]
        
        # 绘制所有关节点（白色圆点）
        for x, y in points:
            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
        
        # 绘制彩色骨骼线
        for finger_indices, color in fingers:
            for i in range(len(finger_indices) - 1):
                start_idx = finger_indices[i]
                end_idx = finger_indices[i + 1]
                cv2.line(image, points[start_idx], points[end_idx], color, 2)
            
            # 连接到手腕（0号点）
            cv2.line(image, points[0], points[finger_indices[0]], color, 2)
    
    def print_mapping_results(self, result: dict, frame_count: int):
        """
        打印映射结果和调试信息
        
        Args:
            result: map_hand_to_aero返回的结果字典
            frame_count: 帧编号
        """
        if result is None:
            return
        
        controls = result['controls']
        debug = result['debug']
        
        # 清屏（使用ANSI转义码）
        print('\033[2J\033[H', end='')
        
        print("=" * 80)
        print(f"Aero Hand 手势映射实时数据 | 帧: {frame_count}")
        print("=" * 80)
        
        print("\n【7个自由度控制值】(0=张开, 1=握拳)")
        print("-" * 80)
        print(f"  1. thumb_abduction : {controls['thumb_abduction']:.3f}  (拇指外展)")
        print(f"  2. thumb_flex      : {controls['thumb_flex']:.3f}  (拇指屈曲)")
        print(f"  3. thumb_tendon    : {controls['thumb_tendon']:.3f}  (拇指弯曲)")
        print(f"  4. index           : {controls['index']:.3f}  (食指)")
        print(f"  5. middle          : {controls['middle']:.3f}  (中指)")
        print(f"  6. ring            : {controls['ring']:.3f}  (无名指)")
        print(f"  7. pinky           : {controls['pinky']:.3f}  (小指)")
        
        print("\n【调试信息 - 原始角度】")
        print("-" * 80)
        print(f"  拇指外展角度: {debug['thumb_abd_angle']:6.2f}°  (范围: 0-90°)")
        print(f"  拇指屈曲角度: {debug['thumb_flex_angle']:6.2f}°  (范围: 0-90°)")
        print(f"  拇指弯曲角度: {debug['thumb_tendon_angle']:6.2f}°  (范围: 0-90°)")
        
        print(f"\n  食指角度: {debug['index_angles'][0]:5.1f}°, {debug['index_angles'][1]:5.1f}°, {debug['index_angles'][2]:5.1f}°  (平均={np.mean(debug['index_angles']):5.1f}°)")
        print(f"  中指角度: {debug['middle_angles'][0]:5.1f}°, {debug['middle_angles'][1]:5.1f}°, {debug['middle_angles'][2]:5.1f}°  (平均={np.mean(debug['middle_angles']):5.1f}°)")
        print(f"  无名指角度: {debug['ring_angles'][0]:5.1f}°, {debug['ring_angles'][1]:5.1f}°, {debug['ring_angles'][2]:5.1f}°  (平均={np.mean(debug['ring_angles']):5.1f}°)")
        print(f"  小指角度: {debug['pinky_angles'][0]:5.1f}°, {debug['pinky_angles'][1]:5.1f}°, {debug['pinky_angles'][2]:5.1f}°  (平均={np.mean(debug['pinky_angles']):5.1f}°)")
        
        print("\n" + "=" * 80)
        print("提示: 按 'q' 键退出程序")
        print("=" * 80)
    
    def run(self):
        """主运行循环"""
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                
                frame_count += 1
                
                # 转换颜色空间
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Mediapipe检测
                results = self.hands.process(rgb_frame)
                
                # 绘制检测结果
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 绘制彩虹骨骼
                        self.draw_rainbow_skeleton(frame, hand_landmarks.landmark)
                        
                        # 提取关键点坐标
                        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        
                        # 计算映射
                        result = map_hand_to_aero(landmarks)
                        
                        # 在画面上显示简要信息
                        if result:
                            controls = result['controls']
                            y_offset = 30
                            cv2.putText(frame, f"Thumb Abd: {controls['thumb_abduction']:.2f}", 
                                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(frame, f"Thumb Flex: {controls['thumb_flex']:.2f}", 
                                      (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(frame, f"Index: {controls['index']:.2f}", 
                                      (10, y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(frame, f"Middle: {controls['middle']:.2f}", 
                                      (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(frame, f"Ring: {controls['ring']:.2f}", 
                                      (10, y_offset + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.putText(frame, f"Pinky: {controls['pinky']:.2f}", 
                                      (10, y_offset + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # 打印到控制台（每10帧打印一次，避免刷屏）
                        if frame_count % 10 == 0:
                            self.print_mapping_results(result, frame_count)
                else:
                    cv2.putText(frame, "No hand detected", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示画面
                cv2.imshow('Aero Hand Gesture Mapping Test', frame)
                
                # 键盘输入处理
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n退出程序...")
                    break
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("\n资源已释放")


def main():
    try:
        tester = HandPoseTester(camera_id=1)  # 可以修改camera_id选择不同摄像头
        tester.run()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
