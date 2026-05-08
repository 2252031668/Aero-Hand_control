#!/usr/bin/env python3
"""
Aero Hand 手势映射算法模块

提供从Mediapipe手部关键点到Aero Hand 7自由度的映射功能。
独立于GUI，便于单独调试和优化映射算法。

使用方法：
    mapper = HandGestureMapper()
    controls = mapper.map(landmarks)  # landmarks: 21个3D关键点
    # controls: {'thumb_abduction': 0.5, 'thumb_flex': 0.3, ...}
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class HandGestureMapper:
    """
    手势映射器 - 将Mediapipe手部关键点映射到Aero Hand的7个自由度
    
    输入：21个3D关键点坐标 [(x,y,z), ...]
    输出：7个自由度控制值 {name: value, ...}，范围0-1
    """
    
    def __init__(self, enable_smoothing: bool = True, smoothing_factor: float = 0.7):
        """
        初始化映射器
        
        Args:
            enable_smoothing: 是否启用平滑滤波
            smoothing_factor: 平滑因子 (0-1)，越大越平滑但响应越慢
        """
        self.enable_smoothing = enable_smoothing
        self.smoothing_factor = smoothing_factor
        
        # 上一帧的控制值（用于平滑）
        self.last_controls: Optional[Dict[str, float]] = None
        
        # 角度权重配置（可在外部调整）
        self.finger_angle_weights = {
            'angle0': 0.5,  # 与手掌面夹角权重
            'angle1': 0.5,  # MCP-PIP-DIP夹角权重
            'angle2': 0.5   # PIP-DIP-TIP夹角权重
            # 注意：总和不一定要为1，可以通过实验调整灵敏度
        }
        
        # 最大角度配置（用于归一化）
        self.max_finger_angle = 90.0  # 四指最大角度
        self.max_thumb_angle = 90.0   # 拇指最大角度
    
    def map(self, landmarks: List[Tuple[float, float, float]]) -> Optional[Dict[str, float]]:
        """
        主映射函数：将21个关键点转换为7个自由度控制值
        
        Args:
            landmarks: Mediapipe检测的21个3D关键点 [(x,y,z), ...]
            
        Returns:
            controls: 7个自由度控制值字典，如果检测失败返回None
            {
                'thumb_abduction': 0.0-1.0,
                'thumb_flex': 0.0-1.0,
                'thumb_tendon': 0.0-1.0,
                'index': 0.0-1.0,
                'middle': 0.0-1.0,
                'ring': 0.0-1.0,
                'pinky': 0.0-1.0
            }
        """
        if len(landmarks) != 21:
            return None
        
        # 计算手掌法向量
        palm_normal = self._calculate_palm_normal(landmarks)
        
        # === 计算大拇指3个自由度 ===
        _, thumb_abd = self._calculate_thumb_abduction(landmarks, palm_normal)
        _, thumb_flex = self._calculate_thumb_flexion(landmarks)
        _, thumb_tendon = self._calculate_thumb_tendon(landmarks)
        
        # === 计算四指弯曲度 ===
        # 食指 (5-MCP, 6-PIP, 7-DIP, 8-TIP)
        _, _, _, index_bend = self._calculate_finger_bend(
            np.array(landmarks[0]), np.array(landmarks[5]),
            np.array(landmarks[6]), np.array(landmarks[7]),
            np.array(landmarks[8]), palm_normal
        )
        
        # 中指 (9-MCP, 10-PIP, 11-DIP, 12-TIP)
        _, _, _, middle_bend = self._calculate_finger_bend(
            np.array(landmarks[0]), np.array(landmarks[9]),
            np.array(landmarks[10]), np.array(landmarks[11]),
            np.array(landmarks[12]), palm_normal
        )
        
        # 无名指 (13-MCP, 14-PIP, 15-DIP, 16-TIP)
        _, _, _, ring_bend = self._calculate_finger_bend(
            np.array(landmarks[0]), np.array(landmarks[13]),
            np.array(landmarks[14]), np.array(landmarks[15]),
            np.array(landmarks[16]), palm_normal
        )
        
        # 小指 (17-MCP, 18-PIP, 19-DIP, 20-TIP)
        _, _, _, pinky_bend = self._calculate_finger_bend(
            np.array(landmarks[0]), np.array(landmarks[17]),
            np.array(landmarks[18]), np.array(landmarks[19]),
            np.array(landmarks[20]), palm_normal
        )
        
        # 组装控制值
        controls = {
            'thumb_abduction': thumb_abd,
            'thumb_flex': thumb_flex,
            'thumb_tendon': thumb_tendon,
            'index_finger': index_bend,
            'middle_finger': middle_bend,
            'ring_finger': ring_bend,
            'pinky_finger': pinky_bend
        }
        
        # 应用平滑滤波
        if self.enable_smoothing:
            controls = self._apply_smoothing(controls)
        
        return controls
    
    def reset(self):
        """重置平滑滤波器状态"""
        self.last_controls = None
    
    # ==================== 内部计算方法 ====================
    
    def _calculate_3d_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        计算三个3D点形成的夹角（顶点在p2）
        如果角度>90度，取其补角，确保返回锐角
        
        Args:
            p1, p2, p3: 三个点的3D坐标 [x, y, z]
            
        Returns:
            angle_deg: 夹角（度数），范围0-90
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        # 如果角度>90度，取补角
        if angle_deg > 90.0:
            angle_deg = 180.0 - angle_deg
        
        return angle_deg
    
    def _calculate_palm_normal(self, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
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
        
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        
        if norm < 1e-8:
            return np.array([0.0, 0.0, 1.0])
        
        return normal / norm
    
    def _calculate_finger_bend(
        self,
        wrist: np.ndarray,
        mcp: np.ndarray,
        pip: np.ndarray,
        dip: np.ndarray,
        tip: np.ndarray,
        palm_normal: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        计算手指弯曲度
        
        Args:
            wrist, mcp, pip, dip, tip: 手指关键点坐标
            palm_normal: 手掌平面法向量
            
        Returns:
            angle0, angle1, angle2, normalized
        """
        # 角度0: 手指方向与手掌面的夹角
        finger_vec = pip - mcp
        finger_norm = np.linalg.norm(finger_vec)
        if finger_norm < 1e-8:
            angle0 = 0.0
        else:
            finger_unit = finger_vec / finger_norm
            cos_angle = np.dot(finger_unit, palm_normal)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_with_normal = np.degrees(np.arccos(cos_angle))
            angle0 = abs(90.0 - angle_with_normal)
        
        # 角度1: MCP-PIP-DIP
        angle1 = self._calculate_3d_angle(mcp, pip, dip)
        
        # 角度2: PIP-DIP-TIP
        angle2 = self._calculate_3d_angle(pip, dip, tip)
        
        # 加权平均
        w = self.finger_angle_weights
        avg_angle = angle0 * w['angle0'] + angle1 * w['angle1'] + angle2 * w['angle2']
        
        # 裁剪到0-max_angle
        clipped_angle = np.clip(avg_angle, 0, self.max_finger_angle)
        
        # 映射到0-1
        normalized = clipped_angle / self.max_finger_angle
        
        return angle0, angle1, angle2, normalized
    
    def _calculate_thumb_abduction(self, landmarks: List[Tuple[float, float, float]],
                                   palm_normal: np.ndarray) -> Tuple[float, float]:
        """
        计算拇指CMC外展程度
        使用(WRIST, CMC)线段与手掌面的夹角
        """
        wrist = np.array(landmarks[0])
        cmc = np.array(landmarks[1])
        
        thumb_vec = cmc - wrist
        thumb_norm = np.linalg.norm(thumb_vec)
        
        if thumb_norm < 1e-8:
            return 0.0, 0.0
        
        thumb_unit = thumb_vec / thumb_norm
        
        cos_angle = np.dot(thumb_unit, palm_normal)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_with_normal = np.degrees(np.arccos(cos_angle))
        
        angle = abs(90.0 - angle_with_normal)
        
        clipped = np.clip(angle, 0, self.max_thumb_angle)
        normalized = clipped / self.max_thumb_angle
        
        return angle, normalized
    
    def _calculate_thumb_flexion(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """
        计算拇指CMC屈曲程度
        使用(WRIST, CMC)线段与(CMC, MCP)线段的夹角
        """
        wrist = np.array(landmarks[0])
        cmc = np.array(landmarks[1])
        mcp = np.array(landmarks[2])
        
        angle = self._calculate_3d_angle(wrist, cmc, mcp)
        
        clipped = np.clip(angle, 0, self.max_thumb_angle)
        normalized = clipped / self.max_thumb_angle
        
        return angle, normalized
    
    def _calculate_thumb_tendon(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """
        计算拇指MCP/IP耦合弯曲
        平均两个关节夹角
        """
        cmc = np.array(landmarks[1])
        mcp = np.array(landmarks[2])
        ip = np.array(landmarks[3])
        tip = np.array(landmarks[4])
        
        angle1 = self._calculate_3d_angle(cmc, mcp, ip)
        angle2 = self._calculate_3d_angle(mcp, ip, tip)
        
        avg_bend = (angle1 + angle2) / 2.0
        
        clipped = np.clip(avg_bend, 0, self.max_thumb_angle)
        normalized = clipped / self.max_thumb_angle
        
        return avg_bend, normalized
    
    def _apply_smoothing(self, controls: Dict[str, float]) -> Dict[str, float]:
        """
        应用指数移动平均平滑滤波
        
        Args:
            controls: 当前帧的控制值
            
        Returns:
            smoothed_controls: 平滑后的控制值
        """
        if self.last_controls is None:
            self.last_controls = controls.copy()
            return controls
        
        smoothed = {}
        for key in controls:
            # 指数移动平均: new = factor * old + (1-factor) * current
            smoothed[key] = (self.smoothing_factor * self.last_controls[key] +
                           (1 - self.smoothing_factor) * controls[key])
        
        self.last_controls = smoothed.copy()
        return smoothed


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 简单测试
    print("HandGestureMapper 模块加载成功")
    print("使用方法:")
    print("  mapper = HandGestureMapper()")
    print("  controls = mapper.map(landmarks)")
