import numpy as np
from scipy import signal
from config import get_config

class AudioEffects:
    def __init__(self):
        self.config = get_config()
        self.sample_rate = self.config.SAMPLE_RATE
        
    def apply_effects(self, audio, scene_type, emotion):
        """应用音频效果"""
        # 基础效果
        audio = self._normalize(audio)
        audio = self._add_reverb(audio, scene_type)
        
        # 场景特定效果
        if scene_type == '战斗':
            audio = self._add_distortion(audio)
            audio = self._add_compression(audio)
        elif scene_type == '神秘':
            audio = self._add_echo(audio)
            audio = self._add_chorus(audio)
        elif scene_type == '和平':
            audio = self._add_soft_compression(audio)
            audio = self._add_warmth(audio)
            
        # 情感特定效果
        if emotion == '紧张':
            audio = self._add_tremolo(audio)
            audio = self._add_filter(audio, 'high')
        elif emotion == '平静':
            audio = self._add_soft_reverb(audio)
            audio = self._add_filter(audio, 'low')
        elif emotion == '欢快':
            audio = self._add_chorus(audio)
            audio = self._add_compression(audio)
            
        return audio
        
    def _normalize(self, audio):
        """音频归一化"""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
        
    def _add_reverb(self, audio, scene_type):
        """添加混响效果"""
        # 根据场景调整混响参数
        if scene_type == '神秘':
            decay = 0.8
            wet_level = 0.4
        elif scene_type == '和平':
            decay = 0.6
            wet_level = 0.3
        else:
            decay = 0.4
            wet_level = 0.2
            
        # 创建混响脉冲响应
        reverb_length = int(self.sample_rate * decay)
        impulse_response = np.exp(-np.linspace(0, 5, reverb_length))
        
        # 应用混响
        reverb = signal.convolve(audio, impulse_response, mode='full')
        reverb = reverb[:len(audio)]  # 裁剪到原始长度
        
        # 混合原始信号和混响
        return (1 - wet_level) * audio + wet_level * reverb
        
    def _add_distortion(self, audio):
        """添加失真效果"""
        # 软削波失真
        return np.tanh(audio * 1.5)
        
    def _add_compression(self, audio):
        """添加压缩效果"""
        # 简单的动态范围压缩
        threshold = 0.5
        ratio = 4
        attack = 0.01
        release = 0.1
        
        # 计算包络
        envelope = np.abs(audio)
        attack_samples = int(attack * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        # 应用压缩
        gain_reduction = np.ones_like(audio)
        mask = envelope > threshold
        gain_reduction[mask] = 1 - (1 - 1/ratio) * (envelope[mask] - threshold) / (1 - threshold)
        
        return audio * gain_reduction
        
    def _add_echo(self, audio):
        """添加回声效果"""
        delay = 0.3  # 300ms延迟
        feedback = 0.3
        delay_samples = int(delay * self.sample_rate)
        
        # 创建延迟信号
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        
        # 混合原始信号和延迟信号
        return audio + feedback * delayed
        
    def _add_chorus(self, audio):
        """添加合唱效果"""
        # 创建调制延迟
        delay = 0.02  # 20ms基础延迟
        depth = 0.002  # 2ms调制深度
        rate = 2  # 2Hz调制速率
        
        # 生成调制信号
        t = np.linspace(0, len(audio)/self.sample_rate, len(audio))
        mod = depth * np.sin(2 * np.pi * rate * t)
        
        # 应用调制延迟
        delayed = np.zeros_like(audio)
        for i in range(len(audio)):
            delay_samples = int((delay + mod[i]) * self.sample_rate)
            if i + delay_samples < len(audio):
                delayed[i + delay_samples] = audio[i]
                
        # 混合原始信号和调制信号
        return 0.7 * audio + 0.3 * delayed
        
    def _add_tremolo(self, audio):
        """添加颤音效果"""
        rate = 5  # 5Hz调制速率
        depth = 0.3  # 30%调制深度
        
        # 生成调制信号
        t = np.linspace(0, len(audio)/self.sample_rate, len(audio))
        mod = 1 - depth * (1 + np.sin(2 * np.pi * rate * t)) / 2
        
        return audio * mod
        
    def _add_filter(self, audio, filter_type):
        """添加滤波器效果"""
        if filter_type == 'low':
            # 低通滤波器
            cutoff = 1000  # 1kHz截止频率
            b, a = signal.butter(4, cutoff/(self.sample_rate/2), 'low')
        else:
            # 高通滤波器
            cutoff = 200  # 200Hz截止频率
            b, a = signal.butter(4, cutoff/(self.sample_rate/2), 'high')
            
        return signal.filtfilt(b, a, audio)
        
    def _add_soft_compression(self, audio):
        """添加柔和压缩效果"""
        # 使用更温和的压缩参数
        threshold = 0.7
        ratio = 2
        attack = 0.05
        release = 0.2
        
        return self._add_compression(audio)
        
    def _add_warmth(self, audio):
        """添加温暖效果"""
        # 使用低通滤波器和轻微谐波增强
        # 低通滤波
        cutoff = 5000  # 5kHz截止频率
        b, a = signal.butter(4, cutoff/(self.sample_rate/2), 'low')
        filtered = signal.filtfilt(b, a, audio)
        
        # 添加轻微谐波
        harmonics = np.tanh(filtered * 0.5)
        
        return 0.7 * filtered + 0.3 * harmonics 