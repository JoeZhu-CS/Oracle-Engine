import numpy as np
from scipy import signal
from config import get_config

class AudioEffectsAdvanced:
    def __init__(self):
        self.config = get_config()
        self.sample_rate = self.config.SAMPLE_RATE
        
    def apply_effects(self, audio, scene_type, emotion):
        """应用音频效果"""
        # 基础效果
        audio = self._normalize(audio)
        audio = self._apply_reverb(audio)
        
        # 场景特定效果
        if scene_type == 'battle':
            audio = self._apply_distortion(audio, amount=0.3)
            audio = self._apply_compression(audio, threshold=-20, ratio=4)
            audio = self._apply_echo(audio, delay=0.1, feedback=0.3)
        elif scene_type == 'mystery':
            audio = self._apply_chorus(audio, rate=2, depth=0.5)
            audio = self._apply_tremolo(audio, rate=5, depth=0.3)
            audio = self._apply_filter(audio, cutoff=2000, resonance=2)
        elif scene_type == 'peace':
            audio = self._apply_soft_compression(audio)
            audio = self._apply_warmth(audio)
            audio = self._apply_stereo_width(audio, width=1.5)
        elif scene_type == 'joy':
            audio = self._apply_chorus(audio, rate=3, depth=0.4)
            audio = self._apply_compression(audio, threshold=-15, ratio=2)
            audio = self._apply_stereo_width(audio, width=1.3)
        elif scene_type == 'sadness':
            audio = self._apply_reverb(audio, room_size=0.8, damping=0.5)
            audio = self._apply_filter(audio, cutoff=3000, resonance=1)
            audio = self._apply_soft_compression(audio)
            
        # 情绪特定效果
        if emotion == 'tension':
            audio = self._apply_distortion(audio, amount=0.2)
            audio = self._apply_compression(audio, threshold=-18, ratio=3)
        elif emotion == 'relaxation':
            audio = self._apply_soft_compression(audio)
            audio = self._apply_warmth(audio)
        elif emotion == 'excitement':
            audio = self._apply_chorus(audio, rate=4, depth=0.6)
            audio = self._apply_compression(audio, threshold=-12, ratio=2.5)
        elif emotion == 'fear':
            audio = self._apply_distortion(audio, amount=0.4)
            audio = self._apply_filter(audio, cutoff=1000, resonance=3)
            audio = self._apply_echo(audio, delay=0.2, feedback=0.4)
            
        return audio
        
    def _normalize(self, audio):
        """归一化音频"""
        return audio / np.max(np.abs(audio))
        
    def _apply_reverb(self, audio, room_size=0.5, damping=0.5):
        """应用混响效果"""
        # 创建延迟线
        delay_samples = int(room_size * self.sample_rate)
        delay_line = np.zeros(len(audio) + delay_samples)
        delay_line[:len(audio)] = audio
        
        # 创建反馈
        feedback = np.exp(-damping * np.arange(delay_samples) / self.sample_rate)
        
        # 应用混响
        reverb = np.zeros_like(audio)
        for i in range(len(audio)):
            reverb[i] = audio[i] + np.sum(delay_line[i:i+delay_samples] * feedback)
            
        return self._normalize(reverb)
        
    def _apply_distortion(self, audio, amount=0.2):
        """应用失真效果"""
        return np.tanh(audio * (1 + amount * 10))
        
    def _apply_compression(self, audio, threshold=-20, ratio=4):
        """应用压缩效果"""
        # 计算增益
        gain = np.ones_like(audio)
        mask = np.abs(audio) > 10 ** (threshold / 20)
        gain[mask] = (1 + (np.abs(audio[mask]) - 10 ** (threshold / 20)) * (1 - 1/ratio)) / np.abs(audio[mask])
        
        return audio * gain
        
    def _apply_soft_compression(self, audio):
        """应用柔和压缩"""
        return self._apply_compression(audio, threshold=-24, ratio=2)
        
    def _apply_echo(self, audio, delay=0.1, feedback=0.3):
        """应用回声效果"""
        delay_samples = int(delay * self.sample_rate)
        echo = np.zeros_like(audio)
        echo[delay_samples:] = audio[:-delay_samples] * feedback
        return self._normalize(audio + echo)
        
    def _apply_chorus(self, audio, rate=2, depth=0.5):
        """应用合唱效果"""
        t = np.arange(len(audio)) / self.sample_rate
        delay = depth * np.sin(2 * np.pi * rate * t)
        delay_samples = (delay * self.sample_rate).astype(int)
        
        chorus = np.zeros_like(audio)
        for i in range(len(audio)):
            if i + delay_samples[i] < len(audio):
                chorus[i] = audio[i] + 0.5 * audio[i + delay_samples[i]]
                
        return self._normalize(chorus)
        
    def _apply_tremolo(self, audio, rate=5, depth=0.3):
        """应用颤音效果"""
        t = np.arange(len(audio)) / self.sample_rate
        tremolo = 1 + depth * np.sin(2 * np.pi * rate * t)
        return audio * tremolo
        
    def _apply_filter(self, audio, cutoff=3000, resonance=1):
        """应用滤波器"""
        nyquist = self.sample_rate / 2
        norm_cutoff = cutoff / nyquist
        b, a = signal.butter(4, norm_cutoff, 'low', analog=False)
        padlen = 3 * max(len(a), len(b))
        print(f"调试：音频长度 {len(audio)}，滤波器系数长度 a={len(a)}, b={len(b)}，padlen={padlen}")
        if len(audio) <= padlen:
            print(f"警告：音频长度 {len(audio)} <= padlen {padlen}，跳过滤波。")
            return audio
        try:
            return signal.filtfilt(b, a, audio)
        except Exception as e:
            print(f"滤波异常：{e}，已跳过滤波。")
            return audio
        
    def _apply_warmth(self, audio):
        """应用温暖效果"""
        # 应用低通滤波器
        audio = self._apply_filter(audio, cutoff=3000, resonance=1)
        # 添加轻微失真
        audio = self._apply_distortion(audio, amount=0.1)
        return audio
        
    def _apply_stereo_width(self, audio, width=1.5):
        """应用立体声宽度效果"""
        # 创建左右声道
        left = audio * (1 + width/2)
        right = audio * (1 - width/2)
        return np.vstack((left, right)).T
        
    def _apply_phaser(self, audio, rate=1, depth=0.5):
        """应用相位效果"""
        t = np.arange(len(audio)) / self.sample_rate
        phase = depth * np.sin(2 * np.pi * rate * t)
        
        # 创建全通滤波器
        b = [1, -np.exp(phase)]
        a = [1, np.exp(phase)]
        
        return signal.filtfilt(b, a, audio)
        
    def _apply_flanger(self, audio, rate=0.5, depth=0.002):
        """应用镶边效果"""
        t = np.arange(len(audio)) / self.sample_rate
        delay = depth * np.sin(2 * np.pi * rate * t)
        delay_samples = (delay * self.sample_rate).astype(int)
        
        flanger = np.zeros_like(audio)
        for i in range(len(audio)):
            if i + delay_samples[i] < len(audio):
                flanger[i] = audio[i] + 0.7 * audio[i + delay_samples[i]]
                
        return self._normalize(flanger)
        
    def _apply_bit_crusher(self, audio, bits=8):
        """应用位压缩效果"""
        scale = 2 ** (bits - 1)
        return np.round(audio * scale) / scale
        
    def _apply_ring_modulation(self, audio, frequency=100):
        """应用环形调制效果"""
        t = np.arange(len(audio)) / self.sample_rate
        carrier = np.sin(2 * np.pi * frequency * t)
        return audio * carrier
        
    def _apply_granular(self, audio, grain_size=0.1, overlap=0.5):
        """应用颗粒合成效果"""
        grain_samples = int(grain_size * self.sample_rate)
        overlap_samples = int(overlap * grain_samples)
        
        granular = np.zeros_like(audio)
        for i in range(0, len(audio) - grain_samples, grain_samples - overlap_samples):
            grain = audio[i:i+grain_samples]
            window = np.hanning(len(grain))
            granular[i:i+grain_samples] += grain * window
            
        return self._normalize(granular) 