import numpy as np
from scipy import signal
from config import get_config

class InstrumentSynthesizer:
    def __init__(self):
        self.config = get_config()
        self.sample_rate = self.config.SAMPLE_RATE
        
        # 乐器参数定义
        self.instrument_params = {
            'piano': {
                'attack': 0.01,
                'decay': 0.1,
                'sustain': 0.7,
                'release': 0.3,
                'harmonics': [1.0, 0.5, 0.25, 0.125],
                'filter_cutoff': 2000
            },
            'violin': {
                'attack': 0.1,
                'decay': 0.2,
                'sustain': 0.8,
                'release': 0.4,
                'harmonics': [1.0, 0.7, 0.5, 0.3, 0.2],
                'filter_cutoff': 4000
            },
            'flute': {
                'attack': 0.05,
                'decay': 0.1,
                'sustain': 0.9,
                'release': 0.2,
                'harmonics': [1.0, 0.3, 0.1],
                'filter_cutoff': 3000
            },
            'drums': {
                'attack': 0.001,
                'decay': 0.1,
                'sustain': 0.0,
                'release': 0.05,
                'harmonics': [1.0, 0.8, 0.6],
                'filter_cutoff': 5000
            },
            'synth': {
                'attack': 0.02,
                'decay': 0.1,
                'sustain': 0.8,
                'release': 0.3,
                'harmonics': [1.0, 0.8, 0.6, 0.4, 0.2],
                'filter_cutoff': 6000
            },
            'bass': {
                'attack': 0.02,
                'decay': 0.1,
                'sustain': 0.8,
                'release': 0.2,
                'harmonics': [1.0, 0.5, 0.25],
                'filter_cutoff': 1000
            },
            'cello': {
                'attack': 0.1,
                'decay': 0.2,
                'sustain': 0.8,
                'release': 0.4,
                'harmonics': [1.0, 0.6, 0.4, 0.2],
                'filter_cutoff': 3000
            },
            'trumpet': {
                'attack': 0.05,
                'decay': 0.1,
                'sustain': 0.9,
                'release': 0.2,
                'harmonics': [1.0, 0.8, 0.6, 0.4],
                'filter_cutoff': 4000
            },
            'harp': {
                'attack': 0.02,
                'decay': 0.1,
                'sustain': 0.7,
                'release': 0.3,
                'harmonics': [1.0, 0.5, 0.25, 0.125],
                'filter_cutoff': 5000
            }
        }
        
    def synthesize_note(self, frequency, duration, instrument, velocity=1.0):
        """合成单个音符"""
        # 获取乐器参数
        params = self.instrument_params[instrument]
        
        # 生成时间序列
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # 生成ADSR包络
        envelope = self._generate_adsr(t, params)
        
        # 生成谐波（主波形用正弦波）
        harmonics = self._generate_harmonics(t, frequency, params['harmonics'], base_wave='sine')
        
        # 应用滤波器
        filtered = self._apply_filter(harmonics, params['filter_cutoff'])
        
        # 应用包络
        note = filtered * envelope * velocity
        
        # 添加10ms淡入淡出
        fade_samples = int(0.01 * self.sample_rate)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        note[:fade_samples] *= fade_in
        note[-fade_samples:] *= fade_out
        
        # 归一化，防止削波
        if np.max(np.abs(note)) > 0:
            note = note / np.max(np.abs(note))
        
        return note
        
    def _generate_adsr(self, t, params):
        """生成ADSR包络"""
        attack = params['attack']
        decay = params['decay']
        sustain = params['sustain']
        release = params['release']
        
        # 计算各个阶段的样本数
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        # 创建包络
        envelope = np.zeros_like(t)
        
        # 起音阶段
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            
        # 衰减阶段
        if decay_samples > 0:
            decay_start = attack_samples
            decay_end = decay_start + decay_samples
            envelope[decay_start:decay_end] = np.linspace(1, sustain, decay_samples)
            
        # 持续阶段
        sustain_start = attack_samples + decay_samples
        release_start = len(t) - release_samples
        envelope[sustain_start:release_start] = sustain
        
        # 释音阶段
        if release_samples > 0:
            envelope[release_start:] = np.linspace(sustain, 0, release_samples)
            
        return envelope
        
    def _generate_harmonics(self, t, frequency, harmonics, base_wave='sine'):
        """生成谐波叠加，主波形可选"""
        if base_wave == 'sine':
            wave_func = np.sin
        elif base_wave == 'triangle':
            wave_func = lambda x: 2/np.pi * np.arcsin(np.sin(x))
        elif base_wave == 'square':
            wave_func = lambda x: np.sign(np.sin(x))
        elif base_wave == 'sawtooth':
            wave_func = lambda x: 2*(x/np.pi - np.floor(0.5 + x/np.pi))
        else:
            wave_func = np.sin
        signal = np.zeros_like(t)
        for i, amp in enumerate(harmonics):
            signal += amp * wave_func(2 * np.pi * frequency * (i+1) * t)
        return signal
        
    def _apply_filter(self, audio_signal, cutoff):
        """应用滤波器"""
        # 设计低通滤波器
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normalized_cutoff, 'low')
        
        # 应用滤波器
        filtered = signal.filtfilt(b, a, audio_signal)
        return filtered
        
    def synthesize_chord(self, frequencies, duration, instrument, velocity=1.0):
        """合成和弦"""
        chord = np.zeros(int(duration * self.sample_rate))
        for freq in frequencies:
            note = self.synthesize_note(freq, duration, instrument, velocity)
            chord += note
        return chord / len(frequencies)  # 归一化
        
    def synthesize_sequence(self, notes, durations, instrument, velocities=None):
        """合成音符序列"""
        if velocities is None:
            velocities = [1.0] * len(notes)
            
        sequence = np.array([])
        for note, duration, velocity in zip(notes, durations, velocities):
            note_audio = self.synthesize_note(note, duration, instrument, velocity)
            sequence = np.concatenate([sequence, note_audio])
            
        return sequence 