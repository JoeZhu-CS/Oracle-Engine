import numpy as np
from collections import defaultdict
import random
from config import get_config

class MelodyGenerator:
    def __init__(self):
        self.config = get_config()
        self.sample_rate = self.config.SAMPLE_RATE
        
        # 音符频率映射（以A4=440Hz为基准）
        self.note_frequencies = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        # 音阶模式
        self.scale_patterns = {
            'major': [0, 2, 4, 5, 7, 9, 11],  # 大调音阶
            'minor': [0, 2, 3, 5, 7, 8, 10],  # 小调音阶
            'pentatonic': [0, 2, 4, 7, 9],     # 五声音阶
            'blues': [0, 3, 5, 6, 7, 10],      # 布鲁斯音阶
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],  # 和声小调
            'melodic_minor': [0, 2, 3, 5, 7, 9, 11],   # 旋律小调
            'dorian': [0, 2, 3, 5, 7, 9, 10],         # 多利亚调式
            'phrygian': [0, 1, 3, 5, 7, 8, 10],       # 弗里吉亚调式
            'lydian': [0, 2, 4, 6, 7, 9, 11],         # 利底亚调式
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],     # 混合利底亚调式
            'locrian': [0, 1, 3, 5, 6, 8, 10]         # 洛克里亚调式
        }
        
        # 和弦进行
        self.chord_progressions = {
            'major': [
                ['I', 'IV', 'V', 'I'],
                ['I', 'vi', 'IV', 'V'],
                ['I', 'V', 'vi', 'IV']
            ],
            'minor': [
                ['i', 'iv', 'v', 'i'],
                ['i', 'VI', 'iv', 'v'],
                ['i', 'v', 'VI', 'iv']
            ]
        }
        
    def generate_melody(self, duration, scale, tempo, emotion):
        """生成主旋律"""
        # 兼容scale只有一个单词的情况
        if isinstance(scale, str) and len(scale.split()) == 1:
            print(f"警告：scale '{scale}' 缺少根音，已自动补为 'C {scale}'")
            scale = f"C {scale}"
        # 计算小节数和每小节拍数
        beats_per_bar = 4
        bars = int(duration * tempo / 60 / beats_per_bar)
        
        # 获取音阶音符
        scale_notes = self._get_scale_notes(scale)
        
        # 生成和弦进行
        chord_progression = self._get_chord_progression(scale, bars)
        
        # 生成旋律音符
        melody_notes = self._generate_notes(scale_notes, chord_progression, bars, emotion)
        
        # 生成音高和时值序列
        frequencies = [self.note_frequencies[note] for note in melody_notes]
        durations = [0.5 for _ in melody_notes]  # 假设每个音符0.5秒
        
        return {'frequencies': frequencies, 'durations': durations}
        
    def _get_scale_notes(self, scale):
        """获取音阶中的所有音符"""
        # 解析音阶（例如：'C major'）
        root_note, scale_type = scale.split()
        root_index = list(self.note_frequencies.keys()).index(root_note)
        
        # 获取音阶模式
        pattern = self.scale_patterns[scale_type]
        
        # 生成音阶音符
        notes = []
        for i in range(3):  # 生成三个八度
            for step in pattern:
                note_index = (root_index + step) % 12
                note = list(self.note_frequencies.keys())[note_index]
                notes.append(note)
                
        return notes
        
    def _get_chord_progression(self, scale, bars):
        """生成和弦进行"""
        root_note, scale_type = scale.split()
        # 兼容所有 minor 类型
        if 'minor' in scale_type:
            progressions = self.chord_progressions['minor']
        else:
            progressions = self.chord_progressions.get(scale_type, self.chord_progressions['major'])
        progression = random.choice(progressions)
        
        # 重复和弦进行以填满所有小节
        full_progression = []
        while len(full_progression) < bars:
            full_progression.extend(progression)
            
        return full_progression[:bars]
        
    def _generate_notes(self, scale_notes, chord_progression, bars, emotion):
        """生成旋律音符序列"""
        notes = []
        current_bar = 0
        
        # 根据情感调整音符选择策略
        if emotion == '紧张':
            # 使用更多的不协和音程和跳跃
            interval_weights = {1: 0.3, 2: 0.2, 3: 0.2, 4: 0.15, 5: 0.1, 6: 0.05}
        elif emotion == '平静':
            # 使用更多的级进和协和音程
            interval_weights = {1: 0.5, 2: 0.3, 3: 0.15, 4: 0.05}
        else:
            # 平衡的音程分布
            interval_weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
            
        # 生成每个小节的音符
        for bar in range(bars):
            chord = chord_progression[bar]
            bar_notes = self._generate_bar_notes(scale_notes, chord, interval_weights)
            notes.extend(bar_notes)
            current_bar += 1
            
        return notes
        
    def _generate_bar_notes(self, scale_notes, chord, interval_weights):
        """生成一个小节的音符"""
        bar_notes = []
        beats_in_bar = 4  # 4/4拍
        
        # 选择起始音符（优先选择和弦音）
        current_note = random.choice(scale_notes)
        
        # 生成小节内的音符
        for beat in range(beats_in_bar):
            # 决定是否生成新音符
            if random.random() < 0.7:  # 70%的概率生成新音符
                # 选择音程
                interval = random.choices(
                    list(interval_weights.keys()),
                    weights=list(interval_weights.values())
                )[0]
                
                # 决定向上还是向下
                direction = random.choice([-1, 1])
                
                # 计算新音符
                current_index = scale_notes.index(current_note)
                new_index = (current_index + interval * direction) % len(scale_notes)
                current_note = scale_notes[new_index]
                
            bar_notes.append(current_note)
            
        return bar_notes
        
    def _notes_to_audio(self, notes, tempo):
        """将音符序列转换为音频"""
        # 计算每个音符的持续时间
        beat_duration = 60 / tempo  # 每拍的秒数
        note_duration = beat_duration / 2  # 假设每个音符是八分音符
        
        # 生成音频
        audio = np.array([])
        for note in notes:
            # 生成音符的音频
            frequency = self.note_frequencies[note]
            t = np.linspace(0, note_duration, int(note_duration * self.sample_rate))
            note_audio = np.sin(2 * np.pi * frequency * t)
            
            # 添加淡入淡出效果
            fade_duration = 0.01  # 10ms的淡入淡出
            fade_samples = int(fade_duration * self.sample_rate)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            note_audio[:fade_samples] *= fade_in
            note_audio[-fade_samples:] *= fade_out
            
            # 将音符添加到音频序列
            audio = np.concatenate([audio, note_audio])
            
        return audio 