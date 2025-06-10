import numpy as np
from config import get_config
import random

class HarmonyGenerator:
    def __init__(self):
        self.config = get_config()
        self.sample_rate = self.config.SAMPLE_RATE
        
        # 和弦定义
        self.chords = {
            'major': {
                'I': [0, 4, 7],    # 大三和弦
                'ii': [2, 5, 9],   # 小三和弦
                'iii': [4, 7, 11], # 小三和弦
                'IV': [5, 9, 0],   # 大三和弦
                'V': [7, 11, 2],   # 大三和弦
                'vi': [9, 0, 4],   # 小三和弦
                'vii°': [11, 2, 5] # 减三和弦
            },
            'minor': {
                'i': [0, 3, 7],    # 小三和弦
                'ii°': [2, 5, 8],  # 减三和弦
                'III': [3, 7, 10], # 大三和弦
                'iv': [5, 8, 0],   # 小三和弦
                'v': [7, 10, 2],   # 小三和弦
                'VI': [8, 0, 3],   # 大三和弦
                'VII': [10, 2, 5]  # 大三和弦
            }
        }
        
        # 和弦进行模式
        self.progression_patterns = {
            'major': [
                ['I', 'IV', 'V', 'I'],
                ['I', 'vi', 'IV', 'V'],
                ['I', 'V', 'vi', 'IV'],
                ['I', 'IV', 'vi', 'V'],
                ['I', 'iii', 'vi', 'IV']
            ],
            'minor': [
                ['i', 'iv', 'v', 'i'],
                ['i', 'VI', 'iv', 'v'],
                ['i', 'v', 'VI', 'iv'],
                ['i', 'iv', 'VI', 'v'],
                ['i', 'III', 'VI', 'iv']
            ]
        }
        
        # 和声节奏模式
        self.rhythm_patterns = {
            'simple': [1, 0, 0, 0],      # 每小节一个和弦
            'moderate': [1, 0, 1, 0],    # 每小节两个和弦
            'complex': [1, 0.5, 0.5, 1]  # 每小节多个和弦
        }
        
    def generate_harmony(self, duration, scale, tempo, emotion):
        """生成和声进行"""
        # 兼容scale只有一个单词的情况
        if isinstance(scale, str) and len(scale.split()) == 1:
            print(f"警告：scale '{scale}' 缺少根音，已自动补为 'C {scale}'")
            scale = f"C {scale}"
        # 解析音阶
        root_note, scale_type = scale.split()
        
        # 获取和弦进行
        progression = self._get_progression(scale_type)
        
        # 计算小节数
        beats_per_bar = 4
        bars = int(duration * tempo / 60 / beats_per_bar)
        
        # 生成和声音符
        harmony_notes = self._generate_harmony_notes(
            progression, 
            scale_type, 
            bars, 
            emotion
        )
        
        # 生成音高和时值序列
        frequencies = [440 * (2 ** ((note - 69) / 12)) if note is not None else 0 for note in harmony_notes]
        durations = [0.5 for _ in harmony_notes]  # 假设每个音符0.5秒
        
        return {'frequencies': frequencies, 'durations': durations}
        
    def _get_progression(self, scale_type):
        """获取和弦进行"""
        # 兼容所有 minor 类型
        if 'minor' in scale_type:
            patterns = self.progression_patterns['minor']
        else:
            patterns = self.progression_patterns.get(scale_type, self.progression_patterns['major'])
        return random.choice(patterns)
        
    def _generate_harmony_notes(self, progression, scale_type, bars, emotion):
        """生成和声音符序列"""
        notes = []
        current_bar = 0
        
        # 根据情感选择节奏模式
        if emotion == '紧张':
            rhythm_pattern = self.rhythm_patterns['complex']
        elif emotion == '平静':
            rhythm_pattern = self.rhythm_patterns['simple']
        else:
            rhythm_pattern = self.rhythm_patterns['moderate']
            
        # 生成每个小节的音符
        while current_bar < bars:
            # 选择当前小节的和弦
            chord_degree = progression[current_bar % len(progression)]
            # 兼容所有 minor 类型
            if 'minor' in scale_type:
                chord_notes = self.chords['minor'][chord_degree]
            else:
                chord_notes = self.chords.get(scale_type, self.chords['major'])[chord_degree]
            
            # 生成和弦音符
            for beat in range(4):  # 4/4拍
                if rhythm_pattern[beat] > 0:
                    # 选择和弦中的音符
                    chord_note = random.choice(chord_notes)
                    # 添加八度变化
                    octave = random.randint(3, 5)
                    note = chord_note + (octave * 12)
                    notes.append(note)
                else:
                    notes.append(None)  # 休止符
                    
            current_bar += 1
            
        return notes
        
    def _notes_to_audio(self, notes, tempo):
        """将音符序列转换为音频"""
        # 计算每个音符的持续时间
        beat_duration = 60 / tempo  # 每拍的秒数
        note_duration = beat_duration / 2  # 假设每个音符是八分音符
        
        # 生成音频
        audio = np.array([])
        for note in notes:
            if note is not None:
                # 生成音符的音频
                frequency = 440 * (2 ** ((note - 69) / 12))  # A4 = 440Hz
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
            else:
                # 添加休止符
                silence = np.zeros(int(note_duration * self.sample_rate))
                audio = np.concatenate([audio, silence])
                
        return audio 