import numpy as np
from config import get_config

class MusicTheory:
    def __init__(self):
        self.config = get_config()
        
        # 定义音阶
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],  # 大调音阶
            'minor': [0, 2, 3, 5, 7, 8, 10],  # 小调音阶
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],  # 和声小调
            'melodic_minor': [0, 2, 3, 5, 7, 9, 11],  # 旋律小调
            'dorian': [0, 2, 3, 5, 7, 9, 10],  # 多利亚调式
            'phrygian': [0, 1, 3, 5, 7, 8, 10],  # 弗里吉亚调式
            'lydian': [0, 2, 4, 6, 7, 9, 11],  # 利底亚调式
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],  # 混合利底亚调式
            'locrian': [0, 1, 3, 5, 6, 8, 10]  # 洛克里亚调式
        }
        
        # 定义和弦
        self.chords = {
            'major': [0, 4, 7],  # 大三和弦
            'minor': [0, 3, 7],  # 小三和弦
            'diminished': [0, 3, 6],  # 减三和弦
            'augmented': [0, 4, 8],  # 增三和弦
            'major7': [0, 4, 7, 11],  # 大七和弦
            'minor7': [0, 3, 7, 10],  # 小七和弦
            'dominant7': [0, 4, 7, 10],  # 属七和弦
            'diminished7': [0, 3, 6, 9],  # 减七和弦
            'half_diminished7': [0, 3, 6, 10],  # 半减七和弦
            'major9': [0, 4, 7, 11, 14],  # 大九和弦
            'minor9': [0, 3, 7, 10, 14],  # 小九和弦
            'dominant9': [0, 4, 7, 10, 14]  # 属九和弦
        }
        
        # 定义和弦进行
        self.progressions = {
            'major': {
                'basic': ['I', 'IV', 'V', 'I'],
                'pop': ['I', 'V', 'vi', 'IV'],
                'jazz': ['ii', 'V', 'I'],
                'blues': ['I', 'I', 'I', 'I', 'IV', 'IV', 'I', 'I', 'V', 'IV', 'I', 'V'],
                'romantic': ['I', 'vi', 'IV', 'V'],
                'classical': ['I', 'V', 'vi', 'iii', 'IV', 'I', 'IV', 'V']
            },
            'minor': {
                'basic': ['i', 'iv', 'V', 'i'],
                'harmonic': ['i', 'iv', 'V7', 'i'],
                'jazz': ['ii°', 'V7', 'i'],
                'romantic': ['i', 'VI', 'III', 'VII'],
                'classical': ['i', 'V', 'iv', 'V', 'i']
            }
        }
        
        # 定义节奏模式
        self.rhythm_patterns = {
            'waltz': [3, 0, 0],  # 3/4拍
            'march': [2, 0],  # 2/4拍
            'swing': [2, 1],  # 摇摆节奏
            'bossanova': [2, 0, 1, 0],  # 波萨诺瓦节奏
            'samba': [1, 0, 1, 0, 1, 0],  # 桑巴节奏
            'tango': [2, 0, 2, 0],  # 探戈节奏
            'rumba': [2, 0, 1, 0, 1, 0],  # 伦巴节奏
            'cha_cha': [2, 0, 1, 0, 1, 0, 1, 0]  # 恰恰节奏
        }
        
    def get_scale_notes(self, root_note, scale_type='major'):
        """获取指定根音和音阶类型的音符"""
        scale = self.scales.get(scale_type, self.scales['major'])
        return [root_note + note for note in scale]
        
    def get_chord_notes(self, root_note, chord_type='major'):
        """获取指定根音和和弦类型的音符"""
        chord = self.chords.get(chord_type, self.chords['major'])
        return [root_note + note for note in chord]
        
    def get_chord_progression(self, key, scale_type='major', progression_type='basic'):
        """获取指定调性和进行类型的和弦进行"""
        progressions = self.progressions.get(scale_type, self.progressions['major'])
        progression = progressions.get(progression_type, progressions['basic'])
        
        # 将罗马数字转换为实际和弦
        chord_map = {
            'I': (0, 'major'),
            'ii': (1, 'minor'),
            'iii': (2, 'minor'),
            'IV': (3, 'major'),
            'V': (4, 'major'),
            'vi': (5, 'minor'),
            'vii°': (6, 'diminished'),
            'i': (0, 'minor'),
            'II': (1, 'major'),
            'III': (2, 'major'),
            'iv': (3, 'minor'),
            'v': (4, 'minor'),
            'VI': (5, 'major'),
            'VII': (6, 'major'),
            'ii°': (1, 'diminished'),
            'V7': (4, 'dominant7')
        }
        
        return [(key + offset, chord_type) for roman_numeral in progression 
                for offset, chord_type in [chord_map[roman_numeral]]]
        
    def get_rhythm_pattern(self, pattern_type='waltz'):
        """获取指定类型的节奏模式"""
        return self.rhythm_patterns.get(pattern_type, self.rhythm_patterns['waltz'])
        
    def analyze_melody(self, notes, scale):
        """分析旋律的音阶符合度"""
        scale_notes = set(scale)
        in_scale = sum(1 for note in notes if note % 12 in scale_notes)
        return in_scale / len(notes) if notes else 0
        
    def analyze_harmony(self, chords, key):
        """分析和声的进行"""
        # 计算和弦进行的紧张度
        tension = 0
        for i in range(len(chords) - 1):
            current = chords[i]
            next_chord = chords[i + 1]
            # 计算和弦之间的音程关系
            tension += abs(current[0] - next_chord[0]) % 12
        return tension / (len(chords) - 1) if len(chords) > 1 else 0
        
    def get_emotion_parameters(self, emotion):
        """根据情绪获取音乐参数"""
        parameters = {
            'happy': {
                'scale': 'major',
                'tempo': 1.2,
                'dynamics': 'forte',
                'rhythm': 'upbeat'
            },
            'sad': {
                'scale': 'minor',
                'tempo': 0.8,
                'dynamics': 'piano',
                'rhythm': 'slow'
            },
            'angry': {
                'scale': 'harmonic_minor',
                'tempo': 1.4,
                'dynamics': 'fortissimo',
                'rhythm': 'intense'
            },
            'peaceful': {
                'scale': 'major',
                'tempo': 0.9,
                'dynamics': 'piano',
                'rhythm': 'gentle'
            },
            'mysterious': {
                'scale': 'locrian',
                'tempo': 1.0,
                'dynamics': 'mezzo-piano',
                'rhythm': 'complex'
            },
            'romantic': {
                'scale': 'major',
                'tempo': 1.1,
                'dynamics': 'mezzo-forte',
                'rhythm': 'waltz'
            }
        }
        return parameters.get(emotion, parameters['peaceful'])
        
    def get_scene_parameters(self, scene_type):
        """根据场景类型获取音乐参数"""
        parameters = {
            'battle': {
                'scale': 'harmonic_minor',
                'tempo': 1.3,
                'dynamics': 'fortissimo',
                'rhythm': 'intense'
            },
            'exploration': {
                'scale': 'major',
                'tempo': 1.0,
                'dynamics': 'mezzo-forte',
                'rhythm': 'moderate'
            },
            'peace': {
                'scale': 'major',
                'tempo': 0.8,
                'dynamics': 'piano',
                'rhythm': 'gentle'
            },
            'mystery': {
                'scale': 'locrian',
                'tempo': 0.9,
                'dynamics': 'mezzo-piano',
                'rhythm': 'complex'
            },
            'joy': {
                'scale': 'major',
                'tempo': 1.2,
                'dynamics': 'forte',
                'rhythm': 'upbeat'
            },
            'sadness': {
                'scale': 'minor',
                'tempo': 0.7,
                'dynamics': 'piano',
                'rhythm': 'slow'
            }
        }
        return parameters.get(scene_type, parameters['peace']) 