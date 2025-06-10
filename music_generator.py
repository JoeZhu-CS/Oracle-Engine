import torch
import torch.nn as nn
import numpy as np
import pretty_midi
import fluidsynth
import os
from text_analyzer import TextAnalyzer
import soundfile as sf
from scipy import signal
import random
from config import get_config
from melody_generator import MelodyGenerator
from harmony_generator import HarmonyGenerator
from audio_effects_advanced import AudioEffectsAdvanced
from instrument_synthesizer import InstrumentSynthesizer
from music_theory import MusicTheory

class MusicGenerator(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2):
        super(MusicGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, 128)  # 128个音符
        
        # 初始化权重
        self._init_weights()
        
        # 音阶映射
        self.scale_mapping = {
            'C': [0, 2, 4, 5, 7, 9, 11],  # C大调
            'D': [2, 4, 6, 7, 9, 11, 13],  # D大调
            'E': [4, 6, 8, 9, 11, 13, 15],  # E大调
            'F': [5, 7, 9, 10, 12, 14, 16],  # F大调
            'G': [7, 9, 11, 12, 14, 16, 18],  # G大调
            'A': [9, 11, 13, 14, 16, 18, 20],  # A大调
        }
        
        # 乐器映射
        self.instrument_mapping = {
            'piano': 0,
            'strings': 48,
            'brass': 61,
            'woodwind': 68,
            'synth': 80,
            'drums': 118,
            'bass': 33,
            'harp': 46,
            'cello': 42,
            'bells': 14
        }

        self.config = get_config()
        self.sample_rate = self.config.SAMPLE_RATE
        self.channels = self.config.CHANNELS
        self.bit_depth = self.config.BIT_DEPTH
        self.melody_generator = MelodyGenerator()
        self.harmony_generator = HarmonyGenerator()
        self.audio_effects = AudioEffectsAdvanced()
        self.instrument_synthesizer = InstrumentSynthesizer()
        self.music_theory = MusicTheory()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, hidden=None):
        """前向传播"""
        # LSTM层
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 输出层
        output = self.fc(context)
        return output, hidden

    def generate_melody(self, params, length=32):
        """生成旋律"""
        # 获取音阶
        scale = self.scale_mapping[params['scale']]
        
        # 生成初始序列
        sequence = []
        for _ in range(length):
            # 根据节奏和情感调整音符长度
            if params['rhythm'] == 'fast':
                duration = np.random.choice([0.25, 0.5])
            elif params['rhythm'] == 'slow':
                duration = np.random.choice([1.0, 2.0])
            else:  # medium
                duration = np.random.choice([0.5, 1.0])
            
            # 选择音符
            if params['harmony'] == 'major':
                note = np.random.choice(scale) + 60  # 从中央C开始
            else:  # minor
                note = np.random.choice([x-1 for x in scale if x > 0]) + 60
            
            sequence.append((note, duration))
        
        return sequence

    def create_midi(self, melody, params):
        """创建MIDI文件"""
        pm = pretty_midi.PrettyMIDI()
        
        # 设置速度
        pm.tempo_changes.append(pretty_midi.TempoChange(params['tempo'], 0))
        
        # 添加乐器
        for instrument_name in params['instruments']:
            if instrument_name in self.instrument_mapping:
                program = self.instrument_mapping[instrument_name]
                instrument = pretty_midi.Instrument(program=program)
                
                # 添加音符
                current_time = 0
                for note, duration in melody:
                    # 根据动态调整音量
                    if params['dynamics'] == 'loud':
                        velocity = np.random.randint(80, 127)
                    elif params['dynamics'] == 'soft':
                        velocity = np.random.randint(40, 80)
                    else:  # medium
                        velocity = np.random.randint(60, 100)
                    
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=note,
                        start=current_time,
                        end=current_time + duration
                    )
                    instrument.notes.append(note)
                    current_time += duration
                
                pm.instruments.append(instrument)
        
        return pm

    def make_loop(self, midi_data):
        """使音乐循环"""
        # 获取最后一个音符的结束时间
        end_time = max(note.end for instrument in midi_data.instruments for note in instrument.notes)
        
        # 为每个乐器添加淡入淡出
        for instrument in midi_data.instruments:
            # 淡出
            for note in instrument.notes:
                if note.end > end_time - 0.5:
                    fade_ratio = (end_time - note.start) / 0.5
                    note.velocity = int(note.velocity * fade_ratio)
            
            # 淡入
            for note in instrument.notes:
                if note.start < 0.5:
                    fade_ratio = note.start / 0.5
                    note.velocity = int(note.velocity * fade_ratio)
        
        return midi_data

    def midi_to_audio(self, midi_data, output_file):
        """将MIDI转换为音频"""
        # 保存MIDI文件
        temp_midi = 'temp.mid'
        midi_data.write(temp_midi)
        
        # 使用FluidSynth转换为音频
        fs = fluidsynth.Synth()
        fs.start()
        
        # 加载SoundFont
        sf_path = os.path.join(os.path.dirname(__file__), 'soundfonts', 'default.sf2')
        if not os.path.exists(sf_path):
            raise FileNotFoundError(f"SoundFont文件不存在: {sf_path}")
        
        fs.load_soundfont(sf_path)
        
        # 转换MIDI为音频
        fs.midi_to_audio(temp_midi, output_file)
        
        # 清理
        fs.delete()
        os.remove(temp_midi)

    def generate(self, scene_type, emotion, duration=30):
        """生成音乐"""
        # 获取音乐理论参数
        theory_params = self.music_theory.get_scene_parameters(scene_type)
        emotion_params = self.music_theory.get_emotion_parameters(emotion)
        
        # 合并参数
        scale = f"C {theory_params['scale']}"
        tempo = int(120 * theory_params['tempo'] * emotion_params['tempo'])
        rhythm = theory_params['rhythm']
        dynamics_str = theory_params['dynamics']
        # 动态映射表
        dynamics_map = {
            'fortissimo': {'melody': 1.0, 'harmony': 0.8, 'rhythm': 0.7},
            'forte': {'melody': 0.8, 'harmony': 0.7, 'rhythm': 0.6},
            'mezzo-forte': {'melody': 0.7, 'harmony': 0.6, 'rhythm': 0.5},
            'mezzo-piano': {'melody': 0.5, 'harmony': 0.4, 'rhythm': 0.3},
            'piano': {'melody': 0.3, 'harmony': 0.2, 'rhythm': 0.2},
        }
        dynamics = dynamics_map.get(dynamics_str, {'melody': 0.7, 'harmony': 0.6, 'rhythm': 0.5})
        
        # 获取乐器组合
        instruments = self._get_instruments(scene_type, emotion)
        
        # 生成旋律
        melody_notes = self.melody_generator.generate_melody(
            duration=duration,
            scale=scale,
            tempo=tempo,
            emotion=emotion
        )
        
        # 生成和声
        harmony_notes = self.harmony_generator.generate_harmony(
            duration=duration,
            scale=scale,
            tempo=tempo,
            emotion=emotion
        )
        
        # 合成各个音轨
        melody_track = self.instrument_synthesizer.synthesize_sequence(
            melody_notes['frequencies'],
            melody_notes['durations'],
            instruments['melody']
        )
        
        harmony_track = self.instrument_synthesizer.synthesize_sequence(
            harmony_notes['frequencies'],
            harmony_notes['durations'],
            instruments['harmony']
        )
        
        # 生成节奏
        rhythm_track = self._generate_rhythm_track(duration, tempo, rhythm, instruments['rhythm'])
        
        # 混音
        mixed = self._mix_tracks([melody_track, harmony_track, rhythm_track], dynamics)
        
        # 应用效果
        processed = self.audio_effects.apply_effects(mixed, scene_type, emotion)
        
        # 确保循环
        looped = self._make_loop(processed)
        
        # 分析音乐
        analysis = self._analyze_music(melody_notes, harmony_notes, scale)
        
        return {
            'audio': looped,
            'analysis': {
                'scene_type': scene_type,
                'emotion': emotion,
                'tempo': tempo,
                'scale': scale,
                'instruments': instruments,
                'rhythm': rhythm,
                'dynamics': dynamics,
                'melody_analysis': analysis['melody'],
                'harmony_analysis': analysis['harmony']
            }
        }
        
    def _get_instruments(self, scene_type, emotion):
        """根据场景和情绪选择乐器"""
        # 定义场景和情绪对应的乐器组合
        instrument_sets = {
            'battle': {
                'melody': 'trumpet',
                'harmony': 'synth',
                'rhythm': 'drums'
            },
            'exploration': {
                'melody': 'flute',
                'harmony': 'harp',
                'rhythm': 'bass'
            },
            'peace': {
                'melody': 'violin',
                'harmony': 'piano',
                'rhythm': 'harp'
            },
            'mystery': {
                'melody': 'cello',
                'harmony': 'synth',
                'rhythm': 'bass'
            },
            'joy': {
                'melody': 'trumpet',
                'harmony': 'piano',
                'rhythm': 'drums'
            },
            'sadness': {
                'melody': 'violin',
                'harmony': 'cello',
                'rhythm': 'piano'
            }
        }
        
        return instrument_sets.get(scene_type, {
            'melody': 'piano',
            'harmony': 'synth',
            'rhythm': 'bass'
        })
        
    def _generate_rhythm_track(self, duration, tempo, rhythm, instrument):
        """生成节奏音轨"""
        # 获取节奏模式
        pattern = self.music_theory.get_rhythm_pattern(rhythm)
        
        # 计算每个节拍的持续时间
        beat_duration = 60.0 / tempo
        
        # 生成节奏序列
        rhythm_sequence = []
        for _ in range(int(duration / (beat_duration * len(pattern)))):
            rhythm_sequence.extend(pattern)
            
        # 合成节奏音轨
        rhythm_track = np.zeros(int(duration * self.sample_rate))
        for i, beat in enumerate(rhythm_sequence):
            if beat:
                start_sample = int(i * beat_duration * self.sample_rate)
                end_sample = int((i + 0.5) * beat_duration * self.sample_rate)
                length = end_sample - start_sample
                note_audio = self.instrument_synthesizer.synthesize_note(
                    100,  # 基础频率
                    beat_duration * 0.5,
                    instrument,
                    velocity=0.8
                )
                # 截断或填充
                if len(note_audio) > length:
                    note_audio = note_audio[:length]
                elif len(note_audio) < length:
                    note_audio = np.pad(note_audio, (0, length - len(note_audio)))
                rhythm_track[start_sample:end_sample] = note_audio
                
        return rhythm_track
        
    def _mix_tracks(self, tracks, dynamics):
        """混音"""
        # 确保所有音轨长度相同
        max_length = max(len(track) for track in tracks)
        padded_tracks = [np.pad(track, (0, max_length - len(track))) for track in tracks]
        
        # 应用动态范围
        mixed = np.zeros(max_length)
        for i, track in enumerate(padded_tracks):
            if i == 0:  # 旋律
                mixed += track * dynamics['melody']
            elif i == 1:  # 和声
                mixed += track * dynamics['harmony']
            else:  # 节奏
                mixed += track * dynamics['rhythm']
                
        # 归一化
        mixed = mixed / np.max(np.abs(mixed))
        return mixed
        
    def _make_loop(self, audio, window_size=4410):
        """确保音频可以无缝循环"""
        # 找到合适的循环点
        best_crossfade = self._find_best_crossfade(audio, window_size)
        
        # 应用交叉淡入淡出
        crossfade = np.linspace(0, 1, window_size)
        if audio.ndim == 2:
            crossfade = crossfade[:, None]
        audio[:window_size] *= crossfade
        audio[-window_size:] *= (1 - crossfade)
        
        return audio
        
    def _find_best_crossfade(self, audio, window_size):
        """找到最佳交叉淡入淡出点"""
        best_diff = float('inf')
        best_start = 0
        
        for i in range(0, len(audio) - window_size, window_size):
            diff = np.sum(np.abs(audio[i:i+window_size] - audio[-window_size:]))
            if diff < best_diff:
                best_diff = diff
                best_start = i
                
        return best_start
        
    def _analyze_music(self, melody_notes, harmony_notes, scale):
        """分析音乐特征"""
        # 分析旋律
        melody_analysis = {
            'scale_conformity': self.music_theory.analyze_melody(
                melody_notes['frequencies'],
                self.music_theory.get_scale_notes(0, scale)
            ),
            'rhythm_complexity': len(set(melody_notes['durations'])) / len(melody_notes['durations']),
            'pitch_range': max(melody_notes['frequencies']) - min(melody_notes['frequencies'])
        }
        
        # 分析和声
        harmony_analysis = {
            'tension': self.music_theory.analyze_harmony(
                [(note, 'major') for note in harmony_notes['frequencies']],
                0
            ),
            'chord_complexity': len(set(harmony_notes['frequencies'])) / len(harmony_notes['frequencies']),
            'rhythm_density': sum(1 for d in harmony_notes['durations'] if d < 0.5) / len(harmony_notes['durations'])
        }
        
        return {
            'melody': melody_analysis,
            'harmony': harmony_analysis
        }
        
    def save_audio(self, audio, filename):
        """保存音频文件"""
        sf.write(filename, audio, self.sample_rate)

class MusicGeneratorWrapper:
    def __init__(self):
        self.generator = MusicGenerator()
        self.generator.eval()  # 设置为评估模式
        
        # 创建输出目录
        self.output_dir = "generated_music"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 创建soundfonts目录
        self.sf_dir = os.path.join(os.path.dirname(__file__), 'soundfonts')
        if not os.path.exists(self.sf_dir):
            os.makedirs(self.sf_dir)

    def generate_music(self, text, output_file=None):
        """生成音乐并保存"""
        try:
            # 生成MIDI
            midi_data, analysis = self.generator.generate(text)
            
            # 如果没有指定输出文件，生成一个
            if output_file is None:
                output_file = os.path.join(self.output_dir, f"music_{hash(text)}.wav")
            
            # 转换为音频
            self.generator.midi_to_audio(midi_data, output_file)
            
            return {
                'status': 'success',
                'file': output_file,
                'analysis': analysis
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            } 