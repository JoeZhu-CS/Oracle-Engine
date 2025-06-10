import numpy as np
import soundfile as sf
from music_generator import MusicGenerator
import os

def test_music_generation():
    """测试音乐生成功能"""
    print("开始测试音乐生成系统...")
    
    # 创建输出目录
    if not os.path.exists('test_outputs'):
        os.makedirs('test_outputs')
    
    # 初始化音乐生成器
    generator = MusicGenerator()
    
    # 测试场景和情绪组合
    test_cases = [
        ('battle', 'tension'),
        ('exploration', 'excitement'),
        ('peace', 'relaxation'),
        ('mystery', 'fear'),
        ('joy', 'happiness'),
        ('sadness', 'sadness')
    ]
    
    for scene_type, emotion in test_cases:
        print(f"\n生成 {scene_type} 场景的 {emotion} 情绪音乐...")
        
        # 生成音乐
        result = generator.generate(scene_type, emotion, duration=10)
        
        # 保存音频文件
        output_file = f'test_outputs/{scene_type}_{emotion}.wav'
        generator.save_audio(result['audio'], output_file)
        
        # 打印分析结果
        print("\n音乐分析结果:")
        print(f"场景类型: {result['analysis']['scene_type']}")
        print(f"情绪: {result['analysis']['emotion']}")
        print(f"速度: {result['analysis']['tempo']} BPM")
        print(f"音阶: {result['analysis']['scale']}")
        print(f"乐器组合: {result['analysis']['instruments']}")
        print(f"节奏模式: {result['analysis']['rhythm']}")
        print(f"动态范围: {result['analysis']['dynamics']}")
        
        print("\n旋律分析:")
        print(f"音阶符合度: {result['analysis']['melody_analysis']['scale_conformity']:.2f}")
        print(f"节奏复杂度: {result['analysis']['melody_analysis']['rhythm_complexity']:.2f}")
        print(f"音高范围: {result['analysis']['melody_analysis']['pitch_range']}")
        
        print("\n和声分析:")
        print(f"紧张度: {result['analysis']['harmony_analysis']['tension']:.2f}")
        print(f"和弦复杂度: {result['analysis']['harmony_analysis']['chord_complexity']:.2f}")
        print(f"节奏密度: {result['analysis']['harmony_analysis']['rhythm_density']:.2f}")
        
        print(f"\n音频文件已保存到: {output_file}")
        
    print("\n测试完成！")

if __name__ == '__main__':
    test_music_generation() 