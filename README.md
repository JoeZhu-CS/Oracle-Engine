# Oracle-Engine

一个基于Python的AI音乐生成系统，能够根据场景和情绪自动生成背景音乐。

## 功能特点

- 支持多种场景（battle、exploration、peace、mystery、joy、sadness）和情绪（tension、excitement、relaxation、fear、happiness、sadness）。
- 自动生成旋律、和声、节奏，并合成完整音频。
- 内置多种乐器音色（如trumpet、flute、violin、piano、synth、drums等）。
- 支持多种音阶（major、minor、harmonic_minor、pentatonic、blues等）。
- 音频效果处理（混响、失真、压缩、滤波、淡入淡出等）。
- 生成的音频文件自动保存到 `test_outputs` 文件夹。

## 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/Oracle-Engine.git
   cd Oracle-Engine
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

运行测试脚本生成示例音乐：
```bash
python test_music_generator.py
```

生成的音频文件将保存在 `test_outputs` 文件夹中，文件名格式为 `{场景}_{情绪}.wav`。

## 示例输出

- battle_tension.wav
- exploration_excitement.wav
- peace_relaxation.wav
- mystery_fear.wav
- joy_happiness.wav
- sadness_sadness.wav

## 注意事项

- 生成的音频可能因参数设置不同而有所差异，建议根据实际需求调整代码中的参数。
- 音频效果处理（如滤波、混响）可能会因音频长度不足而跳过，请查看控制台日志。

## 未来计划

- 支持更多场景和情绪。
- 优化音色和音频效果。
- 添加用户交互界面，方便参数调整和实时预览。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT 