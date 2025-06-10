from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from datetime import datetime
from text_analyzer import TextAnalyzer
from music_generator import MusicGenerator
from config import get_config

# 初始化Flask应用
app = Flask(__name__)
CORS(app)

# 加载配置
app.config.from_object(get_config())

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化分析器和生成器
text_analyzer = TextAnalyzer()
music_generator = MusicGenerator()

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_music():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': '请提供场景描述文本'
            }), 400

        text = data['text']
        
        # 分析文本
        analysis = text_analyzer.analyze(text)
        
        # 生成音乐
        audio_data = music_generator.generate(
            scene_type=analysis['scene'],
            emotion=analysis['emotion'],
            params=analysis['params']
        )
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'music_{timestamp}.wav'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # 保存音频文件
        audio_data.save(filepath)
        
        return jsonify({
            'status': 'success',
            'data': {
                'filename': filename,
                'analysis': analysis
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({
                'status': 'error',
                'message': '文件不存在'
            }), 404
            
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='audio/wav'
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'success',
        'message': '服务正常运行',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=app.config['DEBUG']
    ) 