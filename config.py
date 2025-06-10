import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 基础配置
class Config:
    # 应用配置
    APP_NAME = "游戏音乐生成器 | Game Music Generator"
    DEBUG = True
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    
    # 文件存储配置
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'wav', 'mp3'}
    
    # 音乐生成配置
    SAMPLE_RATE = 44100
    CHANNELS = 2
    BIT_DEPTH = 16
    
    # 音乐参数范围
    TEMPO_RANGE = {
        'min': 60,
        'max': 180
    }
    
    SCALES = {
        'major': ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F'],
        'minor': ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m', 'Fm', 'Cm', 'Gm', 'Dm']
    }
    
    # 乐器配置
    INSTRUMENTS = {
        'piano': {'type': 'acoustic', 'category': 'keyboard'},
        'guitar': {'type': 'acoustic', 'category': 'string'},
        'violin': {'type': 'acoustic', 'category': 'string'},
        'flute': {'type': 'acoustic', 'category': 'wind'},
        'drums': {'type': 'acoustic', 'category': 'percussion'},
        'synth': {'type': 'electronic', 'category': 'keyboard'},
        'bass': {'type': 'acoustic', 'category': 'string'},
        'cello': {'type': 'acoustic', 'category': 'string'},
        'trumpet': {'type': 'acoustic', 'category': 'brass'},
        'harp': {'type': 'acoustic', 'category': 'string'}
    }
    
    # 情感参数映射
    EMOTION_PARAMS = {
        '紧张': {
            'tempo': {'min': 120, 'max': 180},
            'dynamics': 'forte',
            'rhythm': 'complex',
            'harmony': 'dissonant'
        },
        '平静': {
            'tempo': {'min': 60, 'max': 90},
            'dynamics': 'piano',
            'rhythm': 'simple',
            'harmony': 'consonant'
        },
        '欢快': {
            'tempo': {'min': 100, 'max': 140},
            'dynamics': 'mezzo-forte',
            'rhythm': 'moderate',
            'harmony': 'major'
        },
        '神秘': {
            'tempo': {'min': 70, 'max': 100},
            'dynamics': 'mezzo-piano',
            'rhythm': 'moderate',
            'harmony': 'minor'
        }
    }
    
    # 场景参数映射
    SCENE_PARAMS = {
        '战斗': {
            'instruments': ['drums', 'synth', 'bass'],
            'tempo': {'min': 140, 'max': 180},
            'dynamics': 'forte'
        },
        '探索': {
            'instruments': ['piano', 'violin', 'flute'],
            'tempo': {'min': 80, 'max': 120},
            'dynamics': 'mezzo-piano'
        },
        '和平': {
            'instruments': ['harp', 'flute', 'piano'],
            'tempo': {'min': 60, 'max': 90},
            'dynamics': 'piano'
        },
        '神秘': {
            'instruments': ['synth', 'cello', 'harp'],
            'tempo': {'min': 70, 'max': 100},
            'dynamics': 'mezzo-piano'
        }
    }
    
    # 语言配置
    LANGUAGES = {
        'zh': {
            'scene_types': {
                '战斗': 'battle',
                '探索': 'exploration',
                '和平': 'peace',
                '神秘': 'mystery'
            },
            'emotions': {
                '紧张': 'tense',
                '平静': 'calm',
                '欢快': 'joyful',
                '神秘': 'mysterious'
            }
        },
        'en': {
            'scene_types': {
                'battle': '战斗',
                'exploration': '探索',
                'peace': '和平',
                'mystery': '神秘'
            },
            'emotions': {
                'tense': '紧张',
                'calm': '平静',
                'joyful': '欢快',
                'mysterious': '神秘'
            }
        }
    }

# 开发环境配置
class DevelopmentConfig(Config):
    DEBUG = True

# 生产环境配置
class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')

# 配置映射
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# 获取当前配置
def get_config():
    env = os.getenv('FLASK_ENV', 'default')
    return config[env] 