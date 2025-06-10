import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TextAnalyzer:
    def __init__(self):
        # 加载多语言BERT模型
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 预定义场景和情感关键词（中英文）
        self.scene_keywords = {
            '战斗': {
                'zh': ['战斗', '战争', '战斗', 'boss', '敌人', '攻击', '防御'],
                'en': ['battle', 'combat', 'fight', 'boss', 'enemy', 'attack', 'defense']
            },
            '探索': {
                'zh': ['探索', '冒险', '发现', '寻找', '迷宫', '地牢'],
                'en': ['explore', 'adventure', 'discover', 'search', 'maze', 'dungeon']
            },
            '和平': {
                'zh': ['和平', '宁静', '休息', '治愈', '恢复'],
                'en': ['peace', 'calm', 'rest', 'heal', 'recovery']
            },
            '神秘': {
                'zh': ['神秘', '魔法', '未知', '谜题', '隐藏'],
                'en': ['mystery', 'magic', 'unknown', 'puzzle', 'hidden']
            },
            '欢乐': {
                'zh': ['欢乐', '庆祝', '派对', '节日', '快乐'],
                'en': ['joy', 'celebration', 'party', 'festival', 'happy']
            },
            '悲伤': {
                'zh': ['悲伤', '哀伤', '失落', '痛苦', '离别'],
                'en': ['sad', 'sorrow', 'loss', 'pain', 'farewell']
            }
        }
        
        self.emotion_keywords = {
            '紧张': {
                'zh': ['紧张', '焦虑', '恐惧', '危险', '威胁'],
                'en': ['tense', 'anxiety', 'fear', 'danger', 'threat']
            },
            '兴奋': {
                'zh': ['兴奋', '激动', '热情', '活力', '激情'],
                'en': ['excited', 'thrilled', 'passion', 'energy', 'enthusiasm']
            },
            '平静': {
                'zh': ['平静', '安宁', '放松', '舒适', '平和'],
                'en': ['calm', 'peaceful', 'relaxed', 'comfortable', 'serene']
            },
            '神秘': {
                'zh': ['神秘', '奇幻', '魔法', '未知', '探索'],
                'en': ['mysterious', 'fantasy', 'magic', 'unknown', 'explore']
            },
            '悲伤': {
                'zh': ['悲伤', '哀伤', '忧郁', '思念', '回忆'],
                'en': ['sad', 'sorrow', 'melancholy', 'nostalgia', 'memory']
            },
            '欢乐': {
                'zh': ['欢乐', '快乐', '喜悦', '幸福', '欢庆'],
                'en': ['joy', 'happy', 'delight', 'happiness', 'celebration']
            }
        }

    def get_embedding(self, text):
        """获取文本的BERT嵌入向量"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()

    def analyze_scene(self, text):
        """分析场景类型"""
        text_embedding = self.get_embedding(text)
        max_similarity = -1
        best_scene = '探索'  # 默认场景
        
        for scene, keywords in self.scene_keywords.items():
            # 合并中英文关键词
            all_keywords = ' '.join(keywords['zh'] + keywords['en'])
            keyword_embeddings = self.get_embedding(all_keywords)
            similarity = cosine_similarity(text_embedding, keyword_embeddings)[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_scene = scene
        
        return best_scene

    def analyze_emotion(self, text):
        """分析情感类型"""
        text_embedding = self.get_embedding(text)
        max_similarity = -1
        best_emotion = '平静'  # 默认情感
        
        for emotion, keywords in self.emotion_keywords.items():
            # 合并中英文关键词
            all_keywords = ' '.join(keywords['zh'] + keywords['en'])
            keyword_embeddings = self.get_embedding(all_keywords)
            similarity = cosine_similarity(text_embedding, keyword_embeddings)[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_emotion = emotion
        
        return best_emotion

    def get_music_params(self, scene, emotion):
        """根据场景和情感生成音乐参数"""
        params = {
            'tempo': 120,  # 默认速度
            'scale': 'C',  # 默认音阶
            'instruments': [],  # 乐器列表
            'rhythm': 'medium',  # 节奏类型
            'dynamics': 'medium',  # 动态范围
            'harmony': 'major'  # 和声类型
        }
        
        # 根据场景调整参数
        if scene == '战斗':
            params['tempo'] = 160
            params['scale'] = 'D'
            params['instruments'] = ['drums', 'bass', 'brass']
            params['rhythm'] = 'fast'
            params['dynamics'] = 'loud'
            params['harmony'] = 'minor'
        elif scene == '探索':
            params['tempo'] = 100
            params['scale'] = 'G'
            params['instruments'] = ['piano', 'strings', 'woodwind']
            params['rhythm'] = 'medium'
            params['dynamics'] = 'medium'
            params['harmony'] = 'major'
        elif scene == '和平':
            params['tempo'] = 80
            params['scale'] = 'F'
            params['instruments'] = ['piano', 'strings', 'harp']
            params['rhythm'] = 'slow'
            params['dynamics'] = 'soft'
            params['harmony'] = 'major'
        elif scene == '神秘':
            params['tempo'] = 90
            params['scale'] = 'A'
            params['instruments'] = ['synth', 'strings', 'bells']
            params['rhythm'] = 'medium'
            params['dynamics'] = 'medium'
            params['harmony'] = 'minor'
        elif scene == '欢乐':
            params['tempo'] = 140
            params['scale'] = 'C'
            params['instruments'] = ['piano', 'brass', 'drums']
            params['rhythm'] = 'fast'
            params['dynamics'] = 'loud'
            params['harmony'] = 'major'
        elif scene == '悲伤':
            params['tempo'] = 70
            params['scale'] = 'E'
            params['instruments'] = ['piano', 'strings', 'cello']
            params['rhythm'] = 'slow'
            params['dynamics'] = 'soft'
            params['harmony'] = 'minor'
        
        # 根据情感微调参数
        if emotion == '紧张':
            params['tempo'] += 20
            params['dynamics'] = 'loud'
            params['harmony'] = 'minor'
        elif emotion == '兴奋':
            params['tempo'] += 30
            params['dynamics'] = 'loud'
        elif emotion == '平静':
            params['tempo'] -= 20
            params['dynamics'] = 'soft'
        elif emotion == '神秘':
            params['instruments'].append('synth')
            params['harmony'] = 'minor'
        elif emotion == '悲伤':
            params['tempo'] -= 30
            params['dynamics'] = 'soft'
            params['harmony'] = 'minor'
        elif emotion == '欢乐':
            params['tempo'] += 20
            params['dynamics'] = 'loud'
        
        return params

    def analyze(self, text):
        """分析文本并返回音乐参数"""
        scene = self.analyze_scene(text)
        emotion = self.analyze_emotion(text)
        params = self.get_music_params(scene, emotion)
        
        return {
            'scene': scene,
            'emotion': emotion,
            'params': params
        } 