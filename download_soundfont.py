import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # 创建soundfonts目录
    sf_dir = os.path.join(os.path.dirname(__file__), 'soundfonts')
    if not os.path.exists(sf_dir):
        os.makedirs(sf_dir)
    
    # SoundFont文件URL（使用Fluid R3 SoundFont）
    url = "https://github.com/musescore/MuseScore/raw/master/share/sound/FluidR3Mono_GM.sf3"
    filename = os.path.join(sf_dir, 'default.sf2')
    
    # 如果文件不存在，则下载
    if not os.path.exists(filename):
        print("正在下载SoundFont文件...")
        download_file(url, filename)
        print("下载完成！")
    else:
        print("SoundFont文件已存在。")

if __name__ == '__main__':
    main() 