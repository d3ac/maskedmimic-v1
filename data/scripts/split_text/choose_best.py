import json
import os
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    dirpath = ['data/hml3d/texts_qwen3', 'data/hml3d/texts_gptoss', 'data/hml3d/text_gemma3']
    output_dir = 'data/hml3d/texts_best'

    not_similar = []  # 保存格式为 type : x, id : y ，type 1 表示三个中有两个相同，type 3 表示三个都不相同，id就是对应的文件名
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取第一个目录中的所有文件
    dir1_files = sorted(os.listdir(dirpath[0]))
    
    for filename in tqdm(dir1_files, desc="Processing files"):
        file_paths = [os.path.join(dir, filename) for dir in dirpath]
        
        # 检查三个文件是否都存在
        if not all(os.path.exists(fp) for fp in file_paths):
            continue
        
        # 读取三个文件的内容
        contents = []
        for fp in file_paths:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    contents.append(f.read())
            except Exception as e:
                break
        
        if len(contents) != 3:
            continue
        
        # 比较三个文件内容
        content1, content2, content3 = contents
        
        if content1 == content2 == content3:
            # 三个文件都相同，直接保存
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content1)
        
        elif content1 == content2:
            # 文件1和文件2相同，保存它们的内容
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content1)
            not_similar.append({"type": 2, "id": filename, "different": "file3"})
        
        elif content1 == content3:
            # 文件1和文件3相同，保存它们的内容
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content1)
            not_similar.append({"type": 2, "id": filename, "different": "file2"})
        
        elif content2 == content3:
            # 文件2和文件3相同，保存它们的内容
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content2)
            not_similar.append({"type": 2, "id": filename, "different": "file1"})
        
        else:
            # 三个文件都不相同，只记录不保存
            not_similar.append({"type": 3, "id": filename})
    
    # 保存not_similar记录到工作目录
    not_similar_path = os.path.join(os.getcwd(), 'not_similar.json')
    with open(not_similar_path, 'w', encoding='utf-8') as f:
        json.dump(not_similar, indent=2, ensure_ascii=False, fp=f)
