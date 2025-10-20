import requests
import json
from typing import Optional, List, Dict
import tqdm
import os
import glob
import re
import argparse # 导入 argparse 模块

class OllamaLLM:
    """
    简化的 Ollama 本地大模型客户端类
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        初始化 Ollama 客户端
        
        Args:
            base_url (str): Ollama 服务的基础 URL，默认为 "http://localhost:11434"
        """
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/api/chat"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def chat(self, model: str, system_prompt: str, user_input: str) -> Optional[str]:
        """
        发送聊天请求并返回回复内容
        
        Args:
            model (str): 使用的模型名称，例如 "gemma3:27b"
            system_prompt (str): 系统提示词
            user_input (str): 用户输入
        
        Returns:
            str: 模型的回复内容，如果请求失败则返回 None
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        data = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        try:
            response = requests.post(self.chat_url, headers=self.headers, 
                                   data=json.dumps(data))
            response.raise_for_status()
            
            response_data = response.json()
            if "message" in response_data and "content" in response_data["message"]:
                return response_data["message"]["content"]
            return None
                
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None

    def parse_llm_output(self, output: str) -> List[str]:
        """
        解析大语言模型的输出，提取分割后的句子列表
        
        Args:
            output (str): 大语言模型的输出
            
        Returns:
            List[str]: 分割后的句子列表
        """
        if not output:
            return []
        
        # 使用正则表达式提取引号中的内容
        pattern = r'"([^"]+)"'
        matches = re.findall(pattern, output)
        
        if matches:
            return matches
        else:
            # 如果没有找到引号格式，返回原始输出（可能是单个句子）
            return [output.strip()]

def process_text_files(input_dir: str, output_dir: str, model: str = "gemma3:27b", max_files: int = None):
    """
    批量处理文本文件，使用大语言模型进行句子分割
    
    Args:
        input_dir (str): 输入文件目录
        output_dir (str): 输出文件目录
        model (str): 使用的模型名称
        max_files (int): 最大处理文件数量，None表示处理所有文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建客户端
    client = OllamaLLM()
    
    # 系统提示词
    system_prompt = (
        "我有多个文字描述动作的句子，这些句子都是描述同一个动作的，"
        "现在你需要帮我对这些句子进行分割，如果他描述了两个或多个动作，"
        "你需要将其分割出来，并且请注意，你只能进行分割，而不能添加或删除单词，"
        "也不能调换单词位置，如果他只描述了一个动作，或者动作不容易分割，则不需要分割，"
        "下面会给出具体的例子，[input]是我具体给你的东西，[output]是你应该输出的东西，"
        "除此之外你不应该输出任何其他的东西，包括解释。\n\n"
        "例如：\n\n"
        "1. [input] a man squats extraordinarily low then bolts up in an unsatisfactory jump.\n"
        '   [output] "a man squats extraordinarily low", "then bolts up in an unsatisfactory jump".\n\n'
        "2. [input] a man picks up an unseen object to his front left and moves it to an unseen platform on this front right without moving his feet.\n"
        '   [output] "a man picks up an unseen object to his front left", "and moves it to an unseen platform on this front right without moving his feet".\n\n'
        "3. [input] person walks backwards and sits down then gets back and and walks forwards.\n"
        '   [output] "person walks backwards", "and sits down", "then gets back and walks forwards".\n\n'
        "4. [input] a man kicks something or someone with his left leg.\n"
        '   [output] "a man kicks something or someone with his left leg".'
    )
    
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    txt_files.sort()  # 按文件名排序
    
    # 限制处理文件数量
    if max_files is not None:
        txt_files = txt_files[:max_files]
    
    print(f"找到 {len(txt_files)} 个文件需要处理")
    
    # 处理每个文件
    for txt_file in tqdm.tqdm(txt_files, desc="处理文件"):
        filename = os.path.basename(txt_file)
        file_id = os.path.splitext(filename)[0]
        
        # 检查输出文件是否已存在
        output_file = os.path.join(output_dir, f"{file_id}.json")
        if os.path.exists(output_file):
            continue
        
        # 读取文件内容
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"读取文件 {filename} 失败: {e}")
            continue
        
        if not lines:
            print(f"文件 {filename} 为空，跳过")
            continue
        
        # 处理每一行
        processed_data = {
            "file_id": file_id,
            "original_lines": lines,
            "processed_lines": []
        }
        for i, line in enumerate(lines):
            # 调用大语言模型
            response = client.chat(
                model=model,
                system_prompt=system_prompt,
                user_input=f"[input] {line}"
            )
            
            if response:
                # 解析输出
                split_sentences = client.parse_llm_output(response)
                processed_data["processed_lines"].append({
                    "line_index": i,
                    "original": line,
                    "llm_response": response,
                    "split_sentences": split_sentences
                })
            else:
                print(f"处理第 {i+1} 行失败: {line}")
         
        # 保存处理结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存文件 {output_file} 失败: {e}")
    
    print("所有文件处理完成！")

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Ollama LLM 批量处理文本文件并进行句子分割。")
    parser.add_argument("--model", type=str, default="deepseek-r1:70b", help="使用的 Ollama 模型名称，例如 gemma3:27b。")
    parser.add_argument("--output", type=str, default="texts_deepseek", help="输出文件将保存到的子目录名称，例如 texts_gemma3。")

    args = parser.parse_args()

    # 定义输入和输出目录
    input_directory = "/data1/yueyi/code/maskedmimic-v1/data/hml3d/texts_processed"
    output_directory = os.path.join("/data1/yueyi/code/maskedmimic-v1/data/hml3d", args.output)

    # 调用处理函数
    process_text_files(input_directory, output_directory, model=args.model)