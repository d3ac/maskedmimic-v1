import os
import torch
import re
import numpy as np
from typing import List, Optional
from vllm import LLM, SamplingParams
import tqdm
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"

class TextSplitter:
    """
    使用 vLLM 加载的 Qwen3 模型进行文本分割的类
    """
    
    def __init__(
        self, 
        model_path: str = "/home/yueyi/largemodel/Qwen3-235B-A22B-Instruct-2507-AWQ",
        tensor_parallel_size: int = 4,
        max_model_len: int = 262144,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024
    ):
        """
        初始化 TextSplitter
        
        Args:
            model_path (str): Qwen3 模型路径
            tensor_parallel_size (int): 张量并行大小
            max_model_len (int): 最大模型长度
            temperature (float): 采样温度
            top_p (float): nucleus sampling 参数
            max_tokens (int): 最大生成 token 数
        """
        # 设置环境变量
        os.environ["VLLM_USE_MODELSCOPE"] = "true"
        
        # 检查 GPU 数量
        available_gpus = torch.cuda.device_count()
        if available_gpus < tensor_parallel_size:
            raise SystemExit(f"GPU 数量不足，需要 {tensor_parallel_size} 个，但只有 {available_gpus} 个可用。")
        
        # 加载模型
        print(f"正在加载模型: {model_path} ...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            quantization='awq',
            trust_remote_code=True,
            # gpu_memory_utilization=0.9
        )
        print("模型加载完成。")
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # 系统提示词
        self.system_prompt = (
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
    
    def split_text(self, text: str) -> dict:
        """
        对单个文本进行分割
        
        Args:
            text (str): 需要分割的文本
            
        Returns:
            dict: 包含原始文本、模型输出和分割后句子的字典
                {
                    "original": str,
                    "llm_response": str,
                    "split_sentences": List[str]
                }
        """
        # 构建消息列表
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"[input] {text}"}
        ]
        
        # 使用模型的 chat 方法生成
        outputs = self.llm.chat(messages, self.sampling_params, use_tqdm=False)
        
        # 提取生成的文本
        llm_response = outputs[0].outputs[0].text
        
        # 解析输出
        split_sentences = self.parse_llm_output(llm_response)
        
        return {
            "original": text,
            "llm_response": llm_response,
            "split_sentences": split_sentences
        }


def compute_similarity_ranks(
    split_texts_embeddings_queries: List[List[np.ndarray]],
    split_texts_embeddings_documents: List[List[np.ndarray]],
    device: str = "cuda:0",
    batch_size: int = 1000
) -> List[List[List[np.ndarray]]]:
    """
    使用GPU批量计算每个文本片段与所有其他视频的文本片段的余弦相似度，并进行排序
    
    Args:
        split_texts_embeddings_queries: 查询embeddings
            shape: [视频个数][文本描述个数] -> 每个元素是 [分割的文本个数, embedding_dim] 的数组
        split_texts_embeddings_documents: 文档embeddings
            shape: [视频个数][文本描述个数] -> 每个元素是 [分割的文本个数, embedding_dim] 的数组
        device: 使用的GPU设备，默认 "cuda:0"
        batch_size: 批处理大小，控制每次在GPU上处理的目标向量数量
        
    Returns:
        List[List[List[np.ndarray]]]: 相似度排名（紧凑存储）
            [视频][文本描述][分割文本] -> 每个元素是 numpy 结构化数组
            dtype: [('video_idx', 'u4'), ('text_idx', 'u4'), ('split_idx', 'u4'), ('similarity', 'f2')]
            已按相似度降序排序
    """
    num_videos = len(split_texts_embeddings_queries)
    similarity_ranks = []
    
    print(f"\n开始计算 {num_videos} 个视频的相似度排名（使用GPU: {device}）...")
    
    # 首先构建所有目标文档的索引和embeddings
    print("预处理：构建目标文档索引...")
    all_target_embeddings = []
    all_target_indices = []  # 存储 (video_idx, text_idx, split_idx)
    
    for video_idx in range(num_videos):
        num_texts = len(split_texts_embeddings_documents[video_idx])
        for text_idx in range(num_texts):
            target_documents = split_texts_embeddings_documents[video_idx][text_idx]
            num_splits = target_documents.shape[0]
            for split_idx in range(num_splits):
                all_target_embeddings.append(target_documents[split_idx])
                all_target_indices.append((video_idx, text_idx, split_idx))
    
    # 将所有目标embeddings转换为tensor并移到GPU
    all_target_tensor = torch.from_numpy(np.stack(all_target_embeddings)).float().to(device)
    print(f"目标文档总数: {len(all_target_embeddings)}, shape: {all_target_tensor.shape}")
    
    # 归一化目标向量（用于余弦相似度计算）
    all_target_tensor = F.normalize(all_target_tensor, p=2, dim=1)
    
    # 定义紧凑的数据类型：uint32存储索引，float16存储相似度
    similarity_dtype = np.dtype([
        ('video_idx', np.uint32),
        ('text_idx', np.uint32),
        ('split_idx', np.uint32),
        ('similarity', np.float16)
    ])
    
    # 遍历每个视频
    for video_idx in tqdm.tqdm(range(num_videos), desc="处理视频"):
        video_ranks = []
        
        # 遍历当前视频的每个文本描述
        num_texts = len(split_texts_embeddings_queries[video_idx])
        for text_idx in range(num_texts):
            text_ranks = []
            
            # 获取当前文本描述的embeddings (queries)
            current_queries = split_texts_embeddings_queries[video_idx][text_idx]
            num_splits = current_queries.shape[0]
            
            # 遍历当前文本描述的每个分割片段
            for split_idx in range(num_splits):
                # 获取当前查询向量并移到GPU
                query_vector = torch.from_numpy(current_queries[split_idx:split_idx+1]).float().to(device)
                query_vector = F.normalize(query_vector, p=2, dim=1)  #debug 归一化 这个不一定需要
                
                # 批量计算与所有目标的相似度
                similarities_list = []
                
                # 分批处理以避免内存溢出
                for i in range(0, len(all_target_embeddings), batch_size):
                    batch_targets = all_target_tensor[i:i+batch_size]
                    # 计算余弦相似度 (使用点积，因为向量已归一化)
                    batch_similarities = torch.mm(query_vector, batch_targets.t()).squeeze(0)
                    similarities_list.append(batch_similarities.cpu())
                
                # 合并所有批次的相似度，转换为float16
                all_similarities = torch.cat(similarities_list).numpy().astype(np.float16)
                
                # 构建紧凑的结构化数组（排除当前视频）
                valid_indices = [i for i, (v, _, _) in enumerate(all_target_indices) if v != video_idx]
                num_valid = len(valid_indices)
                
                # 创建结构化数组
                similarities_array = np.empty(num_valid, dtype=similarity_dtype)
                for idx, i in enumerate(valid_indices):
                    other_video_idx, other_text_idx, other_split_idx = all_target_indices[i]
                    similarities_array[idx] = (
                        other_video_idx,
                        other_text_idx,
                        other_split_idx,
                        all_similarities[i]
                    )
                
                # 按相似度降序排序
                similarities_array = np.sort(similarities_array, order='similarity')[::-1]
                
                text_ranks.append(similarities_array)
            
            video_ranks.append(text_ranks)
        
        similarity_ranks.append(video_ranks)
    
    print("相似度排名计算完成！")
    return similarity_ranks


def get_top_k_similar(
    similarity_ranks: List[List[List[np.ndarray]]],
    video_idx: int,
    text_idx: int,
    split_idx: int,
    k: int = 10
) -> np.ndarray:
    """
    获取指定文本片段的top-k最相似的片段
    
    Args:
        similarity_ranks: compute_similarity_ranks返回的similarity_ranks
        video_idx: 视频索引
        text_idx: 文本描述索引
        split_idx: 分割片段索引
        k: 返回前k个最相似的
        
    Returns:
        np.ndarray: top-k相似的片段信息（结构化数组）
            dtype: [('video_idx', 'u4'), ('text_idx', 'u4'), ('split_idx', 'u4'), ('similarity', 'f2')]
    """
    try:
        ranks = similarity_ranks[video_idx][text_idx][split_idx]
        return ranks[:k]
    except (IndexError, TypeError):
        print(f"无效的索引: video_idx={video_idx}, text_idx={text_idx}, split_idx={split_idx}")
        return np.array([], dtype=[('video_idx', np.uint32), ('text_idx', np.uint32), 
                                    ('split_idx', np.uint32), ('similarity', np.float16)])


if __name__ == "__main__":    
    process = False
    if process:
        # 如果需要处理 train.pt 文件中的数据
        data = torch.load("train.pt", weights_only=False)
        # 初始化 TextSplitter
        model = SentenceTransformer("/home/yueyi/largemodel/Qwen3-Embedding-8B", device="cuda:4")
        print("成功加载embedding模型")
        splitter = TextSplitter()
        print("成功加载llm模型")
        # 定义需要被保存的变量
        split_texts = []
        split_texts_embeddings_queries = []
        split_texts_embeddings_documents = []

        for i in tqdm.tqdm(range(len(data.motion_labels_raw))):
            texts = []
            embeddings_queries = []
            embeddings_documents = []
            for txt in data.motion_labels_raw[i]:
                result = splitter.split_text(txt)
                texts.append(result["split_sentences"])
                embeddings_queries.append(model.encode(result["split_sentences"], prompt_name="query"))
                embeddings_documents.append(model.encode(result["split_sentences"]))
            split_texts.append(texts)
            split_texts_embeddings_queries.append(embeddings_queries)
            split_texts_embeddings_documents.append(embeddings_documents)
        
        data.split_texts = split_texts
        data.split_texts_embeddings_queries = split_texts_embeddings_queries
        data.split_texts_embeddings_documents = split_texts_embeddings_documents
        torch.save(data, "train_split.pt")
        print("成功保存分割数据到 train_split.pt")
    else:
        data = torch.load("train_split.pt", weights_only=False)
        split_texts = data.split_texts
        split_texts_embeddings_queries = data.split_texts_embeddings_queries
        split_texts_embeddings_documents = data.split_texts_embeddings_documents
        print("成功加载已处理数据")
    
    # 计算相似度排名
    compute_rank = True  # 设置为True来计算rank，False来跳过
    if compute_rank:
        print("\n" + "="*60)
        print("开始计算相似度排名...")
        print("="*60)
        
        similarity_ranks = compute_similarity_ranks(
            split_texts_embeddings_queries,
            split_texts_embeddings_documents,
            device="cuda:0",  # 可以根据需要修改使用的GPU
            batch_size=1000   # 可以根据GPU内存调整批次大小
        )
        
        # 将相似度排名添加到数据对象
        data.similarity_ranks = similarity_ranks
        
        # 保存到新文件，避免覆盖原始文件
        output_filename = "train_split_with_ranks.pt"
        torch.save(data, output_filename)
        print(f"\n成功保存带有相似度排名的数据到: {output_filename}")
        
        # 显示示例
        print("\n" + "="*60)
        print("示例：视频0，文本0，片段0 的 top-10 相似片段：")
        print("="*60)
        top_k = get_top_k_similar(similarity_ranks, 0, 0, 0, k=10)
        for rank, item in enumerate(top_k, 1):
            print(f"Rank {rank:3d}: 视频{item['video_idx']:4d}, 文本{item['text_idx']}, "
                  f"片段{item['split_idx']}, 相似度={float(item['similarity']):.4f}")
        
        print(f"\n原始数据保留在: train_split.pt")
        print(f"带排名数据保存在: {output_filename}")
    else:
        print("\n跳过相似度排名计算（compute_rank=False）")