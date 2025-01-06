import argparse
from datasets import load_dataset
from tqdm import tqdm
from tools.hash import ImageDeduplicator
from tools.template import format_vanilla, format_reasoning_enhance_question, format_check_question
from paddlemix.datacopilot.nn import llms


def deduplicate_dataset(dataset, threshold=5, hash_size=8, batch_size=1000):
    """
    对Huggingface数据集进行图片去重
    
    Args:
        dataset: Huggingface dataset对象
        threshold: 汉明距离阈值，默认5
        hash_size: 哈希大小，默认8
        batch_size: 批处理大小，默认1000
        
    Returns:
        去重后的数据集
    """
    
    deduplicator = ImageDeduplicator(hash_size=hash_size)
    
    # 计算所有图片的哈希值
    all_hashes = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Computing hashes"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        batch_hashes = [deduplicator.compute_dhash(item['image']) for item in batch]
        all_hashes.extend(batch_hashes)
    
    # 找出唯一图片的索引
    unique_indices = set(range(len(all_hashes)))
    duplicates = set()
    
    print("Finding duplicates...")
    for i in tqdm(range(len(all_hashes))):
        if i in duplicates:
            continue
        for j in range(i + 1, len(all_hashes)):
            if j in duplicates:
                continue
            if deduplicator.hamming_distance(all_hashes[i], all_hashes[j]) <= threshold:
                duplicates.add(j)
    
    unique_indices = list(unique_indices - duplicates)
    
    # 创建新数据集
    deduplicated_dataset = dataset.select(unique_indices)
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"After deduplication: {len(deduplicated_dataset)}")
    print(f"Removed {len(dataset) - len(deduplicated_dataset)} duplicates")
    
    return deduplicated_dataset

def noise_detection(dataset, noise_ratio):
    vanilla_model = llms.ErnieEval(model_name="ernie-4.0", access_token="")
    reasoning_enhance_model = llms.ErnieEval(model_name="ernie-4.0", access_token="")
    check_model = llms.ErnieEval(model_name="ernie-speed-128k", access_token="")
    
    vanilla_eval = Eval(samples=dataset)
    vanilla_eval.evaluate(vanilla_model, format_fn = format_vanilla)
    vanilla_predict = vanilla_eval.get_results()
    
    """
    Reasoning-enhanced inference
    """
    reasoning_eval = Eval(samples=dataset)
    reasoning_eval.evaluate(reasoning_enhance_model, format_fn = format_reasoning_enhance_question)
    reasoning_predict = reasoning_eval.get_results()
    
    """
    Data Split
    """
    clean_samples = []
    noisy_samples = []

    data_packed = []
    for i in range(len(vanilla_predict)):
        data_ = dataset[i]
        potential_answer = data_['answer']
        vanilla_pred = vanilla_predict[i]['PredAnswer']
        reasoning_pred = reasoning_predict[i]['PredAnswer']

        data_['vanilla_prediction'] = vanilla_pred
        data_['image'] = data_['image']
        data_['reasoning_prediction'] = reasoning_pred
        data_['potential_answer'] = potential_answer

        template = format_check_question(data_)
        out = check_model.predict(template)
        
        if "Y" in out:
            data_['PseudoLabel'] = potential_answer
            data_['clean_flag'] = 1
            clean_samples.append(data_)
        else:
            data_['clean_flag'] = 0
            noisy_samples.append(data_)
    return clean_samples, noisy_samples
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Noisy Free Fine-tuning')
    # parser.add_argument('--task', type=str, default='mmlu', help='Task name')
    # parser.add_argument('--model', type=str, default='llama3.1-8b', help='Model name')
    # parser.add_argument('--noise_ratio', type=int, default=50, help='Noise ratio')
    # parser.add_argument('--base_url', type=str, default='http://localhost:8002/v1', help='Base URL')
    
    parser.add_argument('--dataset_path', type=str, default='liuhaotian/LLaVA-1.5-Visual-Instruction-665K', help='dataset name')
    
    args = parser.parse_args()
    
    # step0 : Load dataset
    dataset = load_dataset(args.dataset_path)
    train_dataset = dataset['train']
    
    # step1 :  Remove duplicate images using function
    dedu_dataset = deduplicate_dataset(dataset)

    # step2 : Noise detection
    clean, noisy = noise_detection(dedu_dataset, args.noise_ratio)
    
