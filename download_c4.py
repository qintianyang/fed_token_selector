from datasets import load_dataset

# 下载 C4 数据集 (英文部分)
# 这可能需要一些时间，并且会占用大量磁盘空间
print("开始下载 C4 数据集...")
dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
print("数据集下载/加载完成。")

# 创建一个目录来保存数据
import os
os.makedirs('data', exist_ok=True)

# 从数据集中取样并保存到文件
output_file = '/home/qty/code/federated_token_selector/data/c4_sample.jsonl'
num_samples = 10000  # 您想要保存的样本数量

print(f"正在将 {num_samples} 个样本写入到 {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    count = 0
    for example in dataset:
        if count >= num_samples:
            break
        # 将每个样本的文本内容写入文件，每行一个
        f.write(example['text'].strip() + '\n')
        count += 1
        if count % 1000 == 0:
            print(f"已处理 {count}/{num_samples} 个样本...")

print("数据采样和保存完成。")