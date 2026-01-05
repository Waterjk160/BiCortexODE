import os
import pandas as pd
import random

# 数据总目录
data_dir = "/public_bme2/bme-dgshen/caoshui2024/Qilu_1400data/Step05_sMRI_Signal_Surfaces_Metric"

# 获取所有患者文件夹
subject_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
subject_list.sort()  # 可选：按名字排序
random.seed(42)      # 固定随机种子，保证可复现
random.shuffle(subject_list)

# 划分比例
train_ratio = 0.7
valid_ratio = 0.15
test_ratio  = 0.15

n_total = len(subject_list)
n_train = int(n_total * train_ratio)
n_valid = int(n_total * valid_ratio)
n_test = n_total - n_train - n_valid

train_subjects = subject_list[:n_train]
valid_subjects = subject_list[n_train:n_train+n_valid]
test_subjects = subject_list[n_train+n_valid:]

# 构建 DataFrame
df = pd.DataFrame({
    "subject_id": train_subjects + valid_subjects + test_subjects,
    "split": ["train"]*len(train_subjects) + ["valid"]*len(valid_subjects) + ["test"]*len(test_subjects)
})

# 保存 CSV
out="/home_data/home/caoshui2024/DeepLearning_BrainMLSR/CortexODE/data"
csv_path = os.path.join(out, "subject_split.csv")
df.to_csv(csv_path, index=False)
print(f"Saved split CSV to {csv_path}")
