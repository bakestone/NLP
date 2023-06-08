import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

# 从文件中读取数据并进行预处理
zh_data = pd.read_csv("data/news-commentary-v13.zh-en.zh", delimiter="\n", header=None, names=["中文"])
en_data = pd.read_csv("data/news-commentary-v13.zh-en.en", delimiter="\n", header=None, names=["English"])    # 使用WMT数据创建这个文件

# 将中英文数据合并为一个DataFrame
data = pd.concat([zh_data, en_data], axis=1)

data.to_csv("data2/new_en-zh.txt", sep='\t', index=False)

# 将数据集划分为训练集、验证集和测试集
train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)

# 将合并后的 DataFrame 保存为 CSV 文件，并指定制表符作为分隔符
# train_data.to_csv("data2/re_train.txt", sep='\t', index=False)
# valid_data.to_csv("data2/re_valid.txt", sep='\t', index=False)
# test_data.to_csv("data2/re_test.txt", sep='\t', index=False)