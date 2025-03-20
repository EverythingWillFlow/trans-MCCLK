import random

# 读取 train.txt 文件
with open('./train.txt', 'r') as file:
    lines = file.readlines()

# 计算总行数和测试集行数
total_lines = len(lines)
test_lines_count = int(total_lines * 0.2)

# 随机选取 20% 的行作为测试集
test_lines = random.sample(lines, test_lines_count)

# 将选中的行保存为 test.txt
with open('test.txt', 'w') as file:
    file.writelines(test_lines)

print(f"已从 train.txt 中随机选取 {test_lines_count} 行作为测试集，并保存为 test.txt")