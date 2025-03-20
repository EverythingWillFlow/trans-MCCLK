# 读取原始文件
input_file = './ratings_final.txt'  # 输入文件名
output_file = 'train.txt'  # 输出文件名

# 创建一个字典来存储合并后的数据
merged_data = {}

# 读取输入文件
with open(input_file, 'r') as file:
    for line in file:
        # 分割每一行数据
        parts = line.strip().split()
        if len(parts) < 3:
            continue  # 忽略不完整的行

        user_id = parts[0]  # 第一列是用户ID
        item_id = parts[1]  # 第二列是物品ID
        rating = parts[2]   # 第三列是评分（0或1）

        # 如果用户ID已经在字典中，将物品ID添加到对应的列表中
        if user_id in merged_data:
            merged_data[user_id].append(item_id)
        else:
            merged_data[user_id] = [item_id]

# 将合并后的数据写入输出文件
with open(output_file, 'w') as file:
    for user_id, item_ids in merged_data.items():
        # 将用户ID和物品ID列表用空格连接，并写入文件
        file.write(f"{user_id} {' '.join(item_ids)}\n")

print(f"转换完成，结果已保存到 {output_file}")