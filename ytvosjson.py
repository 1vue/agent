import os
import json

root = '../../dataset/ref-youtube'

# 1. 读取原始 valid+test 的元数据（完整数据）
valid_meta_file = os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json')
with open(valid_meta_file, 'r') as f:
    valid_full_data = json.load(f)  # 包含 'videos' 等字段
valid_videos_dict = valid_full_data['videos']  # 字典：视频名 -> 详细数据

# 2. 读取 test 的元数据，仅获取视频名集合
test_meta_file = os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json')
with open(test_meta_file, 'r') as f:
    test_data = json.load(f)
test_videos = set(test_data['videos'].keys())

# 3. 过滤：只保留不在 test 中的视频
filtered_videos_dict = {k: v for k, v in valid_videos_dict.items() if k not in test_videos}

# 4. 构建新的 JSON 结构（与原结构相同，仅 videos 字段被过滤）
filtered_full_data = {
    "videos": filtered_videos_dict,
    # 如果原 JSON 还有其他顶级字段（如 "expressions" 等），也一并保留
    # 这里假设原文件只有 "videos" 字段，如果有其他字段请按需添加
}
# 若原文件还有其他字段，可更通用地复制：
# filtered_full_data = {key: value for key, value in valid_full_data.items() if key != "videos"}
# filtered_full_data["videos"] = filtered_videos_dict

# 5. 保存到 root/meta_expressions/valid/valid_filtered.json
output_dir = os.path.join(root, 'meta_expressions', 'valid')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'valid_filtered.json')
with open(output_path, 'w') as f:
    json.dump(filtered_full_data, f, indent=4)

print(f"已保存过滤后的有效视频元数据到：{output_path}")

# 6. 输出统计信息
print("\n=== 视频数量统计 ===")
print(f"原始 valid 文件中的视频数: {len(valid_videos_dict)}")
print(f"test 文件中的视频数: {len(test_videos)}")
print(f"过滤后剩余的有效视频数: {len(filtered_videos_dict)}")
print(f"移除的视频数（属于 test）: {len(valid_videos_dict) - len(filtered_videos_dict)}")

# 验证：所有 test 视频都被移除
remaining_test_videos = set(filtered_videos_dict.keys()) & test_videos
assert len(remaining_test_videos) == 0, f"错误：仍有 test 视频残留: {remaining_test_videos}"
print("\n验证通过：过滤后的数据中不包含任何 test 视频。")