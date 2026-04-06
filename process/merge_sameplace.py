import json

# ===================== 【在这里指定你的两个文件路径】 =====================
FILE_A_PATH = r"D:\Desktop\Work\Project\UrbanFeel\src\json\ColocationRecognition_true.json"  # 你的第一个JSON文件（SamePlace=True）
FILE_B_PATH = r"D:\Desktop\Work\Project\UrbanFeel\src\json\ColocationRecognition_false.json"  # 你的第二个JSON文件（SamePlace=False）
OUTPUT_PATH = r"D:\Desktop\Work\Project\UrbanFeel\src\json\ColocationRecognition.json"  # 合并后输出的文件
# ==========================================================================

def merge_json_with_sameplace():
    # 1. 读取文件A
    with open(FILE_A_PATH, "r", encoding="utf-8") as f:
        data_a = json.load(f)

    # 2. 读取文件B
    with open(FILE_B_PATH, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    merged = []
    new_id = 0

    # 3. 处理文件A：添加 SamePlace=True，重新编号id
    for item in data_a:
        new_item = item.copy()
        new_item["id"] = new_id
        new_item["SamePlace"] = True
        merged.append(new_item)
        new_id += 1

    # 4. 处理文件B：添加 SamePlace=False，重新编号id
    for item in data_b:
        new_item = item.copy()
        new_item["id"] = new_id
        new_item["SamePlace"] = False
        merged.append(new_item)
        new_id += 1

    # 5. 保存合并后的JSON
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)

    print(f"✅ 合并完成！")
    print(f"📄 文件A条目数：{len(data_a)}")
    print(f"📄 文件B条目数：{len(data_b)}")
    print(f"📦 总合并条目数：{len(merged)}")
    print(f"💾 已保存到：{OUTPUT_PATH}")

if __name__ == "__main__":
    merge_json_with_sameplace()