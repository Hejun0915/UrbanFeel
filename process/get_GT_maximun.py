import json

# ===================== 【在这里输入你的文件路径】 =====================
# 把你的 JSON 文件路径填在这里 (支持绝对路径和相对路径)
INPUT_JSON_PATH = r"D:\Desktop\Work\Project\UrbanFeel\src\json\gemini-2.5-pro_ChangeTypeRecongition.json"  

# 处理后保存的新文件名 (会自动生成，不会覆盖原文件)
OUTPUT_JSON_PATH = r"D:\Desktop\Work\Project\UrbanFeel\src\json\SceneLevelChangeRecognition.json"  
# ====================================================================

def remove_response_field():
    try:
        # 1. 读取原始 JSON 文件
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. 遍历每一条数据，删除 'response' 字段
        for item in data:
            # 如果 'response' 字段存在，就删除它
            if 'response' in item:
                del item['response']

        # 3. 保存处理后的新文件
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            # indent=4 让格式更好看，ensure_ascii=False 支持中文
            json.dump(data, f, indent=4, ensure_ascii=False)

        # 打印成功信息
        print(f"✅ 处理完成！")
        print(f"📥 原始文件：{INPUT_JSON_PATH}")
        print(f"📤 新文件已保存为：{OUTPUT_JSON_PATH}")
        print(f"🗑️  已成功删除所有 'response' 字段")

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{INPUT_JSON_PATH}'，请检查路径是否正确！")
    except Exception as e:
        print(f"❌ 发生错误：{e}")

if __name__ == "__main__":
    remove_response_field()