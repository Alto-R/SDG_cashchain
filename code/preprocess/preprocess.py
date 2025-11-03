import pandas as pd
import json
from tqdm import tqdm

# --- 配置 ---

# 这是上一个脚本 (mapping.py) 的输出文件
INPUT_JSONL_FILE = 'sdg_target_mappings_deepseek_async_output.jsonl'

# 这是我们想要生成的最终CSV文件
OUTPUT_CSV_FILE = 'all_mappings.csv'

# --- 脚本 ---

def process_jsonl_to_csv():
    """
    读取JSON Lines文件，将其展平，并保存为CSV。
    """
    processed_rows = []
    error_count = 0
    success_count = 0
    
    print(f"正在从 {INPUT_JSONL_FILE} 读取数据...")

    try:
        with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # 使用tqdm显示处理进度
        for line in tqdm(lines, desc="正在处理JSONL记录"):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # 检查是否为API调用失败的记录
                if "error" in data:
                    error_count += 1
                    continue
                
                industry_name = data.get("industry_name")
                mappings = data.get("sdg_target_mappings") # 这是一个列表
                
                if not industry_name or not mappings:
                    # 可能是无效的记录或没有映射结果
                    continue
                
                success_count += 1
                
                # 核心逻辑：遍历mappings列表，为每个子目标创建一行
                for mapping in mappings:
                    flat_row = {
                        "industry_name": industry_name,
                        "sdg_target_id": mapping.get("sdg_target_id"),
                        "impact_type": mapping.get("impact_type"),
                        "relevance_score": mapping.get("relevance_score"),
                        "rationale": mapping.get("rationale"),
                        "sdg_target_description": mapping.get("sdg_target_description")
                    }
                    processed_rows.append(flat_row)

            except json.JSONDecodeError:
                print(f"\n警告：跳过一行无法解析的JSON: {line[:100]}...")
                error_count += 1
                
        if not processed_rows:
            print("错误：没有找到任何有效的映射数据。请检查JSONL文件是否正确。")
            return

        print(f"\n处理完毕。成功解析 {success_count} 条行业记录，跳过 {error_count} 条错误记录。")
        
        # 将展平的行列表转换为Pandas DataFrame
        df = pd.DataFrame(processed_rows)
        
        # 保存为CSV
        # 使用 utf-8-sig 编码确保Excel能正确打开包含中文的CSV
        df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
        
        print(f"成功！已将所有映射关系保存到: {OUTPUT_CSV_FILE}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {INPUT_JSONL_FILE}")
        print("请先运行 sdg_mapping.py 来生成该文件。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")

if __name__ == "__main__":
    process_jsonl_to_csv()
