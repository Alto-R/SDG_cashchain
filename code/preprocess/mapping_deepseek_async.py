from openai import AsyncOpenAI
import pandas as pd
import json
import asyncio
from tqdm.asyncio import tqdm as async_tqdm

# --- 配置 ---
SILICONFLOW_API_KEY = ''
client = AsyncOpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url="https://api.siliconflow.cn/v1"
)

# 并发控制
# DeepSeek-V3.1-Terminus 限制: RPM=30,000, TPM=5,000,000
# 推荐并发数：50-100（基于TPM限制和平均请求token数）
MAX_CONCURRENT_REQUESTS = 50  # 最大并发请求数，可根据实际情况调整至100

# 行业数据在一个CSV文件中
INDUSTRY_DATA_FILE = 'industry.csv'
# 假设您的CSV中包含一个名为 'industry_name_jp' 的列
JAPANESE_INDUSTRY_COLUMN_NAME = 'industry'

OUTPUT_FILE = 'sdg_target_mappings_deepseek_async_output.jsonl'  # 输出文件名已更新

# --- 提示词模板 (169子目标,仅日文名称版) ---
PROMPT_TEMPLATE = """
### 角色 (Role) ###
你是一名联合国可持续发展目标 (SDG) 框架的顶级专家，你对17个目标下的169个具体子目标 (Targets) 了如指掌。你的任务是严格、客观地将给定的细分行业映射到 **具体的SDG子目标编号** (例如 "1.1", "7.2", "7.a")。

### 核心指令 (Core Instruction) ###
你将收到一个 **日文** 的【行业名称】。**你没有行业描述。**
你必须 **仅** 基于对这个日文行业名称的理解，推断其核心经济活动，并评估其对 **169个SDG子目标** 的 **(1) 积极贡献** 和 **(2) 消极影响**。

### 评估标准 (Criteria) ###
你的评估必须是具体的、可辩护的，并直接关联到子目标的官方定义。

### 思考过程 (Chain of Thought) - [这是成功的关键] ###
在给出最终JSON输出之前，请你先完成一步一步的思考：
1.  **推断活动 (Inferred Activity Analysis)**：这个 **日文行业名称** (例如：「太陽光パネル製造」) 暗示的核心经济活动是什么？（例如：推断为"太阳能电池板制造"，核心是制造光伏设备，增加可再生能源供应）
2.  **主要目标识别 (High-level Goal Identification)**：该推断活动首先关联到哪几个 **高层SDG目标**？（例如：SDG 7, SDG 9, SDG 13, SDG 12）
3.  **子目标精确定位 (Specific Target Mapping)**：
    * **对于 SDG 7**：该活动如何贡献于SDG 7？（例如：推断与 "7.2 - 增加可再生能源比例" 最相关）
    * **对于 SDG 9**：该活动如何贡献于SDG 9？（例如：推断与 "9.4 - 升级基础设施以实现清洁技术" 相关）
    * **对于 SDG 13**：该活动如何贡献于SDG 13？（例如：推断与 "13.2 - 将气候变化措施纳入政策" 相关）
    * **对于 SDG 12**：该活动是否有负面影响？（例如：推断制造过程可能与 "12.5 - 减少废物产生" 相悖）
4.  **得分与理由 (Scoring and Rationale)**：基于上述分析，为每个识别出的子目标分配相关性得分 (0-1) 和理由。

### 输出格式 (Output Format) - [必须严格要求JSON] ###
请你 **只** 返回一个格式严格的JSON对象，不要包含JSON区块之外的任何解释性文字。JSON结构如下：

{{
  "industry_name": "【输入的日文行业名称】",
  "analysis_chain_of_thought": {{
    "inferred_activity_analysis": "【你的推断活动分析】",
    "high_level_goal_identification": "【你的高层目标识别】",
    "specific_target_mapping": "【你详细的子目标精确定位思考过程】"
  }},
  "sdg_target_mappings": [
    {{
      "sdg_target_id": "7.2",
      "sdg_target_description": "(By 2030, increase substantially the share of renewable energy in the global energy mix)",
      "impact_type": "Positive",
      "relevance_score": 0.9,
      "rationale": "该行业的核心产品（太阳能电池板）直接用于增加可再生能源在全球能源结构中的比例。"
    }},
    {{
      "sdg_target_id": "9.4",
      "sdg_target_description": "(By 2030, upgrade infrastructure and retrofit industries to make them sustainable... with greater adoption of clean and environmentally sound technologies)",
      "impact_type": "Positive",
      "relevance_score": 0.7,
      "rationale": "太阳能技术是实现工业和基础设施可持续性改造所需的关键清洁技术之一。"
    }}
  ]
}}

### 任务开始 (Task Start) ###
请处理以下行业：
**行业名称 (日文)**: {industry_name_jp}
"""


async def get_sdg_mapping(industry_name_jp, semaphore, retry_count=3):
    """
    异步调用 SiliconFlow DeepSeek-V3.1 API 获取单个行业的SDG子目标映射。
    使用信号量控制并发数，支持重试机制。
    """
    prompt = PROMPT_TEMPLATE.format(industry_name_jp=industry_name_jp)

    async with semaphore:  # 控制并发数
        for attempt in range(retry_count):
            try:
                # 使用异步 OpenAI SDK 调用 SiliconFlow DeepSeek-V3.1
                response = await client.chat.completions.create(
                    model="Pro/deepseek-ai/DeepSeek-V3.1-Terminus",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    top_p=0.95,
                    max_tokens=32768,
                    stream=False,
                    response_format={"type": "json_object"}  # 强制JSON输出
                )

                raw_text = response.choices[0].message.content

                # 清理常见的Markdown标记
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:-3].strip()
                elif raw_text.startswith("```"):
                    raw_text = raw_text[3:-3].strip()

                data = json.loads(raw_text)
                return {"success": True, "data": data, "industry_name": industry_name_jp}

            except json.JSONDecodeError as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(1)  # 重试前等待
                    continue
                try:
                    raw_response = response.choices[0].message.content
                    return {
                        "success": False,
                        "error": "JSONDecodeError",
                        "raw_response": raw_response[:500],
                        "industry_name": industry_name_jp
                    }
                except:
                    return {
                        "success": False,
                        "error": "JSONDecodeError",
                        "raw_response": str(e),
                        "industry_name": industry_name_jp
                    }

            except Exception as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2)  # 重试前等待更长时间
                    continue
                return {
                    "success": False,
                    "error": str(e),
                    "industry_name": industry_name_jp
                }


async def process_industries(industries):
    """
    并发处理所有行业数据
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # 创建所有任务
    tasks = [
        get_sdg_mapping(industry_name, semaphore)
        for industry_name in industries
    ]

    # 使用 tqdm 显示进度，并发执行所有任务
    results = []
    for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="正在映射行业到169子目标"):
        result = await coro
        results.append(result)

    return results


async def main_async():
    """
    异步主函数：读取行业列表，并发调用API，保存结果。
    """
    try:
        df = pd.read_csv(INDUSTRY_DATA_FILE)
        if JAPANESE_INDUSTRY_COLUMN_NAME not in df.columns:
            print(f"错误: 你的CSV文件 '{INDUSTRY_DATA_FILE}' 中没有找到列 '{JAPANESE_INDUSTRY_COLUMN_NAME}'")
            print("请确保CSV文件包含该列，或者在脚本中更新 JAPANESE_INDUSTRY_COLUMN_NAME 变量。")
            return
        print(f"成功读取 {len(df)} 个行业，来自 {INDUSTRY_DATA_FILE}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {INDUSTRY_DATA_FILE}")
        print("将使用示例数据...")
        df = pd.DataFrame([
            {JAPANESE_INDUSTRY_COLUMN_NAME: "太陽光パネル製造"},
            {JAPANESE_INDUSTRY_COLUMN_NAME: "石炭採掘"},
            {JAPANESE_INDUSTRY_COLUMN_NAME: "モバイル決済サービス"}
        ])

    # 过滤空值
    industries = df[JAPANESE_INDUSTRY_COLUMN_NAME].dropna().tolist()

    if not industries:
        print("没有找到有效的行业数据")
        return

    print(f"\n开始并发处理 {len(industries)} 个行业...")
    print(f"最大并发数: {MAX_CONCURRENT_REQUESTS}")

    # 并发处理所有行业
    results = await process_industries(industries)

    # 写入结果
    success_count = 0
    fail_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            if result["success"]:
                f.write(json.dumps(result["data"], ensure_ascii=False) + '\n')
                success_count += 1
            else:
                # 保存错误信息
                error_data = {
                    "error": result["error"],
                    "industry_name": result["industry_name"],
                    "raw_response": result.get("raw_response", "")
                }
                f.write(json.dumps(error_data, ensure_ascii=False) + '\n')
                fail_count += 1

    print(f"\n处理完成！")
    print(f"✓ 成功: {success_count}")
    print(f"✗ 失败: {fail_count}")
    print(f"所有结果已保存到 {OUTPUT_FILE}")


def main():
    """
    同步入口函数，用于运行异步主函数
    """
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
