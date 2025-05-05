import sys
import openai
import os

# 设置 OpenAI API 密钥
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_song_info_and_tags(song_name, artist):
    """使用 LLM 获取歌曲标签和相关信息"""
    try:
        print(f"\n正在分析歌曲: {song_name} - {artist} ...")
        query = f"""请分析歌曲 '{song_name}' 的演唱者 '{artist}' 的详细信息。
请按如下格式输出：

1. 基本信息：
   - 创作年份：
   - 所属专辑：
   - 发行公司：

2. 音乐平台分类：
   - Apple Music 分类（如：流行、摇滚、嘻哈、电子、古典、爵士、R&B、民谣、世界音乐等）：
   - 网易云音乐分类（如：华语、欧美、日语、韩语、轻音乐、ACG、影视原声等）：
   - QQ音乐分类（如：流行、摇滚、民谣、电子、说唱、国风、影视等）：

3. 音乐特征标签：
   - 传统音乐分类（如：古典、民族、宗教、等）：
   - 乐器标签（如：钢琴、吉他、鼓、等）：
   - 情绪标签（如：欢快、忧伤、平静、等）：
   - 风格标签（如：流行、摇滚、民谣、等）：
   - 场景标签（如：夜晚、雨天、旅行、运动、工作、学习、聚会、节日等）：
   - 节奏标签（如：慢速、中速、快速、等）：
   - 其他标签（如：励志、治愈、等）：

4. 创作背景：
   - 创作灵感：
   - 有趣故事：

请确保每个分类下至少提供2-6个标签，创作背景请控制在100字以内。
"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "你是一个音乐分析专家，擅长分析歌曲的风格、情感特征、乐器和背景故事，熟悉各大音乐平台的分类标准。"},
                {"role": "user", "content": query}
            ],
            max_tokens=500
        )
        response_text = response.choices[0].message.content.strip()
        return response_text
    except Exception as e:
        return f"获取信息时出错: {e}"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python get_song_info.py 歌手 歌名")
        sys.exit(1)
    artist = sys.argv[1]
    song_name = sys.argv[2]
    result = get_song_info_and_tags(song_name, artist)
    print("\n分析结果：")
    print(result) 