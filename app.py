
import streamlit as st
import requests
import openai
import logging
from bs4 import BeautifulSoup
from prompt_config import SUMMARY_PROMPT, SEARCH_QUERY_PROMPT, EVALUATION_PROMPT
from default_config import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_MODEL_NAME

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- UI: 侧边栏配置 ---
with st.sidebar:
    st.header("⚙️ AI 配置")
    
    # 使用 session_state 来持久化用户输入
    if 'api_key' not in st.session_state:
        st.session_state.api_key = DEFAULT_API_KEY
    if 'base_url' not in st.session_state:
        st.session_state.base_url = DEFAULT_BASE_URL
    if 'model_name' not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL_NAME

    st.session_state.api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
    st.session_state.base_url = st.text_input("Base URL (可选)", value=st.session_state.base_url)
    st.session_state.model_name = st.text_input("模型名称", value=st.session_state.model_name)

    # 新增：单次搜索结果数量配置
    if 'max_search_results_per_iteration' not in st.session_state:
        st.session_state.max_search_results_per_iteration = 5
    st.session_state.max_search_results_per_iteration = st.slider(
        "单次搜索结果数量",
        min_value=1,
        max_value=15,
        value=st.session_state.max_search_results_per_iteration,
        step=1,
        help="每次迭代搜索时获取的网页结果数量。"
    )
    
    st.info("配置信息将在此会话中保持有效。")

# --- 核心逻辑 ---
def search_the_web(query: str, max_results: int = 5):
    """使用 Bing 搜索网页并返回最佳结果。"""
    logging.info(f"开始为查询 '{query}' 进行 Bing 搜索...")
    try:
        search_url = f"https://www.bing.com/search?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        search_items = soup.find_all('li', class_='b_algo')
        
        results = []
        for item in search_items:
            if len(results) >= max_results:
                break
            title_tag = item.find('h2')
            link_tag = title_tag.find('a') if title_tag else None
            if link_tag and link_tag.get('href'):
                title = link_tag.get_text(strip=True)
                href = link_tag.get('href')
                if href.startswith('http') and 'microsoft.com' not in href:
                    results.append({'title': title, 'href': href})

        logging.info(f"Bing 搜索完成，找到 {len(results)} 个有效结果。")
        return results

    except requests.RequestException as e:
        logging.error(f"请求 Bing 搜索时发生网络错误: {e}", exc_info=True)
        return []
    except Exception as e:
        logging.error(f"解析 Bing 搜索结果时发生未知错误: {e}", exc_info=True)
        return []

def scrape_website_content(url: str):
    """从给定 URL 抓取主要文本内容。"""
    logging.info(f"开始抓取 URL: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        logging.info(f"成功抓取 URL: {url}")
        return text
    except requests.RequestException as e:
        logging.error(f"抓取 {url} 时发生错误: {e}", exc_info=True)
        return None

def generate_search_query_with_llm(user_question: str, api_key: str, model_name: str, base_url: str = None):
    """使用 LLM 根据用户问题生成优化的搜索词句。"""
    logging.info(f"开始使用模型 '{model_name}' 生成搜索词句...")
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url if base_url else None)
        
        prompt = SEARCH_QUERY_PROMPT.format(user_question=user_question)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # 较低的温度以获得更确定的搜索词
        )
        search_query = response.choices[0].message.content.strip()
        logging.info(f"生成的搜索词句: {search_query}")
        return search_query

    except Exception as e:
        logging.error(f"生成搜索词句时发生错误: {e}", exc_info=True)
        st.error(f"生成搜索词句时发生错误: {e}")
        return None

def summarize_with_llm(user_question: str, scraped_data: list, api_key: str, model_name: str, base_url: str = None):
    """使用 LLM 生成带引用来源的摘要。"""
    logging.info(f"开始使用模型 '{model_name}' 生成摘要...")
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url if base_url else None)
        
        content_for_prompt = ""
        for item in scraped_data:
            content_for_prompt += f"来源 URL: {item['url']}\n内容:\n{item['content']}\n\n---\n\n"

        # 从配置文件加载并格式化 prompt
        prompt = SUMMARY_PROMPT.format(user_question=user_question, content_for_prompt=content_for_prompt)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        logging.info("摘要生成成功。")
        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"生成摘要时发生严重错误: {e}", exc_info=True)
        st.error(f"生成摘要时发生错误: {e}")
        return None

def evaluate_content_sufficiency_with_llm(user_question: str, scraped_data: list, api_key: str, model_name: str, base_url: str = None):
    """使用 LLM 评估抓取到的内容是否足以回答用户问题。"""
    logging.info(f"开始使用模型 '{model_name}' 评估内容充足性...")
    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url if base_url else None)
        
        content_summary = ""
        for item in scraped_data:
            content_summary += f"来源 URL: {item['url']}\n内容摘要: {item['content'][:500]}...\n\n" # 提取部分内容作为摘要

        prompt = EVALUATION_PROMPT.format(user_question=user_question, scraped_content_summary=content_summary)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # 较低的温度以获得更确定的判断
        )
        evaluation_result = response.choices[0].message.content.strip().upper()
        logging.info(f"AI 内容充足性评估结果: {evaluation_result}")
        return evaluation_result == "YES"

    except Exception as e:
        logging.error(f"评估内容充足性时发生错误: {e}", exc_info=True)
        st.error(f"评估内容充足性时发生错误: {e}")
        return False

# --- UI 配置 ---
st.set_page_config(
    page_title="Easy Search 易搜",
    page_icon="✨",
    layout="wide"
)

# --- 应用标题和描述 ---
st.title("🔍 Easy Search 易搜")
st.markdown("""
欢迎使用 Easy Search 易搜！请用自然语言输入您的问题，我将为您搜索网络、
总结信息，并提供清晰且有来源的答案。
""")

# --- 主要搜索区域 ---
st.header("提出您的问题")

def clear_question():
    st.session_state.user_question_input = ""

cols_question_input = st.columns([0.8, 0.2])
with cols_question_input[0]:
    user_question = st.text_input(
        label="在此输入您的问题:",
        placeholder="例如：为什么我的笔记本电脑风扇声音特别大？（Easy Search 易搜）",
        label_visibility="collapsed",
        key="user_question_input" # 为输入框添加一个key
    )
with cols_question_input[1]:
    st.button("清空问题", on_click=clear_question, help="清空问题输入框")

search_button = st.button("获取智能答案")

# --- 结果区域 ---
cols_answer_header = st.columns([0.9, 0.1]) # 调整比例，让按钮更紧凑
with cols_answer_header[0]:
    st.header("回答")
with cols_answer_header[1]:
    # 清空回答的按钮
    def clear_answer():
        st.session_state.answer_content = ""
        st.session_state.show_placeholder = True
        st.session_state.user_question_input = "" # 清空问题输入框
        # 重新渲染占位符
        answer_display_area.markdown("<p style='color:grey;'>您的问题回答将显示在此...</p>", unsafe_allow_html=True)

    st.button("清空回答", on_click=clear_answer, help="清空回答区域")

results_area = st.container(border=True)

# 初始化 session_state 中的回答内容和占位符显示状态
if 'answer_content' not in st.session_state:
    st.session_state.answer_content = ""
if 'show_placeholder' not in st.session_state:
    st.session_state.show_placeholder = True

# 用于显示回答的动态区域
answer_display_area = results_area.empty()

# 根据状态显示占位符或实际内容
if st.session_state.show_placeholder:
    answer_display_area.markdown("<p style='color:grey;'>您的问题回答将显示在此...</p>", unsafe_allow_html=True)
else:
    answer_display_area.markdown(st.session_state.answer_content)
# results_area.write("您的摘要答案将显示在这里...")

# --- 按钮点击逻辑 ---
if search_button:
    if not st.session_state.api_key:
        results_area.error("请输入您的 API Key。")
    elif not user_question:
        results_area.warning("进行搜索前，请输入一个问题。")
    else:
        try:
            # 清空结果区域，准备显示进度条和最终结果
            # results_area.empty() # 这一行不再需要，因为 answer_display_area 已经控制了显示
            # 每次点击搜索按钮时，重置回答区域为初始状态
            st.session_state.answer_content = ""
            st.session_state.show_placeholder = True
            answer_display_area.markdown("<p style='color:grey;'>您的问题回答将显示在此...</p>", unsafe_allow_html=True)
            
            # 隐藏占位符，准备显示进度条
            st.session_state.show_placeholder = False
            progress_text = "正在初始化..."
            my_bar = results_area.progress(0, text=progress_text)

            try:
                # 步骤 1: 生成搜索词句
                my_bar.progress(20, text="正在分析用户意图并生成搜索词句...")
                optimized_query = generate_search_query_with_llm(
                    user_question=user_question,
                    api_key=st.session_state.api_key,
                    model_name=st.session_state.model_name,
                    base_url=st.session_state.base_url
                )
                if not optimized_query:
                    results_area.error("未能生成有效的搜索词句。请检查 AI 配置或尝试其他问题。")
                    my_bar.empty()
                    st.stop()
                logging.info(f"AI 生成的搜索词句: {optimized_query}")

                # 步骤 2: 迭代搜索和内容抓取，直到内容充足或达到最大迭代次数
                MAX_ITERATIONS = 3 # 设置最大迭代次数
                current_iteration = 0
                all_scraped_content = []
                content_sufficient = False
                
                while not content_sufficient and current_iteration < MAX_ITERATIONS:
                    current_iteration += 1
                    my_bar.progress(30 + (current_iteration * 15), text=f"正在进行第 {current_iteration} 轮搜索和内容抓取...")
                    
                    # 执行网络搜索
                    search_results = search_the_web(optimized_query, max_results=st.session_state.max_search_results_per_iteration * current_iteration) # 每次迭代获取更多结果
                    if not search_results:
                        if current_iteration == 1: # 第一次搜索就没结果，直接报错
                            results_area.error("未能找到任何网络结果。请尝试其他问题。")
                            my_bar.empty()
                            st.stop()
                        else: # 后续迭代没结果，跳出循环
                            logging.warning(f"第 {current_iteration} 轮搜索未能找到更多网络结果。")
                            break
                    logging.info(f"第 {current_iteration} 轮找到了 {len(search_results)} 个网络结果。")

                    # 抓取网页内容
                    new_scraped_content = []
                    for i, result in enumerate(search_results):
                        url = result['href']
                        # 避免重复抓取已有的 URL
                        if url not in [item['url'] for item in all_scraped_content]:
                            my_bar.progress(30 + (current_iteration * 15) + int(5 * (i + 1) / len(search_results)), text=f"第 {current_iteration} 轮抓取: {url}")
                            content = scrape_website_content(url)
                            if content:
                                new_scraped_content.append({"url": url, "content": content})
                            else:
                                logging.warning(f"无法从 {url} 抓取内容")
                    
                    if new_scraped_content:
                        all_scraped_content.extend(new_scraped_content)
                        logging.info(f"第 {current_iteration} 轮成功抓取 {len(new_scraped_content)} 个页面。总计抓取 {len(all_scraped_content)} 个页面。")
                        
                        # 评估内容充足性
                        my_bar.progress(80, text="正在评估内容充足性...")
                        content_sufficient = evaluate_content_sufficiency_with_llm(
                            user_question=user_question,
                            scraped_data=all_scraped_content,
                            api_key=st.session_state.api_key,
                            model_name=st.session_state.model_name,
                            base_url=st.session_state.base_url
                        )
                        if content_sufficient:
                            logging.info("AI 评估内容充足，准备生成摘要。")
                        else:
                            logging.info("AI 评估内容不足，将进行下一轮搜索。")
                    else:
                        logging.warning(f"第 {current_iteration} 轮未能抓取到任何新内容。")
                        break # 如果没有新内容被抓取，则停止迭代

                if not all_scraped_content:
                    results_area.error("未能从任何搜索结果中抓取到有效内容。请尝试其他问题。")
                    my_bar.empty()
                    st.stop()

                if not content_sufficient:
                    results_area.warning(f"在 {MAX_ITERATIONS} 轮搜索后，AI 仍认为内容不足以完全回答您的问题。将基于现有内容生成摘要。")
                
                # 步骤 3: 生成摘要
                my_bar.progress(90, text="正在生成摘要...")
                summary = summarize_with_llm(
                    user_question=user_question, 
                    scraped_data=all_scraped_content,
                    api_key=st.session_state.api_key,
                    model_name=st.session_state.model_name,
                    base_url=st.session_state.base_url
                )
                my_bar.progress(100, text="完成！")
                my_bar.empty() # 清除进度条

                if summary:
                    st.session_state.answer_content = summary
                    answer_display_area.markdown(summary)
                else:
                    results_area.error("生成摘要失败。")
            except Exception as e:
                logging.error("处理过程中发生未捕获的异常: %s", e, exc_info=True)
                results_area.error("处理过程中发生意外错误，请检查控制台日志。")
                my_bar.empty() # 确保错误时也清除进度条
        except Exception as e:
            logging.error("处理过程中发生未捕获的异常: %s", e, exc_info=True)
            st.error("处理过程中发生意外错误，请检查控制台日志。")

# --- 页脚和免责声明 ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: small;">
    <p><strong>免责声明:</strong> 本程序是一个由 AI 驱动的辅助工具。所有信息均从网络搜索结果中自动摘要，不保证其 100% 的准确性。请通过提供的来源链接核实关键信息。</p>
    <p><strong>隐私保护:</strong> 您的查询将被匿名处理，我们不会存储任何查询记录。</p>
</div>
""", unsafe_allow_html=True)
