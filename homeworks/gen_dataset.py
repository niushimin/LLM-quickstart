import csv
import datetime
import os
import re
import time

from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI


os.environ['OPENAI_API_KEY'] = 'replace your openai api key'


# 定义返回对象
class ContentSummaryModel(BaseModel):
    """ContentModel two str together."""
    content: str = Field(..., description="Philosophy of the Book of Changes")
    summary: str = Field(..., description="Summary of philosophy of the Book of Changes")


# 初始化LangChain的GPT-3.5调用
openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125",
                               temperature=1,
                               max_tokens=4095)


def gen_content_summary(raw_content):
    """
    使用LangChain GPT-3.5调用处理单个数据样例。
    :param raw_content: 原始数据样例。
    :return: GPT-3.5模型生成的内容。
    """

    # 构造提示模板
    prompt_template = PromptTemplate.from_template(
        template="""你是中国古典哲学大师，尤其擅长周易的哲学解读，对周易卦象有深刻的理解，乐于分享自己的见解。
            你需要将下面段卦象使用中文进行整理润色，并生成用于大模型训练的内容和格式。

            要求返回结果字数应该不少于200个汉字。
            返回格式要求：
            content:"卦名"
            summary:"卦象内容，包含卦象分布，卦象解读等"
            
            卦象原文：{raw_content}
            """
    )

    openai_chat_model_chain = openai_chat_model.bind_tools([ContentSummaryModel]) | PydanticToolsParser(
        tools=[ContentSummaryModel])

    # 调用模型生成结果
    assistant_message = openai_chat_model_chain.invoke(prompt_template.format_prompt(raw_content=raw_content))

    return assistant_message[0]


def gen_content_summary_pairs(summary):
    """
    生成20对提问和总结的配对。
    :param summary: 内容的总结。
    :return: 包含20对提问和总结的列表。
    """

    # 构造提示模板
    prompt_template = PromptTemplate.from_template(
        template="""你是中国古典哲学爱好者，尤其对周易卦象非常感兴趣，请你根据给定卦象的内容：{summary}，
            使用中文提出20个你最感兴趣的问题，问题需要以问号结尾。
            """
    )

    # 调用模型生成结果
    assistant_message = openai_chat_model.invoke(prompt_template.format_prompt(summary=summary))

    # 提取结果中问题列表
    questions = extract_questions_by_regex(assistant_message.content)

    # 创建提问和总结的配对
    question_summary_pairs = [(question, summary) for question in questions]

    return question_summary_pairs


def extract_questions_by_regex(input_string):
    pattern = r'\d+\.?\s*(.*[？?])'  # 匹配序号和内容的正则表达式模式
    questions = re.findall(pattern, input_string)
    print(questions)
    return questions


def get_raw_data():
    # 初始化一个空列表用于存储原始内容数据
    raw_content_data = []

    # 读取文件并分割数据样例
    with open('data/full_raw_data.txt', 'r', encoding='utf-8') as file:
        content = file.read()
        # 使用连续的换行符('\n\n')作为分隔符来分割文本
        data_samples = content.split('\n\n')

        # 遍历分割后的数据样例并添加到列表中
        for sample in data_samples:
            # 移除每个样例中的额外空白字符（如果有的话）
            cleaned_sample = sample.strip()
            # 仅添加非空样例
            if cleaned_sample:
                raw_content_data.append(cleaned_sample)

    return raw_content_data


def gen_and_save_content_summary():
    # 解析 raw_data.txt 得到 raw_content_data 列表
    raw_content_data = get_raw_data()

    # 创建带有时间戳的CSV文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/zhouyi_dataset_{timestamp}.csv"

    # 创建CSV文件并写入标题行
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['content', 'summary'])

        # 循环遍历 raw_content_data 数据样例
        for raw_content in raw_content_data:
            # 调用 gen_content_summary 方法得到 ai_message_content
            genreated_content = gen_content_summary(raw_content)

            # 解析 ai_message_content 得到 content 和 summary
            print("Content:", genreated_content.content)
            print("Summary:", genreated_content.summary)

            # 调用 gen_content_summary_pairs 得到20组 pairs
            question_summary_pairs = gen_content_summary_pairs(genreated_content.summary)

            # 将 pairs 写入 csv 文件
            for pair in question_summary_pairs:
                print(f"---Question:{pair[0]} Answer:{pair[1]}")
                writer.writerow(pair)

            # 避免被openai限速
            time.sleep(30)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    gen_and_save_content_summary()
