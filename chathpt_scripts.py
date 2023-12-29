import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
my_keys = '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'
openai.api_key = my_keys

# system prompt，用于告诉GPT当前的情景，不了解可以放空，没有影响。
# system prompt例如：'You are a marketing consultant, please answer the client's questions in profession style.'
system_content = ''

# 这里使用了langchain包简化与GPT的对话过程，基于的是GPT-3.5，能力与免费版的chatGPT相同。GPT-4需要自行申请加入waitlist
#messages = [SystemMessage(content=system_content)]

def chat(questions):
    chat = ChatOpenAI(temperature=0.2, openai_api_key=openai.api_key)
    messages = [SystemMessage(content=system_content)]
    messages.append(HumanMessage(content=questions))
    response = chat(messages)
    return response.content

ret = chat("please transfer this text into chinese: Can use the slides you prepared before ha.")