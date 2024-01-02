import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
my_keys = 'sk-hlV40Lxa6Ce1OwJg89083e' + '8b005040D2A10eFa2c096631Bd'
# openai.api_key = my_keys

# # system prompt，用于告诉GPT当前的情景，不了解可以放空，没有影响。
# # system prompt例如：'You are a marketing consultant, please answer the client's questions in profession style.'
# system_content = ''
#
# # 这里使用了langchain包简化与GPT的对话过程，基于的是GPT-3.5，能力与免费版的chatGPT相同。GPT-4需要自行申请加入waitlist
# #messages = [SystemMessage(content=system_content)]
#
# def chat(questions):
#     chat = ChatOpenAI(temperature=0.2, openai_api_key=openai.api_key)
#     messages = [SystemMessage(content=system_content)]
#     messages.append(HumanMessage(content=questions))
#     response = chat(messages)
#     return response.content
#
# ret = chat("please transfer this text into chinese: Can use the slides you prepared before ha.")
# print(ret)

captions = ["There is a black tire visible on the right side of the bus. It's located behind the bus's window and is the second tire from the right.",
            "A car is visible in the background, positioned behind the bus. It appears to be a dark-colored SUV, possibly black, and is located on the right side of the bus.",
            "A black car is parked behind a white van. It's positioned towards the right side of the image, and it's the second car from the right.",
            "A black car is parked behind a white van. It's positioned on the right side of the image, and it's the second car from the right.",
            "A black car is parked behind a white van. It's positioned towards the right side of the image, and it's the second car from the right.",
            "A black car is parked behind the bus, positioned towards the right side of the image. It appears to be a sizable vehicle, possibly a van or a truck, and is located in the background of the scene.",
            "A black car is parked on the street, located behind a white van. It's positioned towards the right side of the image, and it's the second car from the right.",
            "A car is parked on the street, situated behind a bus. The car appears to be a darker color, possibly black, and is positioned in front of a sign.",
            "A black car is positioned behind the bus, seemingly driving through the intersection. It's located on the right side of the bus, and appears to be in motion.",
            "A black car is parked on the street, situated behind a bus. It appears to be a sizable vehicle, possibly a van or a truck.",
            "A black car is parked on the street, positioned behind a white van. It appears to be a sedan, and it's located near the middle of the image.",
            "A black car is parked on the street, positioned behind a bus. It appears to be a sizable vehicle, possibly a van or a truck."]

format_captions = '{'
for i, caption in enumerate(captions):
    format_captions = format_captions + 'frame{}: {},'.format(i, caption)
format_captions = format_captions + '}'

system_messages = [SystemMessage(content="You are an AI visual assistant that can analyze a video. There is a caption of an object in consecutive video frames, but the caption in a small number of frames may contain errors. The caption includes the object category, color, and spatial position. In a video, the object should have a consistent category and color, but the spatial position may change. You need to correct any errors that may exist based on the caption of the object in all frames and reorganize the description of each frame in a unified format. The descriptions to be processed are given in the following format: {frame1: caption1, frame2: caption2, ..., frameT: captionT}.")]

from openai import OpenAI

api_base = "https://cd.aiskt.com/v1"
client = OpenAI(api_key=my_keys, base_url=api_base)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=system_messages + [{"role": "user", "content": "{}".format(format_captions)}]
)

print(completion.choices[0].message.content)