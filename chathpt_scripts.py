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

captions = ["A baby chimp, possibly a baby monkey, is being held by a larger chimp. The baby chimp is in the center of the image, slightly below the larger chimp. The baby chimp appears to be in a playful or relaxed state, and it's being held by the larger chimp.",
            "A baby chimp, possibly a baby monkey, is being held by a larger chimp. The baby chimp is in the center of the image, slightly below the larger chimp. The baby chimp appears to be in a protective position, nestled against the larger chimp.",
            "A baby chimp, possibly a baby monkey, is being held by a larger chimp. The baby chimp is located towards the center of the image, slightly above the midline, and appears to be in the grip of the adult chimp.",
            "A baby chimpanzee is located in the center of the image, slightly towards the right. It is being held by a larger chimpanzee, presumably its mother, and appears to be drinking from her. The baby chimp is small and seems to be enjoying the meal.",
            "A baby chimp, possibly a baby monkey, is being held by a larger chimp. The baby chimp is located towards the center of the image, slightly above the midline, and appears to be in the grip of the adult chimp.",
            "A baby chimp, possibly the youngest in the group, is being held by an adult chimp. The baby chimp is in the middle of the scene, slightly towards the right, and appears to be being cradled by the adult chimp.",
            "A baby chimp, possibly a baby, is being held by an adult chimp. The baby chimp is in the center of the image, slightly towards the right. The baby chimp appears to be in a protective position, nestled against the chest of the adult chimp.",
            "A baby chimp, possibly a baby monkey, is being held by an adult chimp. The baby chimp is small and seems to be getting comforted by the adult.",
            "A baby chimp, possibly a girl, is being held by an adult chimp. The baby chimp is in the center of the image, slightly towards the left. The baby chimp appears to be in a protective position, nestled against the chest of the adult chimp.",
            "A baby chimp is nestled close to its mother, both of them are sitting on the grass. The baby chimp is facing towards the camera, giving a clear view of its face.",
            "A baby chimpanzee is located in the middle of the image, lying on its back and holding its legs up. It seems to be in a playful or relaxed position, surrounded by lush greenery.",
            "A baby chimp, possibly a baby monkey, is located in the middle of the image. It is being held by another chimp, presumably its mother, and is in the process of feeding.",
            "A baby monkey is being held by its mother in the grass. The baby monkey is small and seems to be feeding, while the mother monkey watches over it carefully.",
            "A baby elephant is being held by its mother. The baby elephant is small and seems to be feeding. The scene is heartwarming, showcasing the bond between a mother and her young.",
            "A baby elephant is visible, it appears to be feeding from its mother. The baby elephant is small and seems to be nursing from its mother's leg.",
            "A baby elephant is seen holding a branch in its trunk. The branch appears to be a toy that the baby elephant is playing with. The baby elephant seems to be enjoying its time in the grassy field.",
            "A baby elephant is being fed by its mother. The baby elephant is positioned slightly above the ground, and it appears to be reaching up to its mother for a meal.",
            "A baby chimp is being held by an adult chimp. The baby chimp is small and seems to be feeding from its mother. The scene is set in a grassy field, and the chimp is holding a branch in its mouth.",
            "A baby chimp is being held by an adult chimp. The baby chimp is in the middle of the scene, slightly towards the left. The baby chimp appears to be in a playful or affectionate position, possibly being held by the adult chimp.",
            "A baby chimp is nestled underneath the larger chimp, possibly seeking protection or shade. The baby chimp's small size and the positioning underneath the larger chimp suggests a close relationship, possibly a mother-child bond.",
            "A baby chimp is nestled underneath the larger chimp, possibly seeking protection or shade. The baby chimp's face is visible, showing its adorable features, and it appears to be nestled under the larger chimp's body.",
            "A baby elephant is seen in the middle of the image, its trunk is wrapped around the leg of an adult elephant, possibly its mother. The baby elephant seems to be holding onto the adult for support.",
            "A baby elephant is nestled under a larger elephant, seeking shade and protection. The baby elephant's position is somewhat towards the center of the image, slightly towards the left. The larger elephant's trunk is visible, gently draped over the baby, creating a sense of warmth and guardianship.",
            "A baby elephant is seen in the middle of the grassy field, its trunk is wrapped around the leg of an adult elephant, possibly its mother. The baby elephant seems to be trying to reach up to its mother.",
            "A baby elephant is nestled between two adult elephants, seeking shade and protection. The baby elephant's small size and youthful features are evident, as it stands in the middle of the larger elephants.",
            "A baby chimp's arm is visible, holding onto the back of its mother. The baby chimp is in the middle of the grassy field, slightly hidden by the tall grass.",
            "The right front leg of the elephant is visible, with the foot raised. This leg is positioned in front of the other elephant's face, and it's the one that's closest to the camera.",
            "The middle of the image features a monkey, possibly a baby, lying on the ground. It appears to be in a grassy area, and it's not facing the camera.",
            "The left front leg of the monkey is visible, with the knee bent and the toes pointing downwards."]

format_captions = '['
select_idxs = [1, 2]
format_captions = format_captions + '\{caption1: {}, caption2: {}}, '.format(captions[select_idxs[0]], captions[select_idxs[1]])
format_captions = format_captions + '\{location1: [0.56, 0.44, 0.71, 0.66], location2: [0.51, 0.22, 0.69, 0.41]}'
format_captions = format_captions + ']'

#system_messages = "You are an AI visual assistant that can analyze a video. There is a caption of an object in consecutive video frames, but the caption in a small number of frames may contain errors. The caption includes the object category, color, and spatial position. In a video, the object should have a consistent category and color, but the spatial position may change. You need to correct any errors that may exist based on the caption of the object in all frames and reorganize the description of each frame in a unified format. The descriptions to be processed are given in the following format: {frame1: caption1, frame2: caption2, ..., frameT: captionT}."
#system_messages = "You are an AI visual assistant that can analyze images. There are two images, each with a corresponding caption describing an object in the image. The captions may include the object's category, color, appearance, and spatial position (ignoring spatial position for this task). The two objects are the same object. Please summarize the common features found in the two captions in one sentence. The captions to be processed are given in the following format: {caption1: xx, caption2: xx}. If there are no common features between the two captions that need to be processed, please let me know “No common features found”."
system_messages = "You are an AI visual assistant that can analyze video. There are two sequential video frames, each with a corresponding caption describing an object in the frame. The captions may include the object's category, color, appearance, and spatial position. In addition to the caption, the position of each object is also provided in the form of an outer bounding box, {x1, y1, x2, y2}, where x1, y1 represent the position of the top-left corner of the box, and x2, y2 represent the position of the bottom-right corner of the box. The two objects are the same one. Please determine the reasons why these two objects are considered the same object based on the given captions and positions. The captions and locations to be processed are given in the following format: [{caption1: xx, caption2: xx}, {location1: [x1, y1, x2, y2], location2: [x1, y1, x2, y2]}].\n There are a example: The captions and locations: [{caption1: “A car is visible in the background, positioned behind the bus. It appears to be a dark-colored SUV, possibly black, and is located on the right side of the bus.”, caption2: “A black car is parked behind a white van. It's positioned towards the right side of the image, and it's the second car from the right.”}, {location1: [0.12, 0.23, 0.42, 0.55], location2: [0.16, 0.33, 0.44, 0.64]}]\n Your answer should be: The objects in these two frames are the same. The object is a black SUV car, moving towards the right. The position of the object in the second frame aligns with this motion trajectory."
from openai import OpenAI

api_base = "https://cd.aiskt.com/v1"
client = OpenAI(api_key=my_keys, base_url=api_base)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[{"role": 'system', "content": system_messages}, {"role": "user", "content": "{}".format(format_captions)}]
)

print(completion.choices[0].message.content)