my_keys = 'sk-hlV40Lxa6Ce1OwJg89083e' + '8b005040D2A10eFa2c096631Bd'

format_captions = 'Locations: {frame1: [0.786, 0.25, 0.848, 0.322], frame2: [0.796, 0.252, 0.860, 0.323]}'
system_messages = "You are an AI visual assistant. Given the position of an object in consecutive video frames, please summarize its direction of motion. The position of each object is provided in the form of an outer bounding box, {x1, y1, x2, y2}, where x1, y1 represent the position of the top-left corner of the box, and x2, y2 represent the position of the bottom-right corner of the box. The positions of an object between two frames follows the following format: {frame1: [x1, y1, x2, y2], frame2: [x1, y1, x2, y2]}. x refers to the distance from the left side of the image, and y refers to the distance from the top of the image.\n \
There are some examples:\n \
Locations: {frame1: [0.261, 0.101, 0.787, 0.626], frame2: [0.152, 0.099, 0.676, 0.624]}\n \
Your answer should be: The object is moving to the left.\n \
Locations: {frame1: [0.261, 0.101, 0.787, 0.626], frame2: [0.310, 0.100, 0.832, 0.627]}\n \
Your answer should be: The object is moving to the right.\n \
Locations: {frame1: [0.261, 0.101, 0.787, 0.626], frame2: [0.263, 0.102, 0.785, 0.621]}\n \
Your answer should be: Object shows no apparent movement."
from openai import OpenAI

api_base = "https://cd.aiskt.com/v1"
client = OpenAI(api_key=my_keys, base_url=api_base)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[{"role": 'system', "content": system_messages}, {"role": "user", "content": "{}".format(format_captions)}]
)

print(completion.choices[0].message.content)