my_keys = 'sk-hlV40Lxa6Ce1OwJg89083e' + '8b005040D2A10eFa2c096631Bd'

captions = ["A car is visible in the background, positioned behind the bus. It appears to be a dark-colored SUV, possibly black, and is located on the right side of the bus.",
            "A car is parked behind the bus, visible through the window. It's located on the right side of the bus, and appears to be a darker vehicle, possibly black.",
            "A black car is visible in the background, positioned behind the bus. It appears to be moving in the same direction as the bus.",
            "A black car is seen behind the bus, positioned on the street. It appears to be moving in the opposite direction of the bus, adding to the bustling scene of the street.",
            "A black car is visible in the background, positioned behind the bus. It appears to be moving in the same direction as the bus, adding to the bustling scene of the street.",
            "A black car is visible in the background, positioned behind the bus. It appears to be moving in the same direction as the bus, adding to the bustling city scene.",
            "A black car is visible in the background, positioned behind the bus. It appears to be parked on the street, likely behind a parking meter.",
            "A black car is visible on the right side of the image, positioned behind a bus. It appears to be in motion, driving away from the viewer.",
            "A black car is parked behind the bus, positioned towards the right side of the image. It appears to be a sizable vehicle, possibly a van or a truck, and is situated in front of a green sign.",
            "A black car is parked on the street, situated behind a white van. It appears to be a sizable vehicle, possibly a truck, and is positioned directly behind the van.",
            "A black car is parked on the street, positioned behind a white van. It appears to be a sizable vehicle, possibly a truck, and is located near the center of the image.",
            "A black car is parked on the street, located behind a white van. It's positioned towards the right side of the image, and it's the second car from the right.",
            "A car is parked behind a bus, visible towards the right side of the image. The car appears to be a darker color, possibly black, and is situated on the street.",
            "A red fire hydrant is visible in the background, situated behind a bus. The hydrant is situated on the sidewalk, and it's visible between the bus and a purple sign.",
            "A black car is visible in the background, positioned behind a white truck. It appears to be a smaller vehicle, possibly a sedan, and is located on the right side of the image.",
            "A car is visible in the background, positioned behind a pole. It appears to be a dark-colored SUV, possibly black, and is located towards the right side of the image.",
            "A car is visible in the background, positioned behind the bus. It appears to be a black car, although the details are not clear.",
            "A bus is parked behind the car, visible only partially due to the angle of the car.",
            "A panda bear is sitting on a log, its front paws gripping the bark as it holds onto a branch. The bear appears to be eating bamboo, its mouth wide open as it reaches for the plant."]
idx = -6
format_captions = 'Descriptions: ' + captions[idx] + "\n " + "Label category: car."
# for caption in captions[:5]:
#     format_captions = format_captions + "\"" + caption + "\", "
# format_captions = format_captions + "]\n"
# format_captions = format_captions + "Label category: chimp\n And please give the reason.\n"

# system_messages = "You are an AI visual assistant. There is a description of an object in each frame of a video, and these descriptions may contain errors. Given the label category of the object, please determine if the object category described is consistent with the label category. Synonyms are considered consistent, such as chimp and monkey. The format of the description that needs to be evaluated is [caption 1, caption 2, ..., caption n]. Your answer should follow the following format [True, False, ..., True], where True represents that the corresponding caption is reliable, and False represents that the corresponding caption is incorrect or conflicts with the label category.\n \
# There is a example:\n \
# Descriptions: [\"A baby chimp, possibly a baby monkey, is located in the middle of the image. It is being held by another chimp, presumably its mother, and is in the process of feeding.\", \"A baby monkey is being held by its mother in the grass. The baby monkey is small and seems to be feeding, while the mother monkey watches over it carefully.\", \"A baby elephant is being held by its mother. The baby elephant is small and seems to be feeding. The scene is heartwarming, showcasing the bond between a mother and her young.\", \"A baby elephant is visible, it appears to be feeding from its mother. The baby elephant is small and seems to be nursing from its mother's leg.\"]\n \
# Label category: chimp\n \
# Your answer should be: [True, True, False, False]\n"
system_messages = "You are an AI visual assistant. There is a description of an object, and the description may contain errors. Given the label category of the object, please determine if the object category in the description is consistent with the label category. Synonyms are considered consistent, such as chimp and gorilla. Your answer should True or False, where True represents that the object category in the description is similar or consistent with the label category, while False indicates that there is a significant conflict between them.\n"
# There is some examples:\n \
# Descriptions: A baby chimp, possibly a baby monkey, is located in the middle of the image. It is being held by another chimp, presumably its mother, and is in the process of feeding.\n \
# Label category: chimp\n \
# Your answer should be: True\n \
# Descriptions: A baby chimp, possibly a baby monkey, is located in the middle of the image. It is being held by another chimp, presumably its mother, and is in the process of feeding.\n \
# Label category: ape\n \
# Your answer should be: True\n \
# Descriptions: A baby elephant is being held by its mother. The baby elephant is small and seems to be feeding. The scene is heartwarming, showcasing the bond between a mother and her young.\n \
# Label category: chimp\n \
# Your answer should be: False\n \
# Descriptions: An adult, possibly male, gorilla is prominently featured in the image. This gorilla is standing upright, with its full body visible, and appears to be holding a baby gorilla in its arms. The two gorillas are standing in a lush green field, surrounded by trees.\n \
# Label category: chimp\n \
# Your answer should be: True\n \
# "
from openai import OpenAI

api_base = "https://cd.aiskt.com/v1"
client = OpenAI(api_key=my_keys, base_url=api_base)

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[{"role": 'system', "content": system_messages}, {"role": "user", "content": "{}".format(format_captions)}]
)

print(completion.choices[0].message.content)