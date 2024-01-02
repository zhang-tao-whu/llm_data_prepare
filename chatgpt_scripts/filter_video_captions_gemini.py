my_keys = 'AIzaSyCQcruT1qLvh4udqZdKTbs' + 'B6036uNpZUdQ'

captions = ["An adult, possibly male, gorilla is prominently featured in the image. This gorilla is standing upright, with its full body visible, and appears to be holding a baby gorilla in its arms. The two gorillas are standing in a lush green field, surrounded by trees.",
            "An adult, possibly male, chimpanzee is seen in the image. This chimpanzee is standing tall, with its mouth open, possibly making a distinctive sound or displaying some form of communication. It appears to be the dominant figure in the scene.",
            "A large, mature gorilla is visible, notable for its size and the two smaller gorillas nestled within its protective embrace. This gorilla, possibly the mother, is standing tall in the grassy field, its towering stature emphasized by its surroundings.",
            "A large, hairy, and possibly dirty ape is sitting in the grass. The ape appears to be eating something, perhaps leaves or berries, and it's holding its hand up to its mouth as if it's laughing.",
            "An adult, possibly male, chimpanzee is prominently featured in the image. This chimp is standing in a grassy field and appears to be holding a small baby chimp in its arms. The two are standing close together, showcasing a strong bond between them.",
            "An adult, possibly male, chimpanzee is visible in the image. This chimp appears to be interacting with a baby chimp, and it's the larger of the two.",
            "An adult, possibly the mother, is visible in the image. She is standing and appears to be holding a baby chimp in her arms.",
            "An adult, possibly the mother, is standing protectively over a baby chimp. This larger chimp is facing the camera, with its head turned to the side. The mother chimp's posture suggests that she is alert and attentive, fully engaged with her surroundings.",
            "On the right side of the image, there is a larger, presumably older, monkey. This monkey is facing towards the left and appears to be holding the smaller monkey in its arms.",
            "The right side of the image features a large, tall gorilla, possibly the mother, who is holding a baby gorilla. She is standing in the grass, and her face is clearly visible.",
            "A young, wild, and untamed baby chimpanzee is sitting in the grass. The chimp is holding onto a branch with both hands, and it's staring directly at the camera. The chimp's fur is a mixture of black and light brown, and it's sitting on the ground, with a grassy field as its backdrop.",
            "An adult, possibly a mother, chimpanzee is visible in the image. She is standing tall, with her head held high, and appears to be protecting a younger chimp.",
            "An adult, possibly the mother, is visible in the image. She is the taller of the two monkeys and is located on the right side of the image.",
            "An adult, possibly the mother, is visible in the image. She is standing and appears to be holding a baby elephant.",
            "An adult, possibly a mother, elephant is prominently featured in the image. She is holding a baby elephant, providing a sense of protection and care. The two elephants are standing in a grassy field, with trees in the background, creating a natural and peaceful setting.",
            "A baby elephant is prominently featured in the image, with its trunk playfully wrapped around one of its legs. The baby elephant is standing in a grassy field, and it's the main focus of the image.",
            "A baby elephant is in the center of the image, its trunk playfully wrapped around an adult elephant's leg. The baby elephant's curiosity and playfulness are evident in this action of holding onto the adult.",
            "A large, mature, and possibly adult ape is seen in the image. This ape is standing in a grassy field and appears to be holding a branch in its mouth. It's a dominant presence in the scene, with its size and stature making it stand out against the backdrop of the grassy field.",
            "A young, dark-colored monkey is standing in the grassy field. It seems to be a baby monkey, as it is smaller in size compared to the other monkey. The monkey is facing towards the right, and it appears to be playfully biting the other monkey's leg.",
            "A large, adult, and possibly mother chimpanzee is sitting in the grass. She is facing towards the right, with her head turned away from the camera. She appears to be relaxed and is not the baby chimp.",
            "The right side of the image features a large, possibly adult, monkey. This monkey is facing towards the camera and appears to be looking directly at it. The monkey's face is clearly visible, and it seems to be the main focus of the image.",
            "A large, possibly adult, monkey is sitting in the grass. It appears to be resting or relaxing, as it's not in a position that would indicate it's actively moving.",
            "A large, mature elephant is seen in the middle of the image, its face fully visible. It appears to be the focal point of the image, with its trunk wrapped around another elephant, possibly a baby. The elephant's skin is wrinkled, suggesting it is an adult.",
            "A young, small, and dark-skinned monkey is sitting in the grass. It appears to be a baby monkey, as it is quite small and seems to be in close proximity to its mother. The monkey is sitting, possibly on the ground, and is looking upwards, seemingly at the camera.",
            "A large, black, and brown monkey is sitting in the grass. It appears to be holding another monkey in its arms, creating a poignant scene of affection.",
            "A young, small, and dark-skinned monkey is sitting in the grass. It appears to be a baby monkey, as it is quite small and seems to be in close proximity to its mother.",
            "A large, black, and possibly young elephant is sitting in the grass. It seems to be in a relaxed state, possibly waving its trunk around. The elephant is positioned in the middle of the field, and it appears to be looking directly at the camera.",
            "A large, possibly adult, elephant is prominently featured in the image. It is sitting in a grassy field, with its face completely covered by its hands. The elephant appears to be in a relaxed or contented state, possibly enjoying the serene environment.",
            "A large, black, and possibly dirty-looking monkey is sitting in the grass. The monkey seems to be holding its hands up in front of its face, possibly in a protective or playful manner."]

format_captions = 'Descriptions: ['
for caption in captions:
    format_captions = format_captions + "\"" + caption + "\", "
format_captions = format_captions + "]\n"
format_captions = format_captions + "Label category: chimp\n"

system_messages = "You are an AI visual assistant. There is a description of an object in each frame of a video, and these descriptions may contain errors. Given the label category of the object, please determine the reliability of each description, where the category in the description may be a synonym of the label category. The format of the description that needs to be evaluated is [caption 1, caption 2, ..., caption n]. Your answer should follow the following format [True, False, ..., True], where True represents that the corresponding caption is reliable, and False represents that the corresponding caption is incorrect or conflicts with the label category.\n \
There is a example:\n \
Descriptions: [\"A baby chimp, possibly a baby monkey, is located in the middle of the image. It is being held by another chimp, presumably its mother, and is in the process of feeding.\", \"A baby monkey is being held by its mother in the grass. The baby monkey is small and seems to be feeding, while the mother monkey watches over it carefully.\", \"A baby elephant is being held by its mother. The baby elephant is small and seems to be feeding. The scene is heartwarming, showcasing the bond between a mother and her young.\", \"A baby elephant is visible, it appears to be feeding from its mother. The baby elephant is small and seems to be nursing from its mother's leg.\"]\n \
Label category: chimp\n \
Your answer should be: [True, True, False, False]"

import google.generativeai as genai
genai.configure(api_key="xxxxx")
model = genai.GenerativeModel(model_name="gemini-pro")
prompt_parts = [
    "写一个 Python 函数并向我解释",
]
response = model.generate_content(prompt_parts)
print(response.text)
