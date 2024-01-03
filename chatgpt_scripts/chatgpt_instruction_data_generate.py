import json
import os
from openai import OpenAI
from chatgpt_scripts.system_messages import system_messages_caption_check, system_messages_motion,\
    system_messages_positive_instruction_generate, system_messages_style_transfer

my_keys = 'sk-hlV40Lxa6Ce1OwJg89083e' + '8b005040D2A10eFa2c096631Bd'

class InstructionGenerater(object):
    def __init__(self):
        self.api_key = my_keys
        self.api_base = "https://cd.aiskt.com/v1"
        self.gen_data_nums = 0

    def _check_obj(self, obj):
        for caption in obj["captions"]:
            if caption is None:
                return False
        return True

    def _gpt_check_category(self, caption, category):
        format_captions = 'Descriptions: ' + caption + "\n " + "Label category: {}.".format(category)

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_caption_check},
                      {"role": "user", "content": "{}".format(format_captions)}]
        )
        return completion.choices[0].message.content

    def _check_captions(self, obj1):
        for caption in obj1["captions"]:
            check_ret = self._gpt_check_category(caption, obj1['categories'])
            if 'False' in check_ret or 'flase' in check_ret:
                return False
        return True

    def _add_motion(self, obj):
        w, h = obj['image_size']
        box_strings = []
        for box in obj["bboxes"]:
            normalized_box = [box[0] * 1.0 / w, box[1] * 1.0 / h, box[2] * 1.0 / w, box[3] * 1.0 / h]
            normalized_box = ["{:.3f}".format(num) for num in normalized_box]
            normalized_box_string = "[" + normalized_box[0] + ', ' + normalized_box[1] + ', '\
                                    + normalized_box[2] + ', ' + normalized_box[3] + ']'
            box_strings.append(normalized_box_string)

        format_captions = 'Locations: {frame1: ' + box_strings[0] + ', frame2: ' + box_strings[1] + '}'
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_motion},
                      {"role": "user", "content": "{}".format(format_captions)}]
        )

        motion_cap = completion.choices[0].message.content
        obj['motion'] = motion_cap
        return

    def _generate_pos_data(self, caption1, caption2, motion_caption):
        format_captions = "The captions and motion: {caption1: \"" + caption1 + "\", " + "caption2: \"" + caption2 + \
                          "\"}\n"
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_positive_instruction_generate},
                      {"role": "user", "content": "{}".format(format_captions)}],
            temperature=0.01
        )
        motion_reason = "In addition, in the first frame, " + motion_caption + " In the second frame, the object\'s position aligns with this motion tendency."

        return completion.choices[0].message.content + motion_reason

    def _change_style(self, words):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_style_transfer},
                      {"role": "user", "content": "{}".format(words)}],
            temperature=0.01
        )
        return completion.choices[0].message.content

    def get_instruction_datas(self, obj1, obj2):
        # obj {'image_files': [img1_path, img2_path], 'segmentations': [{rle1}, {rle2}],
        # "captions": [cap1, cap2], 'image_size': [w, h], 'bboxes': [box1, box2], "categories": class_name}
        if not self._check_obj(obj1) or not self._check_obj(obj2):
            return []
        if not self._check_captions(obj1) or not self._check_captions(obj2):
            return []

        # passed the check process

        self._add_motion(obj1)
        self._add_motion(obj2)

        positive_instruction_data = self._generate_pos_data(obj1['captions'][0], obj1['captions'][1], obj1['motion'])



        return


generater = InstructionGenerater()
ret = generater._generate_pos_data(caption1="A car is visible in the background, positioned behind the bus. It appears to be a dark-colored SUV, possibly black, and is located on the right side of the bus.",
                                   caption2="A black car is parked behind a white van. It's positioned towards the right side of the image, and it's the second car from the right.",
                                   motion_caption="The object is moving to the right.")
print(ret)

print('transfered ------------')

print(generater._change_style(ret))