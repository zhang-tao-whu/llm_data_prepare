import json
import os
from openai import OpenAI
from chatgpt_scripts.system_messages import system_messages_caption_check, system_messages_motion,\
    system_messages_positive_instruction_generate, system_messages_style_transfer, system_messages_rewrite,\
    system_messages_negative_instruction_generate, system_messages_motion_compare, system_messages_style_transfer_neg,\
    system_messages_rewrite_neg
from chatgpt_scripts.question_answers import justify_positive_answers, justify_questions, reason_questions,\
    justify_negative_answers, justify_negative_answers_briefly
import random
import copy

def random_select(data_list):
    length = len(data_list)
    idx = random.randint(0, length - 1)
    return copy.deepcopy(data_list[idx])

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

    def _rewrite(self, words):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_rewrite},
                      {"role": "user", "content": "{}".format(words)}],
            temperature=0.01
        )
        return completion.choices[0].message.content

    def _generate_pos_data(self, caption1, caption2, motion_caption):
        format_captions = "The captions: {caption1: \"" + caption1 + "\", " + "caption2: \"" + caption2 + \
                          "\"}\n"
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_positive_instruction_generate},
                      {"role": "user", "content": "{}".format(format_captions)}],
            temperature=0.01
        )
        motion_reason = "The positions of the objects in the two frames are close. In addition, in the first frame, we observed that " + motion_caption + " And, in the second frame, the object\'s position aligns with this motion tendency ({}).".format(motion_caption)

        reasons = completion.choices[0].message.content + motion_reason
        reasons = self._change_style(reasons)
        reasons = self._rewrite(reasons)

        conversations = []
        conversations.append({'from': 'human', "value": random_select(justify_questions) + ' ' +\
                                                        random_select(reason_questions)})
        conversations.append({'from': 'gpt', "value": random_select(justify_positive_answers) + ' ' +\
                                                      reasons})
        return conversations

    def _compare_motion(self, motion1, motion2):
        format_captions = "Input: {object1: \"" + motion1 + "\", " + "object2: \"" + motion2 + \
                          "\"}\n"
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_motion_compare},
                      {"role": "user", "content": "{}".format(format_captions)}],
            temperature=0.01
        )
        return completion.choices[0].message.content

    def _generate_neg_data(self, caption1, caption2, motion1, motion2):
        format_captions = "The captions: {caption1: \"" + caption1 + "\", " + "caption2: \"" + caption2 + \
                          "\"}\n"
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_negative_instruction_generate},
                      {"role": "user", "content": "{}".format(format_captions)}],
            temperature=0.01
        )

        motion_compare = self._compare_motion(motion1, motion2)
        if "True" in motion_compare:
            motion_compare = " The above differences are sufficient to determine that these two objects are not the same, despite their similar movement directions."
            motion = motion1.split('object')[-1]
            motion = " These two objects" + motion
            motion_compare = motion_compare[:-1] + "({}).".format(motion)
        reasons = completion.choices[0].message.content + motion_compare
        reasons = self._change_style_neg(reasons)
        reasons = self._rewrite_neg(reasons)

        conversations = []
        conversations.append({'from': 'human', "value": random_select(justify_questions) + ' ' + \
                                                        random_select(reason_questions)})
        conversations.append({'from': 'gpt', "value": random_select(justify_negative_answers) + ' ' + \
                                                      reasons})
        return conversations

    def _change_style(self, words):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_style_transfer},
                      {"role": "user", "content": "{}".format(words)}],
            temperature=0.01
        )
        return completion.choices[0].message.content

    def _change_style_neg(self, words):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_style_transfer_neg},
                      {"role": "user", "content": "{}".format(words)}],
            temperature=0.01
        )
        return completion.choices[0].message.content

    def _rewrite_neg(self, words):
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": 'system', "content": system_messages_rewrite_neg},
                      {"role": "user", "content": "{}".format(words)}],
            temperature=0.01
        )
        return completion.choices[0].message.content

    def _add_other_informations(self,
                                image_paths=[],
                                segmentations=[],
                                image_size=[],
                                bboxes=[],
                                conversations=[],
                                type=None):
        assert len(image_size) == len(segmentations) == len(bboxes) == len(image_paths) == 2
        ret = {'images_path': image_paths, 'images_size': image_size, 'segmentations': segmentations, 'bboxes': bboxes,
               'conversations': conversations, 'type': type}
        return ret

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

        ret = []

        # generate positive datas
        positive_instruction_data = self._generate_pos_data(obj1['captions'][0], obj1['captions'][1], obj1['motion'])
        ret.append(self._add_other_informations(image_paths=obj1['image_files'], segmentations=obj1['segmentations'],
                                                image_size=[obj1['image_size'], obj1['image_size']],
                                                bboxes=obj1['bboxes'], conversations=positive_instruction_data,
                                                type='positive'))

        # positive_instruction_data = self._generate_pos_data(obj2['captions'][0], obj2['captions'][1], obj2['motion'])
        # ret.append(self._add_other_informations(image_paths=obj2['image_files'], segmentations=obj2['segmentations'],
        #                                         image_size=[obj2['image_size'], obj2['image_size']],
        #                                         bboxes=obj2['bboxes'], conversations=positive_instruction_data,
        #                                         type='positive'))
        #
        # # generate negative datas
        # negative_instruction_data = self._generate_neg_data(obj1['captions'][0], obj2['captions'][1],
        #                                                     obj1['motion'], obj2['motion'])
        # ret.append(self._add_other_informations(image_paths=[obj1['image_files'][0], obj2['image_files'][1]],
        #                                         segmentations=[obj1['segmentations'][0], obj2['segmentations'][1]],
        #                                         image_size=[obj1['image_size'], obj2['image_size']],
        #                                         bboxes=[obj1['bboxes'][0], obj2['bboxes'][1]],
        #                                         conversations=negative_instruction_data, type='negative'))
        #
        # negative_instruction_data = self._generate_neg_data(obj2['captions'][0], obj1['captions'][1],
        #                                                     obj2['motion'], obj1['motion'])
        # ret.append(self._add_other_informations(image_paths=[obj2['image_files'][0], obj1['image_files'][1]],
        #                                         segmentations=[obj2['segmentations'][0], obj1['segmentations'][1]],
        #                                         image_size=[obj2['image_size'], obj1['image_size']],
        #                                         bboxes=[obj2['bboxes'][0], obj1['bboxes'][1]],
        #                                         conversations=negative_instruction_data, type='negative'))
        return ret


class YouTubeVIS_Annotations(object):

    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.json_file = json.load(f)
        print("loaded ...")
        self.videos = self._get_video_id()
        self.video_id2annotations = self._get_video_annotations()
        self.video_ids = list(self.videos.keys())
        self.class_id2class_name = self._get_class_name()
        self.video_ids.sort()
        self.exhibit_lists = []

    def _get_video_id(self):
        ret = {}
        for video in self.json_file['videos']:
            ret[video['id']] = video
        return ret

    def _get_class_name(self):
        ret = {}
        categories = self.json_file['categories']
        for cat_info in categories:
            ret[cat_info["id"]] = cat_info["name"]
        return ret

    def _get_video_annotations(self):
        ret = {}
        for annotation in self.json_file['annotations']:
            if annotation['video_id'] in ret.keys():
                ret[annotation['video_id']].append(annotation)
            else:
                ret[annotation['video_id']] = [annotation]
        return ret

    def _get_specific_video(self):
        def get_avaliable_frames(datas):
            ret = []
            for i, val in enumerate(datas):
                if val is not None:
                    ret.append(i)
            return ret

        def reorginize_data(obj_anno, video_info, select_idx):
            # obj {'image_files': [img1_path, img2_path], 'segmentations': [{rle1}, {rle2}],
            # "captions": [cap1, cap2], 'image_size': [w, h], 'bboxes': [box1, box2], "categories": class_name}
            ret = {}
            ret['image_files'] = video_info['file_names'][select_idx: select_idx + 2]
            ret['segmentations'] = obj_anno['segmentations'][select_idx: select_idx + 2]
            ret['bboxes'] = obj_anno['bboxes'][select_idx: select_idx + 2]
            ret['captions'] = obj_anno['captions'][select_idx: select_idx + 2]
            ret['image_size'] = [video_info['width'], video_info['height']]
            class_id = obj_anno["category_id"]
            class_name = self.class_id2class_name[class_id]
            ret["categories"] = class_name
            return ret

        select_idx = random.randint(0, len(self.video_ids))
        video_id = self.video_ids[select_idx]

        objects_annotations = self.video_id2annotations[video_id]
        if len(objects_annotations) < 2:
            return None
        n_objects = len(objects_annotations)

        object_idx1 = random.randint(0, n_objects - 1)
        object_idx2 = random.randint(0, n_objects - 1)
        while object_idx1 == object_idx2:
            object_idx2 = random.randint(0, n_objects - 1)
        min_idx, max_idx = min(object_idx1, object_idx2), max(object_idx1, object_idx2)
        object_anno_1 = copy.deepcopy(objects_annotations[min_idx])
        object_anno_2 = copy.deepcopy(objects_annotations[max_idx])
        avaliable_list_1 = get_avaliable_frames(object_anno_1["segmentations"])
        if len(avaliable_list_1) < 2:
            return None
        avaliable_list_2 = get_avaliable_frames(object_anno_2["segmentations"])
        if len(avaliable_list_2) < 2:
            return None

        start = max(avaliable_list_1[0], avaliable_list_2[0])
        end = min(avaliable_list_1[-1], avaliable_list_2[-1])
        if end - start < 1:
            return None

        select_frame = random.randint(start, end - 1)

        _unique_id = select_idx * 1000000 + (min_idx * 100 + max_idx) * 1000 + select_frame
        if _unique_id in self.exhibit_lists:
            return None
        else:
            self.exhibit_lists.append(_unique_id)

        video_info = copy.deepcopy(self.videos[video_id])
        obj1 = reorginize_data(object_anno_1, video_info, select_frame)
        obj2 = reorginize_data(object_anno_2, video_info, select_frame)
        return obj1, obj2

    def _get_two_objects_from_2frames(self, same_category=True):
        requeied_data = self._get_specific_video()
        while requeied_data is None or (same_category and requeied_data[0]["categories"] != requeied_data[1]["categories"]):
            requeied_data = self._get_specific_video()
        return requeied_data

    def save_processed_json_file(self, save_dir):
        self._merge_caption_to_annotations()
        new_json_file = {}
        for key in ['info', 'licenses', 'categories']:
            new_json_file.update({key: self.json_file[key]})
        new_json_file['videos'] = []
        for video_id in self.video_ids:
            new_json_file['videos'].append(self.videos[video_id])
        new_json_file['annotations'] = []
        for video_id in self.video_ids:
            new_json_file['annotations'].extend(self.video_id2annotations[video_id])
        with open(save_dir, 'w') as f:
            json.dump(new_json_file, f)
        return

generater = InstructionGenerater()
ytvis_datas = YouTubeVIS_Annotations(json_file='./processed_0.json')

obj1, obj2 = ytvis_datas._get_two_objects_from_2frames()
instruction_data = generater.get_instruction_datas(obj1, obj2)
print(instruction_data)
