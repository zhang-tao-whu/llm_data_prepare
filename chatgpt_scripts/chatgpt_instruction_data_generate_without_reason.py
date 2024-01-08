import json
from chatgpt_scripts.question_answers import justify_positive_answers, justify_questions,\
    justify_negative_answers, justify_negative_answers_briefly, briefly_settings, justify_positive_answers_briefly
import random
import copy

def random_select(data_list):
    length = len(data_list)
    idx = random.randint(0, length - 1)
    return copy.deepcopy(data_list[idx])

class InstructionGeneraterWithoutReason(object):
    def __init__(self):
        self.generated_num = 1

    def _generate_pos_data(self, ):
        conversations = []
        if random.random() < 0.5:
            conversations.append({'from': 'human', "value": random_select(justify_questions) + ' ' + \
                                                            random_select(briefly_settings)})
            conversations.append({'from': 'gpt', "value": random_select(justify_positive_answers_briefly)})
        else:
            conversations.append({'from': 'human', "value": random_select(justify_questions)})
            conversations.append({'from': 'gpt', "value": random_select(justify_positive_answers)})
        return conversations

    def _generate_neg_data(self, ):
        conversations = []
        if random.random() < 0.5:
            conversations.append({'from': 'human', "value": random_select(justify_questions) + ' ' + \
                                                            random_select(briefly_settings)})
            conversations.append({'from': 'gpt', "value": random_select(justify_negative_answers_briefly)})
        else:
            conversations.append({'from': 'human', "value": random_select(justify_questions)})
            conversations.append({'from': 'gpt', "value": random_select(justify_negative_answers)})
        return conversations

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
        # generate positive datas
        ret = []
        positive_instruction_data = self._generate_pos_data()
        ret.append(self._add_other_informations(image_paths=obj1['image_files'], segmentations=obj1['segmentations'],
                                                image_size=[obj1['image_size'], obj1['image_size']],
                                                bboxes=obj1['bboxes'], conversations=positive_instruction_data,
                                                type='positive'))

        positive_instruction_data = self._generate_pos_data()
        ret.append(self._add_other_informations(image_paths=obj2['image_files'], segmentations=obj2['segmentations'],
                                                image_size=[obj2['image_size'], obj2['image_size']],
                                                bboxes=obj2['bboxes'], conversations=positive_instruction_data,
                                                type='positive'))

        # generate negative datas
        negative_instruction_data = self._generate_neg_data()
        ret.append(self._add_other_informations(image_paths=[obj1['image_files'][0], obj2['image_files'][1]],
                                                segmentations=[obj1['segmentations'][0], obj2['segmentations'][1]],
                                                image_size=[obj1['image_size'], obj2['image_size']],
                                                bboxes=[obj1['bboxes'][0], obj2['bboxes'][1]],
                                                conversations=negative_instruction_data, type='negative'))

        negative_instruction_data = self._generate_neg_data()
        ret.append(self._add_other_informations(image_paths=[obj2['image_files'][0], obj1['image_files'][1]],
                                                segmentations=[obj2['segmentations'][0], obj1['segmentations'][1]],
                                                image_size=[obj2['image_size'], obj1['image_size']],
                                                bboxes=[obj2['bboxes'][0], obj1['bboxes'][1]],
                                                conversations=negative_instruction_data, type='negative'))

        self.generated_num += 1
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
            ret['image_size'] = [video_info['width'], video_info['height']]
            class_id = obj_anno["category_id"]
            class_name = self.class_id2class_name[class_id]
            ret["categories"] = class_name
            return ret

        select_idx = random.randint(0, len(self.video_ids) - 1)
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

generater = InstructionGeneraterWithoutReason()
ytvis_datas = YouTubeVIS_Annotations(json_file='./processed_0.json')

while generater.generated_num < 1500:
    if generater.generated_num < 750:
        sample_same_class = True
    else:
        sample_same_class = False
    obj1, obj2 = ytvis_datas._get_two_objects_from_2frames(same_category=sample_same_class)
    instruction_data = generater.get_instruction_datas(obj1, obj2)
    while len(instruction_data) == 0:
        obj1, obj2 = ytvis_datas._get_two_objects_from_2frames()
        instruction_data = generater.get_instruction_datas(obj1, obj2)
    with open('./no_reasoning_datas_v1/{}.json'.format(generater.generated_num), 'w') as f:
        json.dump(instruction_data, f)
    print(generater.generated_num)
