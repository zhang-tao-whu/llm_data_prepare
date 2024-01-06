import json
import pycocotools.mask as mask_util
import numpy as np
from Osprey_cap.demo.osprey_inference import Osprey
import cv2
import os
from tqdm import tqdm
import random

class YouTubeVIS_Annotations(object):

    def __init__(self, json_file, debug=False, split=None):
        with open(json_file, 'r') as f:
            self.json_file = json.load(f)
        print("loaded ...")
        self.videos = self._get_video_id()
        self.video_id2annotations = self._get_video_annotations()
        self.video_ids = list(self.videos.keys())
        self.video_ids.sort()

        self.class_id2class_name = self._get_class_name()
        print("buided indexes ...")

        if debug:
            max_length = 1
            print(len(self.video_ids))
            self.video_ids = self.video_ids[:max_length]
        else:
            if split is not None:
                start, end = split
                end = min(len(self.video_ids), end)
                self.video_ids = self.video_ids[start: end]

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

    def _rle2mask(self, segm, h, w):
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_util.frPyObjects(segm, h, w)
            rle = mask_util.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = mask_util.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        mask = mask_util.decode([rle])
        return mask

    def get_image_and_annos(self):
        self.video_id2captions = {}
        self.allow_push = False
        for video_id in self.video_ids:
            print("processing video {} ...".format(video_id))
            height, width = self.videos[video_id]["height"], self.videos[video_id]["width"]
            for image_id, image_path in tqdm(enumerate(self.videos[video_id]['file_names'])):
                self.cur_process = (video_id, image_id)
                annotation = self.video_id2annotations[video_id]
                image_annotation = []
                annotation_categories = []
                self.cur_valid = []
                for object_annotation in annotation:
                    if object_annotation['segmentations'][image_id] is None:
                        self.cur_valid.append(False)
                    else:
                        self.cur_valid.append(True)
                        image_annotation.append(self._rle2mask(object_annotation['segmentations'][image_id],
                                                               h=height, w=width))
                        annotation_categories.append(self.class_id2class_name[object_annotation['category_id']])
                self.allow_push = True
                yield image_path, image_annotation, annotation_categories

                while self.allow_push:
                    Warning("get an image and annotation, but have not push the caption")
                    yield None, None, None

    def push_image_captions(self, captions):
        if not self.allow_push:
            Warning("Please get an image and annotation from this dataset")
            return None
        video_id, image_id = self.cur_process
        if video_id not in self.video_id2captions.keys():
            # init for per object
            assert image_id == 0
            self.video_id2captions[video_id] = [[] for item in self.cur_valid]
            cap_id = 0
            for i, valid in enumerate(self.cur_valid):
                if not valid:
                    self.video_id2captions[video_id][i].append(None)
                else:
                    self.video_id2captions[video_id][i].append(captions[cap_id])
                    cap_id += 1
        else:
            # check length
            for item in self.video_id2captions[video_id]:
                assert len(item) == image_id
            cap_id = 0
            for i, valid in enumerate(self.cur_valid):
                if not valid:
                    self.video_id2captions[video_id][i].append(None)
                else:
                    self.video_id2captions[video_id][i].append(captions[cap_id])
                    cap_id += 1
        self.allow_push = False
        return None

    def _merge_caption_to_annotations(self):
        for video_id in self.video_id2captions.keys():
            assert len(self.video_id2captions[video_id]) == len(self.video_id2annotations[video_id])
            for obj_idx in range(len(self.video_id2captions[video_id])):
                self.video_id2annotations[video_id][obj_idx]['caption'] = self.video_id2captions[video_id][obj_idx]
        self.video_id2captions = {}
        return

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

class Mask2Caption(object):
    def __init__(self, osprey_checkpoint, image_dir):
        # self.osprey_model = Osprey(osprey_checkpoint)
        self.image_dir = image_dir

        self.superclsss_questions = {
            'person': ['wearing', 'facing', 'doing', 'posture', 'relation'],
            'animal': ['color', 'facing', 'doing', 'posture', 'relation'],
            'object': ['color', 'orientation', 'relation']
        }

        self.class2superclass = {
            "airplane": 'object',
            "bear": 'animal',
            "bird": 'animal',
            "boat": 'object',
            "car": 'object',
            "cat": 'animal',
            "cow": 'animal',
            "deer": 'animal',
            "dog": 'animal',
            "duck": 'animal',
            "earless_seal": 'animal',
            "elephant": 'animal',
            "fish": 'animal',
            "flying_disc": 'object',
            "fox": 'animal',
            "frog": 'animal',
            "giant_panda": 'animal',
            "giraffe": 'animal',
            "horse": 'animal',
            "leopard": 'animal',
            "lizard": 'animal',
            "monkey": 'animal',
            "motorbike": 'object',
            "mouse": 'animal',
            "parrot": 'animal',
            "person": 'person',
            "rabbit": 'animal',
            "shark": 'animal',
            "skateboard": 'object',
            "snake": 'animal',
            "snowboard": 'object',
            "squirrel": 'animal',
            "surfboard": 'object',
            "tennis_racket": 'object',
            "tiger": 'animal',
            "train": 'object',
            "truck": 'object',
            "turtle": 'animal',
            "whale": 'animal',
            "zebra": 'animal'
        }
        self.osprey_model = Osprey(osprey_checkpoint, pre_categories=list(self.class2superclass.keys()))
    def process_image_masks(self, image_path, masks, categories):
        image_path = os.path.join(self.image_dir, image_path)
        captions = []
        image = cv2.imread(image_path)[:, :, ::-1]
        for mask, category in zip(masks, categories):
            mask = mask[:, :, 0]
            # mask = mask[None, :, :]
            caption = self.osprey_model.osprey_predict_more(
                image, mask, category=category,
                other_questions=self.superclsss_questions[self.class2superclass[category]])
            # caption = self.osprey_model.osprey_predict(image, mask, type='short description')
            captions.append(caption)
        return captions

work_id = 0
need_process_nums = 100

ytvis_annotations = YouTubeVIS_Annotations('./ytvis21/train/instances.json', split=(work_id * need_process_nums, work_id * need_process_nums + need_process_nums))
# ytvis_annotations = YouTubeVIS_Annotations('./ytvis21/train/instances.json', debug=True)
mask2caption = Mask2Caption('./checkpoint_osprey/Osprey-7b/', './ytvis21/train/JPEGImages')


for image_path, image_annotations, annotation_categories in ytvis_annotations.get_image_and_annos():
    if len(image_annotations) != 0:
        captions = mask2caption.process_image_masks(image_path, image_annotations, annotation_categories)
    else:
        captions = []
    ytvis_annotations.push_image_captions(captions)

ytvis_annotations.save_processed_json_file('./processed_{}.json'.format(work_id))

