import json
import pycocotools.mask as mask_util
import numpy as np

class YouTubeVIS_Annotations(object):

    def __init__(self, json_file, debug=False):
        with open(json_file, 'r') as f:
            self.json_file = json.load(f)
        print("loaded ...")
        self.videos = self._get_video_id()
        self.video_id2annotations = self._get_video_annotations()
        self.video_ids = list(self.videos.keys())
        self.video_ids.sort()
        print("buided indexes ...")

        if debug:
            max_length = 3
            self.video_ids = self.video_ids[:max_length]

    def _get_video_id(self):
        ret = {}
        for video in self.json_file['videos']:
            ret[video['id']] = video
        return ret

    def _get_video_annotations(self):
        ret = {}
        for annotation in self.json_file['annotations']:
            if annotation['video_id'] in ret.keys():
                ret[annotation['video_id']].append(annotation)
            else:
                ret[annotation['video_id']] = [annotation]
        return ret

    def _rle2mask(self, rles):
        mask = mask_util.decode([rles])
        print(mask.shape)
        return mask

    def get_image_and_annos(self):
        self.video_id2captions = {}
        self.allow_push = False
        for video_id in self.video_ids:
            for image_id, image_path in enumerate(self.videos[video_id]['file_names']):
                self.cur_process = (video_id, image_id)
                annotation = self.video_id2annotations[video_id]
                image_annotation = []
                self.cur_valid = []
                for object_annotation in annotation:
                    if object_annotation['segmentations'][image_id] is None:
                        self.cur_valid.append(False)
                    else:
                        self.cur_valid.append(True)
                        image_annotation.append(self._rle2mask(object_annotation['segmentations'][image_id]))
                self.allow_push = True
                yield image_path, image_annotation

                while self.allow_push:
                    Warning("get an image and annotation, but have not push the caption")
                    yield None, None

    def push_image_captions(self, captions):
        if not self.allow_push:
            Warning("Please get an image and annotation from this dataset")
            return None
        video_id, image_id = self.cur_process
        if video_id not in self.video_id2captions.keys():
            # init for per object
            assert image_id == 0
            self.video_id2annotations[video_id] = [[] for item in self.cur_valid]
            cap_id = 0
            for i, valid in enumerate(self.cur_valid):
                if not valid:
                    self.video_id2annotations[video_id][i].append(None)
                else:
                    self.video_id2annotations[video_id][i].append(captions[cap_id])
                    cap_id += 1
        else:
            # check length
            for item in self.video_id2annotations[video_id]:
                assert len(item) == image_id
            cap_id = 0
            for i, valid in enumerate(self.cur_valid):
                if not valid:
                    self.video_id2annotations[video_id][i].append(None)
                else:
                    self.video_id2annotations[video_id][i].append(captions[cap_id])
                    cap_id += 1
        self.allow_push = False
        return None

    def _merge_caption_to_annotations(self):
        for video_id in self.video_id2captions.keys():
            assert len(self.video_id2captions[video_id]) == len(self.video_id2annotations[video_id])
            for obj_idx in range(len(self.video_id2captions[video_id])):
                self.video_id2annotations[video_id][obj_idx]['caption'] = self.video_id2captions[video_id][obj_idx]
        return

    def save_processed_json_file(self, save_dir):
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


ytvis_annotations = YouTubeVIS_Annotations('./ytvis21/train/instances.json')

for image_path, image_annotations in ytvis_annotations.get_image_and_annos():
    print(image_path, '  ', len(image_annotations), '  ')
    if len(image_annotations) != 0:
        print(image_annotations[0].shape)
    captions = ['this is for test'] * len(image_annotations)
    ytvis_annotations.push_image_captions(captions)

ytvis_annotations.save_processed_json_file('./test_out.json')

