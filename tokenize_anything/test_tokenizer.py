import torch
import numpy as np
from PIL import Image
from .utils import im_rescale
from .utils import im_vstack

from .build_model import model_registry_
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

np.random.seed(1234)
inference_mode = torch.inference_mode()
inference_mode.__enter__()

def show_mask(mask, ax):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    ax.imshow(mask.reshape(mask.shape[-2:] + (1,)) * color.reshape(1, 1, -1))

import sys
sys.path.append("..")


model_type = "tap_vit_l"
checkpoint = "/home/zhangtao19/lmms/LLaVA/work_dirs/tokenize-anything/models/tap_vit_l_03f8ec.pkl"
model = model_registry_[model_type](checkpoint=checkpoint)
model.eval()
model.cuda()

#concept_weights = "/home/zhangtao19/lmms/LLaVA/work_dirs/tokenize-anything/concepts/merged_2560.pkl"
#model.concept_projector.reset_weights(concept_weights)
#model.text_decoder.reset_cache(max_batch_size=8)

img = cv2.imread("/home/zhangtao19/lmms/LLaVA/work_dirs/test.jpg")
vis_img = img.copy()[:, :, ::-1]
plt.figure(figsize=(10, 10))
plt.imshow(vis_img)
# plt.show()

# img_list, img_scales = im_rescale(img, scales=[1024], max_size=1024)
# input_size, original_size = img_list[0].shape, img.shape[:2]
# print(input_size, "<-", original_size, "*", img_scales[0])
#
# img_batch = im_vstack(img_list, fill_value=model.pixel_mean_value, size=(1024, 1024))

processed_images = model.image_processor.preprocess(img)
output = model.foward_for_image_tokenize(processed_images['pixel_values'], grid_size=8,
                                         image_size=processed_images['image_size'],
                                         original_size=processed_images['original_size'])
print(output['sem_embeds'].shape)

# output = model.foward_for_image_tokenize(img_batch, grid_size=8, image_size=input_size[:2], original_size=original_size)
# inputs = model.get_inputs({"img": img_batch})
# inputs.update(model.get_features(inputs))
#
# # inputs["points"] = np.array([[[1050, 600, 1], [0, 0, 4]]], "float32") # (1, 2, 3)
# # inputs["points"][:, :, :2] *= np.array(img_scales, "float32")
#
# grid_size = 8
# offset = 1 / (2 * grid_size)
# points_one_side = np.linspace(offset, 1 - offset, grid_size)
# points_x = np.tile(points_one_side[None, :], (grid_size, 1))
# points_y = np.tile(points_one_side[:, None], (1, grid_size))
# points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2) * input_size[:2][::-1]
# points = points[:, None, :]  # (64, 1, 2)
# labels = np.ones((points.shape[0], 1), dtype=np.int64)  # (64, 1)
# inputs.update({"points": (points, labels)})
#
# # Decode outputs for the point prompt.
# outputs = model.get_outputs(inputs)
#
# # Select final mask.
# iou_score, mask_pred = outputs["iou_pred"], outputs["mask_pred"]
# #iou_score[:, 0] -= 1000.0  # Penalize the score of boundary boxes.
# iou_score = iou_score.flatten(0, 1)
# mask_pred = mask_pred.flatten(0, 1)
# keep = iou_score > 0.8
# mask_index = torch.arange(iou_score.shape[0])[keep]
#
# # Upscale masks to the original image resolution.
# iou_scores, masks = iou_score[mask_index], mask_pred[mask_index]
# masks = model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
# masks = masks[..., : input_size[0], : input_size[1]]
# masks = model.upscale_masks(masks, original_size).gt(0).cpu().numpy()

# # Predict concepts and generate captions.
# sem_tokens, sem_embeds = outputs["sem_tokens"], outputs["sem_embeds"]
# #concepts, scores = model.predict_concept(sem_embeds[mask_index])
# #captions = model.generate_text(sem_tokens[mask_index][:, None, :])

# # Display comprehensive visual understanding.
# text_contents = [v.flatten()[0] for v in (iou_scores, iou_scores, iou_scores)]
# vis_text = "{} ({:.2f}, {:.2f}):".format(*text_contents)


# masks = output['mask_pred']
# plt.figure(figsize=(10,10))
# plt.imshow(vis_img)
# # plt.figtext(0.5, 0.1, vis_text, fontsize=16, ha="center")
# for i in range(masks.shape[0]):
#     show_mask(masks[i:i+1], plt.gca())
# plt.axis('off')
# plt.savefig('/home/zhangtao19/lmms/LLaVA/work_dirs/test_fig.png')

# def expand2square(pil_img, background_color):
#     width, height = pil_img.size
#     if width == height:
#         return pil_img
#     elif width > height:
#         result = Image.new(pil_img.mode, (width, width), background_color)
#         result.paste(pil_img, (0, (width - height) // 2))
#         return result
#     else:
#         result = Image.new(pil_img.mode, (height, height), background_color)
#         result.paste(pil_img, ((height - width) // 2, 0))
#         return result
#
# def preprocess_images(imgs):
#     """Preprocess the inference images."""
#     im_batch, im_shapes, im_scales = [], [], []
#     for img in imgs:
#         scaled_imgs, scales = im_rescale(img, scales=[1024])
#         im_batch.__iadd__(scaled_imgs), im_scales.__iadd__(scales)
#         im_shapes += [x.shape[:2] for x in scaled_imgs]
#     im_batch = im_vstack(im_batch, tokenizer.pixel_mean_value, size=(1024, 1024))
#     im_shapes = np.array(im_shapes)
#     im_scales = np.array(im_scales).reshape((len(im_batch), -1))
#     im_info = np.hstack([im_shapes, im_scales]).astype("float32")
#     return im_batch, im_info
#
# tokenizer = model_registry_['tap_vit_b'](checkpoint='/home/zhangtao19/lmms/LLaVA/work_dirs/tokenize-anything/models/tap_vit_b_b45cbf.pkl')
#
# image = '/home/zhangtao19/lmms/LLaVA/work_dirs/test.jpg'
# image = Image.open(image).convert('RGB')
#
# images = [np.array(image)]
# images, im_info = preprocess_images(images)
# images = torch.Tensor(images)
# input_size = im_info[0, :2].astype("int")
#
# outputs = tokenizer.foward_for_image_tokenize(images, grid_size=8, image_size=1024)
#
# mask_pred = outputs["mask_pred"]
# mask_pred = tokenizer.upscale_masks(mask_pred.unsqueeze(0), images.shape[1:-1])[0]
# masks = mask_pred[:, :input_size[0], :input_size[1]]
# image = np.array(image)
# masks = tokenizer.upscale_masks(masks.unsqueeze(0), image.shape[:2])[0]
# masks = masks.gt(0).cpu().numpy()
#
# for key in outputs.keys():
#     print(outputs[key].shape)
#     if key == 'boxes':
#         print(outputs[key])
#
# # masks = outputs['mask_pred']  # (N, H, W)
# import cv2
# image = '/home/zhangtao19/lmms/LLaVA/work_dirs/test.jpg'
# image = Image.open(image).convert('RGB')
# image = np.array(image, dtype=np.uint8)
# image[masks[33]] = 255
# cv2.imwrite('/home/zhangtao19/lmms/LLaVA/work_dirs/test_result.jpg', image)
# # masks = masks.cpu().numpy().astype(np.uint8) * 255
# #
# # # image[masks[0]] = 255
# # import cv2
# # for i in range(masks.shape[0]):
# #     mask_ = masks[i]
# #     cv2.imwrite('/home/zhangtao19/lmms/LLaVA/work_dirs/test_results/{}.jpg'.format(i), mask_)
# # # cv2.imwrite('/home/zhangtao19/lmms/LLaVA/work_dirs/test_result.jpg', image)





