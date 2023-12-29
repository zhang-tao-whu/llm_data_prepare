# ------------------------------------------------------------------------
# Copyright (c) 2023-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Image tokenizer."""

import numpy as np
import torch
from torch import nn
from torchvision.ops.boxes import batched_nms
from ..utils import im_rescale
from ..utils import im_vstack


class ImageTokenizer(nn.Module):
    """Tokenize image regions with visual prompts."""

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        image_decoder,
        concept_projector=None,
        text_tokenizer=None,
        text_decoder=None,
        pixel_mean=(103.53, 116.28, 123.675),
        pixel_std=(57.375, 57.12, 58.395),
    ):
        super(ImageTokenizer, self).__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.image_decoder = image_decoder
        self.concept_projector = concept_projector
        self.text_tokenizer = text_tokenizer
        self.text_decoder = text_decoder
        self.pixel_mean_value = pixel_mean  # BGR order.
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean))
        self.register_buffer("pixel_rsig", torch.Tensor(pixel_std).reciprocal_())
        self.image_processor = TAP_image_processor(pixel_mean_value=self.pixel_mean_value)

    def get_inputs(self, inputs):
        """Return the model inputs.

        Parameters
        ----------
        inputs : dict
            The initial inputs.

        Returns
        -------
        dict
            The model inputs.

        """
        if not isinstance(inputs["img"], torch.Tensor):
            inputs["img"] = torch.from_numpy(inputs["img"])
        if inputs["img"].device != self.pixel_mean.device:
            inputs["img"] = inputs["img"].to(device=self.pixel_mean.device)
        inputs["img"] = inputs["img"].to(dtype=self.pixel_mean.dtype)
        inputs["img"] = inputs["img"].sub(self.pixel_mean).mul_(self.pixel_rsig)
        inputs["img"] = inputs["img"].permute(0, 3, 1, 2)
        return inputs

    def load_model(self, ):
        from ..build_model import load_weights_
        load_weights_(self, self.checkpoint)
        return

    def get_inputs_llava(self, inputs):
        """Return the model inputs.

        Parameters
        ----------
        inputs : dict
            The initial inputs.

        Returns
        -------
        dict
            The model inputs.

        """
        if not isinstance(inputs["img"], torch.Tensor):
            inputs["img"] = torch.from_numpy(inputs["img"])
        if inputs["img"].device != self.pixel_mean.device:
            inputs["img"] = inputs["img"].to(device=self.pixel_mean.device)
        inputs["img"] = inputs["img"].to(dtype=self.pixel_mean.dtype)
        inputs["img"] = inputs["img"].permute(0, 2, 3, 1)
        inputs["img"] = inputs["img"].sub(self.pixel_mean).mul_(self.pixel_rsig)
        inputs["img"] = inputs["img"].permute(0, 3, 1, 2)
        return inputs

    def get_features(self, inputs):
        """Return the image features.

        Parameters
        ----------
        inputs : dict
            The inputs.

        Returns
        -------
        dict
            The image features.

        """
        features = self.image_encoder(inputs["img"])
        img_embeds = features[0].permute(0, 2, 3, 1).unsqueeze_(1)
        return {"features": features, "img_embeds": img_embeds}

    def get_outputs(self, inputs):
        """Return the model outputs.

        Parameters
        ----------
        inputs : dict
            The model inputs.

        Returns
        -------
        dict
            The model outputs.

        """
        inputs.update(self.prompt_encoder(inputs))
        return self.image_decoder(inputs)

    def foward_for_image_tokenize(self, images, grid_size=8, image_size=1024, original_size=None,
                                  iou_threthold=0.8, stable_threthold=0.8, nms_threthold=0.7, input_format='llava'):
        #  images (b, c, h, w)
        if self.is_loaded == False:
            self.load_model()
        assert images.shape[0] == 1
        inputs = {'img': images}
        if input_format != 'llava':
            inputs = self.get_inputs(inputs)
        else:
            inputs = self.get_inputs_llava(inputs)
        # get image feature
        inputs.update(self.get_features(inputs))

        # gen_grid points
        if isinstance(image_size, int):
            image_size = np.array([image_size, image_size])
        offset = 1 / (2 * grid_size)
        points_one_side = np.linspace(offset, 1 - offset, grid_size)
        points_x = np.tile(points_one_side[None, :], (grid_size, 1))
        points_y = np.tile(points_one_side[:, None], (1, grid_size))
        points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2) * image_size[::-1]
        points = points[:, None, :]  # (64, 1, 2)
        labels = np.ones((points.shape[0], 1), dtype=np.int64)  # (64, 1)
        inputs.update({"points": (points, labels)})

        outputs = self.get_outputs(inputs)
        # {"iou_pred" (64, 4), "mask_pred" (64, 4, h, w), "sem_tokens" (64, 4, c), "sem_embeds" (64, 4, 1024)}
        outputs["iou_pred"][:, :1] -= 1000  # remove box out
        keep_index = torch.arange(outputs["iou_pred"].shape[0]), outputs["iou_pred"].argmax(1)  # select the max score
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_index]
        sparse_embeds = inputs['sparse_embeds'][:, 0, :]
        outputs['sparse_embeds'] = sparse_embeds

        # filter according iou score
        keep_iou_score = outputs['iou_pred'] > iou_threthold
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_iou_score]

        # filter according mask stable
        stable_score = calculate_stability_score(
            outputs["mask_pred"],
        )
        keep_stable_score = stable_score > stable_threthold
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_stable_score]

        # perform nms
        # get bbox from mask
        outputs["mask_pred_"] = outputs["mask_pred"].gt(0)
        outputs["boxes"] = batched_mask_to_box(outputs["mask_pred_"])
        # filter none mask
        keep_boxes = outputs['boxes'].sum(dim=-1) != 0
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_boxes]

        keep_by_nms = batched_nms(
            outputs["boxes"].float(),
            outputs["iou_pred"],
            torch.zeros_like(outputs["boxes"][:, 0]),  # categories
            iou_threshold=nms_threthold,
        )
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_by_nms]

        # no need for masks
        # # return outputs_["sem_embeds"]  # (N, 1024)
        # masks = self.upscale_masks(outputs_['mask_pred'][:, None], images.shape[1:-1])
        # masks = masks[..., : image_size[0], : image_size[1]]
        # masks = self.upscale_masks(masks, original_size).gt(0).cpu().numpy()
        # outputs_['mask_pred'] = masks
        del outputs['mask_pred'], outputs["mask_pred_"]
        # use outputs['sem_embeds'] (N, 1024)
        return outputs

    def foward_for_object2part(self, images, object_masks, grid_size=4, image_size=1024, original_size=None,
                               iou_threthold=0.8, stable_threthold=0.8, nms_threthold=0.7):


        #  images (1, h, w, c)
        assert images.shape[0] == 1
        inputs = {'img': images}
        inputs = self.get_inputs(inputs)

        object_boxes = batched_mask_to_box(torch.from_numpy(object_masks).to(self.pixel_mean_value.device))
        object_wh = object_boxes[:, 2:] - object_boxes[:, :2]
        object_left = object_boxes[:, :2]

        # get image feature
        inputs.update(self.get_features(inputs))



        # gen_grid points
        if isinstance(image_size, int):
            image_size = np.array([image_size, image_size])
        offset = 1 / (2 * grid_size)
        points_one_side = np.linspace(offset, 1 - offset, grid_size)
        points_x = np.tile(points_one_side[None, :], (grid_size, 1))
        points_y = np.tile(points_one_side[:, None], (1, grid_size))
        points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2) * image_size[::-1]
        points = points[:, None, :]  # (64, 1, 2)
        labels = np.ones((points.shape[0], 1), dtype=np.int64)  # (64, 1)



        inputs.update({"points": (points, labels)})

        outputs = self.get_outputs(inputs)
        # {"iou_pred" (64, 4), "mask_pred" (64, 4, h, w), "sem_tokens" (64, 4, c), "sem_embeds" (64, 4, 1024)}
        outputs["iou_pred"][:, :1] -= 1000  # remove box out
        keep_index = torch.arange(outputs["iou_pred"].shape[0]), outputs["iou_pred"].argmax(1)  # select the max score
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_index]
        sparse_embeds = inputs['sparse_embeds'][:, 0, :]
        outputs['sparse_embeds'] = sparse_embeds

        # filter according iou score
        keep_iou_score = outputs['iou_pred'] > iou_threthold
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_iou_score]

        # filter according mask stable
        stable_score = calculate_stability_score(
            outputs["mask_pred"],
        )
        keep_stable_score = stable_score > stable_threthold
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_stable_score]

        # perform nms
        # get bbox from mask
        outputs["mask_pred_"] = outputs["mask_pred"].gt(0)
        outputs["boxes"] = batched_mask_to_box(outputs["mask_pred_"])
        # filter none mask
        keep_boxes = outputs['boxes'].sum(dim=-1) != 0
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_boxes]

        keep_by_nms = batched_nms(
            outputs["boxes"].float(),
            outputs["iou_pred"],
            torch.zeros_like(outputs["boxes"][:, 0]),  # categories
            iou_threshold=nms_threthold,
        )
        for key in outputs.keys():
            outputs[key] = outputs[key][keep_by_nms]

        # no need for masks
        # # return outputs_["sem_embeds"]  # (N, 1024)
        # masks = self.upscale_masks(outputs_['mask_pred'][:, None], images.shape[1:-1])
        # masks = masks[..., : image_size[0], : image_size[1]]
        # masks = self.upscale_masks(masks, original_size).gt(0).cpu().numpy()
        # outputs_['mask_pred'] = masks
        del outputs['mask_pred'], outputs["mask_pred_"]
        # use outputs['sem_embeds'] (N, 1024)
        return outputs

    def forward(self, inputs):
        """Define the computation performed at every call.

        Parameters
        ----------
        inputs : dict
            The initial inputs.

        Returns
        -------
        dict
            The model outputs.

        """
        # process the image, such as normalize
        inputs = self.get_inputs(inputs)
        # get image feature
        inputs.update(self.get_features(inputs))
        return self.get_outputs(inputs)

    def upscale_masks(self, masks, size):
        """Upscale masks using bilinear interpolation.

        Parameters
        ----------
        masks : torch.Tensor
            The input masks.
        size : Union[int, Tuple[int]]
            The output size.

        Returns
        -------
        torch.Tensor
            The output masks.

        """
        return nn.functional.interpolate(masks, size, mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def predict_concept(self, visual_embeds, k=1):
        """Predict top-k concepts based on visual embeddings.

        Parameters
        ----------
        visual_embeds: torch.Tensor
            The embeddings to predict visual content.
        k : int, optional, default=1
            The k value.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            The concept scores and indices.

        """
        return self.concept_projector.decode(visual_embeds, k)

    @torch.inference_mode()
    def generate_text(self, visual_tokens, max_gen_len=None, temperature=0):
        """Generate text sequences based on visual tokens.

        Parameters
        ----------
        visual_tokens: torch.Tensor
            The tokens to prompt visual context.
        max_gen_len : int, optional
            The maximum length of the generated text sequences.
        temperature : float, optional
            The temperature for controlling randomness in sampling.

        Returns
        -------
        np.ndarray
            An array of generated texts.

        """
        max_gen_len = max_gen_len or self.text_decoder.max_seq_len
        prompts = self.text_decoder.get_prompts(visual_tokens)
        out_shape = (prompts.size(0), self.text_decoder.max_text_len)
        tokens = np.full(out_shape, self.text_tokenizer.pad_id, "int64")
        tokens[:, 0], prev_pos = self.text_tokenizer.bos_id, 0
        eos_reached = np.array([False] * tokens.shape[0])
        for cur_pos in range(1, max_gen_len):
            decode_seq_len = cur_pos - prev_pos
            x = torch.from_numpy(tokens[:, prev_pos:cur_pos]).to(device=prompts.device)
            logits = self.text_decoder.transformer(prompts, x, prev_pos)
            next_logits = logits[: x.size(0), decode_seq_len - 1]
            if temperature > 0:
                p = nn.functional.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(p, 1).cpu().numpy().flatten()
            else:
                next_token = next_logits.argmax(-1).cpu().numpy()
            tokens[:, cur_pos] = next_token
            eos_reached |= next_token == self.text_tokenizer.eos_id
            prev_pos, logits, next_logits = cur_pos, None, None
            if eos_reached.all():
                break
        return np.array(self.text_tokenizer.detokenize(tokens))


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out

def calculate_stability_score(
    masks: torch.Tensor, mask_threshold=0.0, threshold_offset=1.0
) -> torch.Tensor:
    """
    Computes the stability score for a batch of masks. The stability
    score is the IoU between the binary masks obtained by thresholding
    the predicted mask logits at high and low values.
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    unions = (
        (masks > (mask_threshold - threshold_offset))
        .sum(-1, dtype=torch.int16)
        .sum(-1, dtype=torch.int32)
    )
    return intersections / (unions + 1)

class TAP_image_processor(object):
    def __init__(self, pixel_mean_value):
        self.image_size = 1024
        self.pixel_mean_value = pixel_mean_value

    def preprocess(self, image, return_tensors='pt'):
        image = np.array(image)
        img_list, img_scales = im_rescale(image, scales=[self.image_size], max_size=self.image_size)
        input_size, original_size = img_list[0].shape, image.shape[:2]

        img_batch = im_vstack(img_list, fill_value=self.pixel_mean_value, size=(self.image_size, self.image_size))
        if return_tensors == 'pt':
            img_batch = torch.from_numpy(img_batch)
            img_batch = img_batch.permute(0, 3, 1, 2)
        ret = {'pixel_values': img_batch, 'image_size': input_size[:2], 'original_size': original_size}
        return ret
