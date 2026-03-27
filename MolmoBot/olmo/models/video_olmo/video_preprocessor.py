from dataclasses import dataclass, field
import logging
from typing import List, Optional, Union, Tuple, Any, Dict

import numpy as np
from PIL import Image

from olmo.preprocessing.image_preprocessor import ImagePreprocessor
from olmo.preprocessing.multimodal_preprocessor import MultimodalPreprocessor
from olmo.preprocessing.preprocessor_utils import batch_pixels_to_patches, TokenizedVisionData, \
    TensorSpec
from olmo.preprocessing.text_preprocessor import TextPreprocessorConfig
from olmo.tokenizer import HfTokenizerWrapper

from olmo.config import D
from olmo.data.video_loader import VideoFrames, VideoLoader, VideoLoaderConfig

from olmo.preprocessing.multicrop_preprocessor import arange_for_pooling, \
    MultiCropImagePreprocessor, MultiCropConfig

from os import environ
from transformers import AutoProcessor

from olmo.util import interpolate_frame_scores
log = logging.getLogger(__name__)


@dataclass
class MultiModalVideoPreprocessorConfig(VideoLoaderConfig, TextPreprocessorConfig):
    time_mode: str = "per-frame"

    subtitle_mode: str = "frame_1"  # "frame_N", "all", "truncate_N", "ignore"

    max_crops: int = 1
    """Max crops to use for each image"""

    overlap_margins: Tuple[float, float] = (4, 4)
    """Margin to use if building overlapping crops"""

    use_col_tokens: bool = False
    """Add col tokens to each image"""

    periodic_high_res_frame: Optional[int] = None
    """Periodic high resolution frame rate"""

    high_low_train_mode: Optional[str] = "local_rnd"  # Allowed - "periodic", "local_rnd", "global_rnd", "local_rnd_noqsl", "global_rnd_noqsl"
    """Whether to use random high and low resolution frames"""

    high_res_frame_sample_options: Optional[Tuple[int]] = None
    """Switching options for periodic high resolution frame rate"""

    periodic_sample_rate_training: Optional[Dict[int, List[float]]] = field(
        default_factory=lambda: {4: [0.9, 0.03, 0.03, 0.04], 3: [0.6, 0.2, 0.2]}
    )
    """Sampling rate from low to high periodicity"""

    skip_low_res_in_high_low: Optional[bool] = False
    """Whether to skip low resolution frames in high low mode"""

    pooling_w: int = 3
    """pooling w stride"""

    pooling_h: int = 3
    """pooling h stride"""

    high_res_pooling_w: Optional[int] = 3
    """High res pooling w stride"""

    high_res_pooling_h: Optional[int] = 3
    """High res pooling h stride"""

    query_based_resolution_selection: bool = False
    """Whether to use query based resolution selection"""

    max_queries_for_resolution_selection: int = 8
    """Max number of questions to use for query based resolution selection"""

    use_frame_special_tokens: bool = True
    """Whether to use frame special tokens in the video preprocessor"""

    frame_sel_clip_identifier: str = "google/siglip2-so400m-patch14-384"
    """Frame selection model to use for clip based frame selection"""

    image_padding_mask: Union[bool, int] = False

    max_subtitle_tokens: Optional[int] = None

    image: Optional[MultiCropConfig] = None

    topk: Optional[float] = None
    
    prune_from_frame: int = 0

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "legacy_image_mask" in config:
            del config["legacy_image_mask"]
        if "tokenizer" in config:
            del config["tokenizer"]
        if "image" not in config:
            # We assume legacy model did not support image inputs in practice even though they
            # had config options for it
            for k in [
                "crop_mode",
                "max_crops",
                "overlap_margins",
                "max_images",
                "max_multi_image_crops",
            ]:
                if k in config:
                    config.pop(k)
        return config

    def build(self, tokenizer, image_preprocessor, text_seq_len=None, max_sequence_length=None):
        if self.image_padding_mask:
            raise NotImplementedError("Image padding mask is not implemented for VideoOlmo.")
        if self.image is not None:
            image, multi_image = self.image.build_image_preprocessor(
                tokenizer,
                image_preprocessor,
                image_padding_mask=self.image_padding_mask
            )
        else:
            image, multi_image = None, None
        return MultimodalPreprocessor.build(
            text_preprocessor=self.build_text_preprocessor(tokenizer, max_sequence_length),
            image_preprocessor=image,
            multi_image_preprocessor=multi_image,
            video_preprocessor=VideoPreprocessor(
                tokenizer,
                image_preprocessor,
                video_loader=self.build_video_loader(),
                max_crops=self.max_crops,
                overlap_margins=self.overlap_margins,
                use_col_tokens=self.use_col_tokens,
                image_pooling_w=self.pooling_w,
                image_pooling_h=self.pooling_h,
                high_res_pooling_w=self.high_res_pooling_h,
                high_res_pooling_h=self.high_res_pooling_w,
                periodic_high_res_frame=self.periodic_high_res_frame,
                high_res_frame_sample_options=self.high_res_frame_sample_options,
                periodic_sample_rate_training=self.periodic_sample_rate_training,
                skip_low_res_in_high_low=self.skip_low_res_in_high_low,
                frame_sel_clip_identifier=self.frame_sel_clip_identifier,
                high_low_train_mode=self.high_low_train_mode,
                time_mode=self.time_mode,
                subtitle_mode=self.subtitle_mode,
                query_based_resolution_selection=self.query_based_resolution_selection,
                max_queries_for_resolution_selection=self.max_queries_for_resolution_selection,
                use_frame_special_tokens=self.use_frame_special_tokens,
                max_subtitle_tokens=self.max_subtitle_tokens,
                topk=self.topk,
                prune_from_frame=self.prune_from_frame,
            ),
            text_seq_len=text_seq_len
        )


VIDEO_SUBSEGMENT_ID = 10000


@dataclass
class VideoPreprocessor:
    tokenizer: HfTokenizerWrapper
    image_preprocessor: ImagePreprocessor
    video_loader: VideoLoader
    max_crops: int = 1
    overlap_margins: Tuple = (4, 4)
    use_col_tokens: bool = True
    image_pooling_w: int = 2
    image_pooling_h: int = 2
    query_based_resolution_selection: bool = False
    high_res_pooling_w: Optional[int] = None
    high_res_pooling_h: Optional[int] = None
    time_mode: str = "per-frame"
    periodic_high_res_frame: Optional[int] = None
    high_res_frame_sample_options: Optional[Tuple[int]] = None
    periodic_sample_rate_training: Optional[Dict[int, List[float]]] = None
    high_low_train_mode: Optional[str] = ""
    skip_low_res_in_high_low: bool = False
    max_queries_for_resolution_selection: int = 1
    max_frame_prefix_tokens: int = 16
    use_frame_special_tokens: bool = False
    frame_selection_pre_processor=None
    max_subtitle_tokens: Optional[int] = None
    frame_sel_clip_identifier: str = "google/siglip2-so400m-patch14-384"
    subtitle_mode: str = "frame_3"  # "frame_N", "all", "truncate_N", "ignore"
    topk: Optional[float] = None
    prune_from_frame: int = 0

    def __post_init__(self):
        if self.query_based_resolution_selection:
            self.frame_selection_pre_processor = AutoProcessor.from_pretrained(
                self.frame_sel_clip_identifier,
                token=environ.get("HF_ACCESS_TOKEN"),
            )
        if self.subtitle_mode.startswith("truncate"):
            assert self.max_subtitle_tokens is None
            self.max_subtitle_tokens = int(self.subtitle_mode.split("_")[1])
            self.subtitle_mode = "truncate"

        if self.subtitle_mode.startswith("frame_"):
            self.subtitle_time_window = int(self.subtitle_mode.split("_")[1])
            self.subtitle_mode = "frame"
        else:
            self.subtitle_time_window = 0

    @property
    def max_frames(self) -> int:
        return self.video_loader.sampler.max_frames

    def load_video(self, *args, **kwargs):
        return self.video_loader(*args, **kwargs)

    def get_output_shapes(self):
        h, w = self.image_preprocessor.base_image_input_size
        fake_input = VideoFrames(
            np.zeros([self.max_frames, h, w, 3], dtype=np.uint8),
            timestamps=np.arange(self.max_frames)*32.5,
            target_fps=None
        )
        if self.high_res_frame_sample_options is None:
            out = self(fake_input, ["fake query"])
        else:
            assert len(self.high_res_frame_sample_options) % 2 == 0, "Max frame and sample rate options must be even"
            max_periodic_sample_rate = -1
            max_high_res_frames = -1
            for candidate_idx in range(len(self.high_res_frame_sample_options) // 2):
                num_high_res_frames = self.high_res_frame_sample_options[candidate_idx * 2] // self.high_res_frame_sample_options[candidate_idx * 2 + 1]
                if num_high_res_frames > max_high_res_frames:
                    max_high_res_frames = num_high_res_frames

                if self.high_res_frame_sample_options[candidate_idx * 2 + 1] > max_periodic_sample_rate:
                    max_periodic_sample_rate = self.high_res_frame_sample_options[candidate_idx * 2 + 1]

            out = self(fake_input, ["fake query"], high_res_sample_rate_override=max_periodic_sample_rate)

            fake_all_high_input = VideoFrames(
                np.zeros([max_high_res_frames, h, w, 3], dtype=np.uint8),
                timestamps=np.arange(max_high_res_frames) * 32.5,
                target_fps=None
            )
            all_high_out = self(fake_all_high_input, ["fake query"], high_res_sample_rate_override=1)

            # Replace dummy matrix so having more tokens and bigger token pooling matrix even if num of frames lower for all high res setting
            if all_high_out.token_pooling.shape[0] > out.token_pooling.shape[0]:
                out.token_pooling = all_high_out.token_pooling
            if all_high_out.tokens.shape[0] > out.tokens.shape[0]:
                out.tokens = all_high_out.tokens

        return TensorSpec.get_spec(out)

    def image_to_patches_and_tokens(
        self,
        image,
        pooling_h: int,
        pooling_w: int,
        patch_id: int,
        is_training=False,
        rng=None,
    ):
        max_crops = self.max_crops
        overlap_margins = self.overlap_margins
        base_image_input_size = self.image_preprocessor.base_image_input_size
        image_patch_size = self.image_preprocessor.image_patch_size

        if isinstance(base_image_input_size, int):
            base_image_input_size = (base_image_input_size, base_image_input_size)

        base_image_input_d = image_patch_size
        crop_patch_w = base_image_input_size[1] // base_image_input_d
        crop_patch_h = base_image_input_size[0] // base_image_input_d

        if self.max_crops == 1:
            resized, resized_mask, resize_idx = self.image_preprocessor.build_single_crop(image, is_training=is_training, rng=rng)
            resize_idx = np.arange(crop_patch_w*crop_patch_h).reshape([crop_patch_h, crop_patch_w])
            pooling_idx = arange_for_pooling(resize_idx, pooling_h, pooling_w)
            h, w = pooling_idx.shape[:2]
            pooling_idx = pooling_idx.reshape([-1, pooling_h*pooling_w])
            per_row = np.full(
                (w,),
                patch_id,
                dtype=np.int32
            )
            if self.use_col_tokens:
                per_row = np.concatenate([per_row, [self.tokenizer.image_col_token_id]], 0)
            extra_tokens = np.tile(per_row, [h])
            if self.use_frame_special_tokens:
                joint = [
                    [self.tokenizer.frame_start_token_id],
                    extra_tokens,
                    [self.tokenizer.frame_end_token_id],
                ]
            else:
                joint = [
                    [self.tokenizer.image_start_token_id],
                    extra_tokens,
                    [self.tokenizer.image_end_token_id],
                ]
            if resized_mask is not None:
                resized_mask = batch_pixels_to_patches(resized_mask, image_patch_size).mean(-1)
            resized = batch_pixels_to_patches(resized, image_patch_size)
            return np.concatenate(joint, 0), resized, resized_mask, pooling_idx
        else:
            raise NotImplementedError("Multi-crop video")

    @staticmethod
    def process_text_for_siglip(message, metadata):
        if metadata is not None and "frame_sel_input" in metadata:
            message = metadata["frame_sel_input"]

        # Model was trained with lowercased text, so make sure your text labels are preprocessed the same way
        message = message.lower()

        # a prompt template of "This is a photo of {label}." should be passed to the processor
        message = f"This photo has: {message}"

        return message

    def __call__(
        self,
        video_frames: VideoFrames,
        message_list: Union[List[str], List[List[str]]],
        metadata=None,
        is_training=False,
        rng=None,
        high_res_sample_rate_override=None,
    ) -> TokenizedVisionData:
        """
        Interleave video and text tokens into multi-modal features for the model
        """
        video_tokens = []
        frame_id_token_ids = []
        for frame_idx, frame_time in enumerate(video_frames.timestamps):
            if self.time_mode == "numbered-frames":
                prev_space = " " if frame_idx > 0 else ""
                frame_id = prev_space + f"{frame_idx+1}: " # explicit whitespace before/after image tokens
                frame_id_token_ids.append(self.tokenizer.encode(frame_id))
            elif self.time_mode == "per-frame":
                prev_space = " " if frame_idx > 0 else ""
                frame_id = prev_space + f"time {frame_time:.2f} " # explicit whitespace before/after image tokens
                frame_id_token_ids.append(self.tokenizer.encode(frame_id))
            elif self.time_mode == "per-frame-compact":
                prev_space = " " if frame_idx > 0 else ""
                frame_id = prev_space + f"{frame_time:.1f} " # explicit whitespace before/after image tokens
                frame_id_token_ids.append(self.tokenizer.encode(frame_id))
            else:
                frame_id_token_ids.append(None)

        average_time_delta = 1 / video_frames.sampled_fps
        if self.time_mode in ["sampled-fps-prefix", "numbered-frames"]:
            if video_frames.sampling_augmentation:
                prefix = f"Aug={video_frames.sampling_augmentation} FPS={video_frames.sampled_fps:0.2f}"
            else:
                prefix = f"FPS={video_frames.sampled_fps:0.2f}"
            video_tokens.append(self.tokenizer.encode(prefix))
        elif self.time_mode == "time-delta-prefix":
            assert video_frames.sampling_augmentation is None
            prefix = self.tokenizer.encode(f"Sampling Delta {average_time_delta:0.2f}")
            video_tokens.append(prefix)
        elif self.time_mode == "fps-prefix":
            # This mode is for backward compatibility. Don't use for new runs - it pairs the fps and average time delta
            assert video_frames.sampling_augmentation is None
            prefix = self.tokenizer.encode(f"FPS {average_time_delta:0.2f}")
            video_tokens.append(prefix)
        elif self.time_mode == 'none':
            # do nothing
            pass
        elif self.time_mode not in ["per-frame", "per-frame-compact"]:
            raise NotImplementedError(self.time_mode)

        if len(video_tokens) > 0:
            video_token_prefix_len = len(np.concatenate(video_tokens, 0))
        else:
            video_token_prefix_len = 0

        video_masks = []
        all_frame_patches = []
        low_res_pooled_idx = []
        high_res_pooled_idx = []
        low_res_pooled_idx_no_offset = None
        high_res_pooled_idx_no_offset = None
        low_res_token_place_holders = None
        high_res_token_place_holders = None

        if high_res_sample_rate_override is not None:
            high_res_sample_rate = high_res_sample_rate_override
        elif self.high_res_frame_sample_options is not None:
            assert self.periodic_high_res_frame is not None, "Periodic high-res frame acts as flag. Must be set if high-res frame sample options are set"
            candidate_sampling = []
            num_frames = len(video_frames.timestamps)
            for candidate_idx in range(len(self.high_res_frame_sample_options) // 2):
                if num_frames <= self.high_res_frame_sample_options[2 * candidate_idx]:
                    candidate_sampling.append(self.high_res_frame_sample_options[2 * candidate_idx + 1])

            if is_training:
                sorted_candidate_sampling = sorted(candidate_sampling)
                if self.periodic_sample_rate_training is not None and len(sorted_candidate_sampling) in self.periodic_sample_rate_training:
                    sample_weights = self.periodic_sample_rate_training[len(sorted_candidate_sampling)]
                else:
                    sample_weights = None

                high_res_sample_rate = np.random.choice(sorted_candidate_sampling, p=sample_weights)
            else:
                # to maximize tokens in eval, we minimize sparsity after frame are sampled
                high_res_sample_rate = min(candidate_sampling)
        else:
            # if not self.high_res_frame_sample_options, self.periodic_high_res_frame the default high-low behavior
            high_res_sample_rate = self.periodic_high_res_frame

        if high_res_sample_rate is not None:
            high_res_index_list = [1 if frame_idx % high_res_sample_rate == 0 else 0 for frame_idx in range(len(video_frames.timestamps))]
            if is_training and high_res_sample_rate > 1:
                if self.high_low_train_mode.startswith("global_rnd"):
                    rng.shuffle(high_res_index_list)
                elif self.high_low_train_mode.startswith("local_rnd"):
                    for chunk_idx in range(int(np.ceil(len(video_frames.timestamps) / high_res_sample_rate))):
                        chunk = high_res_index_list[chunk_idx * high_res_sample_rate:(chunk_idx + 1) * high_res_sample_rate]
                        rng.shuffle(chunk)
                        high_res_index_list[chunk_idx * high_res_sample_rate:(chunk_idx + 1) * high_res_sample_rate] = chunk
        else:
            # Default - Do no frame selection and no high and low res
            high_res_index_list = [0 for _ in range(len(video_frames.timestamps))]

        for frame_idx, frame in enumerate(video_frames.frames):
            if high_res_index_list[frame_idx] == 1:
                # If the frame is a high res frame, use the high res token length
                frame_pooling_w = self.high_res_pooling_w
                frame_pooling_h = self.high_res_pooling_h
                patch_id = self.tokenizer.image_patch_token_id

                frame_tokens, frame_patches, frame_masks, pooled_idx = self.image_to_patches_and_tokens(
                    frame, frame_pooling_h, frame_pooling_w, patch_id, is_training, rng
                )
            else:
                frame_pooling_w = self.image_pooling_w
                frame_pooling_h = self.image_pooling_h
                if high_res_sample_rate:
                    patch_id = self.tokenizer.image_low_res_token_id
                else:
                    patch_id = self.tokenizer.image_patch_token_id

                frame_tokens, frame_patches, frame_masks, pooled_idx = self.image_to_patches_and_tokens(
                    frame, frame_pooling_h, frame_pooling_w, patch_id, is_training, rng
                )

            offset = sum(np.prod(x.shape[:2]) for x in all_frame_patches)
            pooled_idx_with_offset = np.where(pooled_idx >= 0, pooled_idx + offset, pooled_idx)
            all_frame_patches.append(frame_patches)

            if high_res_index_list[frame_idx] != 1 and self.skip_low_res_in_high_low:
                continue

            if high_res_index_list[frame_idx] == 1:
                high_res_pooled_idx.append(pooled_idx_with_offset)
                high_res_pooled_idx_no_offset = pooled_idx
                high_res_token_place_holders = frame_tokens

            else:
                low_res_pooled_idx.append(pooled_idx_with_offset)
                low_res_pooled_idx_no_offset = pooled_idx
                low_res_token_place_holders = frame_tokens

            video_masks.append(frame_masks)
            if frame_id_token_ids[frame_idx]:
                video_tokens.append(np.array(frame_id_token_ids[frame_idx], dtype=np.int32))
            video_tokens.append(frame_tokens)

        prefix_plus_video_token_len = len(np.concatenate(video_tokens, 0))
        if video_frames.subtitle:
            subtitle = video_frames.subtitle
            if subtitle is not None and self.subtitle_mode != 'ignore':
                if isinstance(subtitle, str):
                    subtitle_str = "subtitle\n" + subtitle
                elif isinstance(subtitle, dict):
                    subtitle_str = "subtitle\n"
                    for (s, e), txt in subtitle.items():
                        if self.subtitle_mode == "frame":
                            for t in video_frames.timestamps:
                                if not (e < (t - self.subtitle_time_window) or (t + self.subtitle_time_window) < s):
                                    subtitle_str += f"{s:.1f} - {e:.1f} {txt}\n"
                                    break
                        else:
                            subtitle_str += f"{s:.1f} - {e:.1f} {txt}\n"
                else:
                    raise ValueError("Subtitle must be a string or a dict")

                subtitle_tokens = self.tokenizer.encode(subtitle_str)
                if self.max_subtitle_tokens is not None:
                    if len(subtitle_tokens) > self.max_subtitle_tokens:
                        subtitle_tokens = subtitle_tokens[:self.max_subtitle_tokens]
                video_tokens.append(subtitle_tokens)

        all_frame_patches = np.concatenate(all_frame_patches, 0)
        video_tokens = np.concatenate(video_tokens, 0)
        
        if self.topk:
            patch_id = self.tokenizer.image_patch_token_id
            num_pruned = int(81 * self.topk * (all_frame_patches.shape[0] - self.prune_from_frame))
            
            if num_pruned > 0:
                patch_indices = np.where(video_tokens == patch_id)[0]
                indices_to_remove = patch_indices[-num_pruned:]
                keep_mask = np.ones(len(video_tokens), dtype=bool)
                # we remove these tokens only to reduce token estimation
                # we need to rearrange them in the model later
                keep_mask[indices_to_remove] = False
                video_tokens = video_tokens[keep_mask]

        data = TokenizedVisionData(
            tokens=video_tokens,
            images=all_frame_patches,
        )
        if self.image_preprocessor.use_image_mask:
            data.image_masks = np.concatenate(video_masks, 0)

        if high_res_sample_rate:
            if high_res_pooled_idx:
                data.token_pooling = np.concatenate(high_res_pooled_idx, 0)
            if low_res_pooled_idx:
                data.low_res_token_pooling = np.concatenate(low_res_pooled_idx, 0)
        else:
            data.token_pooling = np.concatenate(low_res_pooled_idx, 0)

        out = {}
        if self.query_based_resolution_selection:
            out["high_res_indices"] = np.array(high_res_index_list)

            frame_prefix_token_list = []
            for frame_idx in range(len(video_frames.timestamps)):
                prefix_tokens = [-1 for _ in range(self.max_frame_prefix_tokens)]
                if frame_idx == 0:
                    start_plus_fps_len = video_token_prefix_len  # Add fps tokens at the start of the video input
                    prefix_tokens[:start_plus_fps_len] = video_tokens[:start_plus_fps_len]
                    if frame_id_token_ids[frame_idx]:
                        assert len(frame_id_token_ids[frame_idx]) + start_plus_fps_len <= self.max_frame_prefix_tokens, "Frame ID token length exceeds max_frame_prefix_tokens"
                        prefix_tokens[start_plus_fps_len:start_plus_fps_len+len(frame_id_token_ids[frame_idx])] = frame_id_token_ids[frame_idx]
                elif frame_id_token_ids[frame_idx]:
                    assert len(frame_id_token_ids[frame_idx]) <= self.max_frame_prefix_tokens, "Frame ID token length exceeds max_frame_prefix_tokens"
                    prefix_tokens[:len(frame_id_token_ids[frame_idx])] = frame_id_token_ids[frame_idx]

                frame_prefix_token_list.append(prefix_tokens)

            frame_prefix_token_list = np.array(frame_prefix_token_list)
            out["frame_prefix_tokens"] = frame_prefix_token_list

            # Not using padding lens here and creating padded matrices in the preprocessor due to eval issues. Improve in the future.
            siglip_text_valid_mask_list = [-1 for _ in range(self.max_queries_for_resolution_selection)]
            siglip_text_token_ids = np.array([[-1 for _ in range(64)] for _ in range(self.max_queries_for_resolution_selection)])

            if isinstance(message_list[0], str):
                siglip_input_text = self.process_text_for_siglip(message_list[0], metadata)
                input_ids_siglip = self.frame_selection_pre_processor(text=siglip_input_text, padding="max_length", max_length=64)
                input_ids_siglip = input_ids_siglip['input_ids'].numpy()[0]

                eval_input_index = 0
                while len(input_ids_siglip) > 64:
                    if eval_input_index >= self.max_queries_for_resolution_selection:  # Number of eval conditions is capped by max_queries_for_resolution_selection
                        break

                    siglip_text_valid_mask_list[eval_input_index] = 1
                    siglip_text_token_ids[eval_input_index] = np.concatenate([input_ids_siglip[:63], np.array([1])])

                    eval_input_index += 1
                    input_ids_siglip = input_ids_siglip[63:]

                if len(input_ids_siglip) > 0 and eval_input_index < self.max_queries_for_resolution_selection:
                    siglip_text_valid_mask_list[eval_input_index] = 1
                    if len(input_ids_siglip) < 64:
                        final_set_with_padding = np.concatenate([input_ids_siglip, np.array([0 for _ in range(64 - len(input_ids_siglip))])])
                        siglip_text_token_ids[eval_input_index] = final_set_with_padding
                    else:
                        siglip_text_token_ids[eval_input_index] = input_ids_siglip

            else:
                for message_set_idx, message_tuple in enumerate(message_list):
                    if message_set_idx >= self.max_queries_for_resolution_selection:
                        break

                    siglip_text_valid_mask_list[message_set_idx] = 1
                    for message_idx, message in enumerate(message_tuple):
                        siglip_input_text = self.process_text_for_siglip(message, metadata)
                        input_ids_siglip = self.frame_selection_pre_processor(text=siglip_input_text, padding="max_length", truncation=True, max_length=64)
                        siglip_text_token_ids[message_set_idx] = input_ids_siglip['input_ids'].numpy()[0]
                        break

            out['siglip_text_token_ids'] = np.array(siglip_text_token_ids)
            out['siglip_text_valid_mask_list'] = np.array(siglip_text_valid_mask_list)
            out['high_res_pooled_idx_no_offset'] = high_res_pooled_idx_no_offset
            out['high_res_token_place_holders'] = high_res_token_place_holders
            out['frame_time_stamps'] = video_frames.timestamps
            out['high_res_sample_rate'] = np.array([high_res_sample_rate])
            out['prefix_plus_video_token_len'] = np.array([prefix_plus_video_token_len])

            # Added dummy vectors for the collator. Will not be used in training/inference since frame flag is -1
            frame = video_frames.frames[0]
            _, _, _, high_res_pooled_idx = self.image_to_patches_and_tokens(frame, self.high_res_pooling_h, self.high_res_pooling_w, self.tokenizer.image_patch_token_id, is_training, rng)
            low_res_frame_tokens, _, _, low_res_pooled_idx = self.image_to_patches_and_tokens(frame, self.image_pooling_h, self.image_pooling_w, self.tokenizer.image_low_res_token_id, is_training, rng)

            # Avoid token_pooling and low_res_token_pooling collator issues. Tokens will be replicated on train side
            data.token_pooling = np.concatenate([high_res_pooled_idx], 0)
            data.low_res_token_pooling = np.concatenate([low_res_pooled_idx], 0)

            # Can happen if len(video_frames.frames) == 1. Or the preprocessor selects all high res frames
            if low_res_token_place_holders is None:
                low_res_token_place_holders = low_res_frame_tokens
            if low_res_pooled_idx_no_offset is None:
                low_res_pooled_idx_no_offset = low_res_pooled_idx
            out['low_res_token_place_holders'] = low_res_token_place_holders
            out['low_res_pooled_idx_no_offset'] = low_res_pooled_idx_no_offset

            out['scaled_avg_scores'] = np.ones(len(video_frames)) * -1
            if metadata is not None and "scaled_avg_scores" in metadata:
                # Interpolate the scores to match the number of frames
                out['scaled_avg_scores'] = interpolate_frame_scores(metadata["scaled_avg_scores"], len(video_frames))

        data.other_data = out
        return data