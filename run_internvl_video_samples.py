"""Run LongVideoBench samples with InternVL-Chat-V1-5.

This script mirrors ``run_llava_video_samples.py`` but targets the
``OpenGVLab/InternVL-Chat-V1-5`` model. It consumes frames prepared by
``LongVideoBenchDataset`` instead of re-sampling from raw videos.
"""

import importlib
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from longvideobench_dataset import LongVideoBenchDataset

MODEL_PATH = os.getenv("INTERNVL_MODEL_PATH", "OpenGVLab/InternVL-Chat-V1-5")
DEVICE_MAP = os.getenv("INTERNVL_DEVICE_MAP", "auto")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "64"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "9999"))
RESULT_LOG_PATH = os.getenv(
    "INTERNVL_RESULT_LOG", os.path.join("output", "internvl_video_samples.json")
)
DEFAULT_IMAGE_SIZE = int(os.getenv("INTERNVL_IMAGE_SIZE", "448"))


def _split_inputs(
    inputs: Sequence[Any],
) -> Tuple[List[np.ndarray], List[str], str, List[str]]:
    """Separate frames, subtitles, question, and candidates while keeping order."""

    frames: List[np.ndarray] = []
    timeline: List[str] = []
    question: str = ""
    candidates: List[str] = []

    for item in inputs:
        if isinstance(item, str):
            if item.startswith("Question: "):
                question = item[len("Question: ") :].strip()
            elif item.strip().lower().startswith("answer with"):
                continue
            elif len(item) >= 3 and item[1:3] == ". ":
                candidates.append(item.strip())
            else:
                timeline.append(f"Subtitle: {item.strip()}")
        else:
            frames.append(np.array(item))
            timeline.append(f"Frame {len(frames)}")

    return frames, timeline, question, candidates


def _build_prompt(timeline: List[str], question: str, candidates: List[str], frame_count: int) -> str:
    frame_prefix = "\n".join(f"Frame{i + 1}: <image>" for i in range(frame_count))
    timeline_text = "\n".join(f"{idx + 1}. {entry}" for idx, entry in enumerate(timeline))
    option_text = "" if not candidates else "\nOptions:\n" + "\n".join(candidates)

    return (
        f"{frame_prefix}\n"
        f"The following visual input contains {frame_count} uniformly sampled frames.\n"
        "Subtitles are interleaved in the timeline below to preserve their order relative to the frames.\n"
        f"Timeline:\n{timeline_text}\n"
        f"Question: {question}{option_text}\n"
        "Answer with the option's letter if choices are provided."
    )


def _to_pil_frames(frames: List[np.ndarray]) -> List[Image.Image]:
    if not frames:
        frames = [np.zeros((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3), dtype=np.uint8)]
    return [Image.fromarray(frame.astype(np.uint8)) for frame in frames]


def _append_sample_result(log_path: str, idx: int, sample: Dict[str, Any], reply: str) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    entry = {
        "sample_index": idx,
        "id": sample.get("id"),
        "correct_choice": sample.get("correct_choice", "@"),
        "model_reply": reply,
    }

    existing: List[Dict[str, Any]] = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
                if isinstance(loaded, list):
                    existing = loaded
        except json.JSONDecodeError:
            existing = []

    existing.append(entry)

    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(existing, fp, ensure_ascii=False, indent=2)


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_target_dtype(model: torch.nn.Module) -> torch.dtype:
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.bfloat16

    if model_dtype in (torch.int8, torch.uint8, torch.int32, torch.int64):
        model_dtype = torch.bfloat16
    return model_dtype


def _resolve_preprocess_fns(model, input_size: int):
    transform = None
    dynamic_preprocess = None

    if hasattr(model, "build_transform") and callable(getattr(model, "build_transform")):
        transform = model.build_transform(input_size=input_size)
    if hasattr(model, "dynamic_preprocess") and callable(
        getattr(model, "dynamic_preprocess")
    ):
        dynamic_preprocess = model.dynamic_preprocess

    module_candidates = [
        "transformers.models.internvl.processing_internvl",
        "transformers.models.internvl2.processing_internvl2",
        "internvl.model.internvl_chat",
    ]
    for module_path in module_candidates:
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            continue
        if transform is None and hasattr(module, "build_transform"):
            transform = getattr(module, "build_transform")(input_size=input_size)
        if dynamic_preprocess is None and hasattr(module, "dynamic_preprocess"):
            dynamic_preprocess = getattr(module, "dynamic_preprocess")
        if transform is not None and dynamic_preprocess is not None:
            break

    if transform is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        def _fallback_transform(image: Image.Image) -> torch.Tensor:
            resized = image.resize((input_size, input_size))
            array = np.array(resized).astype("float32") / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            return (tensor - mean) / std

        transform = _fallback_transform

    if dynamic_preprocess is None:

        def _fallback_dynamic_preprocess(image: Image.Image, image_size: int, **_: Any):
            return [image.resize((image_size, image_size))]

        dynamic_preprocess = _fallback_dynamic_preprocess

    return transform, dynamic_preprocess


def _prepare_pixel_values(
    model, frames: List[np.ndarray], input_size: int, max_num: int = 1
) -> Tuple[torch.Tensor, List[int]]:
    pil_frames = _to_pil_frames(frames)
    transform, dynamic_preprocess = _resolve_preprocess_fns(model, input_size)

    pixel_values_list: List[torch.Tensor] = []
    num_patches_list: List[int] = []

    for frame in pil_frames:
        tiles = dynamic_preprocess(frame, image_size=input_size, use_thumbnail=True, max_num=max_num)
        if not isinstance(tiles, list):
            tiles = [tiles]
        transformed_tiles: List[torch.Tensor] = []
        for tile in tiles:
            tensor = transform(tile)
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(np.array(tile)).permute(2, 0, 1).float()
            transformed_tiles.append(tensor)
        pixel_values_list.append(torch.stack(transformed_tiles))
        num_patches_list.append(len(transformed_tiles))

    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list


def _run_sample(model, tokenizer, sample: Dict[str, Any]) -> str:
    frames, timeline, question, candidates = _split_inputs(sample["inputs"])
    prompt = _build_prompt(timeline, question, candidates, len(frames))

    pixel_values, num_patches_list = _prepare_pixel_values(
        model, frames, input_size=DEFAULT_IMAGE_SIZE
    )

    target_device = _get_model_device(model)
    target_dtype = _get_target_dtype(model)
    pixel_values = pixel_values.to(device=target_device, dtype=target_dtype)

    generation_config = getattr(model, "generation_config", None)

    response, _ = model.chat(
        tokenizer,
        pixel_values,
        prompt,
        generation_config=generation_config,
        num_patches_list=num_patches_list,
        history=None,
        return_history=True,
    )
    return response


def run_samples() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=DEVICE_MAP,
    ).eval()

    dataset = LongVideoBenchDataset(output_root="./output", max_num_frames=MAX_FRAMES)

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[idx]
        reply = _run_sample(model, tokenizer, sample)

        print(f"\n=== Sample {idx} (ID: {sample.get('id')}) ===")
        print("Correct choice:", sample.get("correct_choice", "@"))
        print("Model reply:", reply)
        _append_sample_result(RESULT_LOG_PATH, idx, sample, reply)


if __name__ == "__main__":
    run_samples()
