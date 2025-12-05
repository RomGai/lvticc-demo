"""Run LongVideoBench samples with mPLUG-Owl3."""

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from longvideobench_dataset import LongVideoBenchDataset

MODEL_PATH = os.getenv("OWL3_MODEL_PATH", "mPLUG/mPLUG-Owl3-7B-240728")
DEVICE = os.getenv("OWL3_DEVICE", "cuda")
DEVICE_MAP = os.getenv("OWL3_DEVICE_MAP", "auto")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "64"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "9999"))
RESULT_LOG_PATH = os.getenv(
    "OWL3_RESULT_LOG", os.path.join("output", "owl3_video_samples.json")
)


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
    timeline_text = "\n".join(f"{idx + 1}. {entry}" for idx, entry in enumerate(timeline))

    option_text = "" if not candidates else "\nOptions:\n" + "\n".join(candidates)

    return (
        "<|video|>\n"
        f"The following visual input contains {frame_count} uniformly sampled frames.\n"
        "Subtitles are interleaved in the timeline below to preserve their order relative to the frames.\n"
        f"Timeline:\n{timeline_text}\n"
        f"Question: {question}{option_text}\n"
        "Answer with the option's letter if choices are provided."
    )


def _to_pil_frames(frames: List[np.ndarray]) -> List[Image.Image]:
    if not frames:
        frames = [np.zeros((336, 336, 3), dtype=np.uint8)]
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


def _prepare_inputs(tokenizer, processor, prompt: str, frames: List[np.ndarray]):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""},
    ]

    pil_frames = _to_pil_frames(frames)
    inputs = processor(messages=messages, images=None, videos=[pil_frames])

    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(DEVICE)

    inputs.update(
        {
            "tokenizer": tokenizer,
            "max_new_tokens": 256,
            "decode_text": True,
        }
    )
    return inputs


def _run_sample(tokenizer, processor, model, sample: Dict[str, Any]) -> str:
    frames, timeline, question, candidates = _split_inputs(sample["inputs"])
    prompt = _build_prompt(timeline, question, candidates, len(frames))

    inputs = _prepare_inputs(tokenizer, processor, prompt, frames)

    result = model.generate(**inputs)
    if isinstance(result, str):
        return result
    if isinstance(result, list) and result:
        return str(result[0])
    return ""


def run_samples() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE_MAP,
    )
    processor = model.init_processor(tokenizer)
    model.eval()

    dataset = LongVideoBenchDataset(output_root="./output", max_num_frames=MAX_FRAMES)

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[idx]
        reply = _run_sample(tokenizer, processor, model, sample)

        print(f"\n=== Sample {idx} (ID: {sample.get('id')}) ===")
        print("Correct choice:", sample.get("correct_choice", "@"))
        print("Model reply:", reply)
        _append_sample_result(RESULT_LOG_PATH, idx, sample, reply)


if __name__ == "__main__":
    run_samples()
