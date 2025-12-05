"""Run LongVideoBench samples with the Aria model."""

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AriaForConditionalGeneration, AriaProcessor

from longvideobench_dataset import LongVideoBenchDataset

MODEL_ID = os.getenv("ARIA_MODEL_ID", "rhymes-ai/Aria")
DEVICE_MAP = os.getenv("ARIA_DEVICE_MAP", "auto")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "64"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "9999"))
RESULT_LOG_PATH = os.getenv(
    "ARIA_RESULT_LOG", os.path.join("output", "aria_video_samples.json")
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
        "The following visual input contains "
        f"{frame_count} uniformly sampled frames.\n"
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


def _build_messages(
    timeline: List[str], question: str, candidates: List[str], frame_count: int
) -> Tuple[List[Dict[str, Any]], str]:
    prompt = _build_prompt(timeline, question, candidates, frame_count)
    messages = [{"role": "user", "content": []}]

    for _ in range(frame_count):
        messages[0]["content"].append({"type": "image"})
    messages[0]["content"].append({"type": "text", "text": prompt})

    return messages, prompt


def _run_sample(model, processor, sample: Dict[str, Any]) -> str:
    frames, timeline, question, candidates = _split_inputs(sample["inputs"])
    messages, _ = _build_messages(timeline, question, candidates, len(frames))

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    images = _to_pil_frames(frames)

    inputs = processor(text=text, images=images, return_tensors="pt")
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    inputs = inputs.to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        stop_strings=["<|im_end|>"],
        tokenizer=processor.tokenizer,
        do_sample=True,
        temperature=0.9,
    )
    output_ids = output[0][inputs["input_ids"].shape[1] :]
    response = processor.decode(output_ids, skip_special_tokens=True)
    return response


def run_samples() -> None:
    model = AriaForConditionalGeneration.from_pretrained(
        MODEL_ID, device_map=DEVICE_MAP, torch_dtype=torch.bfloat16
    )
    processor = AriaProcessor.from_pretrained(MODEL_ID)
    model.eval()

    dataset = LongVideoBenchDataset(output_root="./output", max_num_frames=MAX_FRAMES)

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[idx]
        reply = _run_sample(model, processor, sample)

        print(f"\n=== Sample {idx} (ID: {sample.get('id')}) ===")
        print("Correct choice:", sample.get("correct_choice", "@"))
        print("Model reply:", reply)
        _append_sample_result(RESULT_LOG_PATH, idx, sample, reply)


if __name__ == "__main__":
    run_samples()
