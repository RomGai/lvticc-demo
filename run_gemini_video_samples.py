"""Run LongVideoBench samples with Gemini via the Google GenAI API."""

import io
import json
import os
from typing import Any, List, Sequence

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

from longvideobench_dataset import LongVideoBenchDataset

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "64"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "9999"))
RESULT_LOG_PATH = os.getenv(
    "GEMINI_RESULT_LOG", os.path.join("output", "gemini_video_samples.json")
)


def _append_sample_result(log_path: str, idx: int, sample: dict, reply: str) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    entry = {
        "sample_index": idx,
        "id": sample.get("id"),
        "correct_choice": sample.get("correct_choice", "@"),
        "model_reply": reply,
    }

    existing: List[dict] = []
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


def _to_pil_image(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, np.ndarray):
        return Image.fromarray(frame.astype(np.uint8))
    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def _image_part(image: Image.Image) -> types.Part:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def _prepare_contents(inputs: Sequence[Any]) -> List[Any]:
    contents: List[Any] = [
        (
            "You will receive interleaved video frames and subtitle text in order. "
            "Use them to answer the question, replying with the option letter when provided."
        )
    ]

    for item in inputs:
        if isinstance(item, str):
            text = item.strip()
            if text.startswith("Question: "):
                contents.append(f"Question: {text[len('Question: '):].strip()}")
            elif len(text) >= 3 and text[1:3] == ". ":
                contents.append(f"Option {text}")
            elif text.lower().startswith("answer with"):
                contents.append(
                    "Answer with the option's letter if choices are provided."
                )
            else:
                contents.append(f"Subtitle: {text}")
        else:
            try:
                image = _to_pil_image(item)
            except TypeError:
                continue
            contents.append(_image_part(image))

    return contents


def _run_sample(client: genai.Client, sample: dict) -> str:
    contents = _prepare_contents(sample["inputs"])

    response = client.models.generate_content(model=MODEL_NAME, contents=contents)

    if hasattr(response, "text") and response.text:
        return response.text
    return ""


def run_samples() -> None:
    client = genai.Client()
    dataset = LongVideoBenchDataset(output_root="./output", max_num_frames=MAX_FRAMES)

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[idx]
        reply = _run_sample(client, sample)

        print(f"\n=== Sample {idx} (ID: {sample.get('id')}) ===")
        print("Correct choice:", sample.get("correct_choice", "@"))
        print("Model reply:", reply)
        _append_sample_result(RESULT_LOG_PATH, idx, sample, reply)


if __name__ == "__main__":
    run_samples()
