"""Run LongVideoBench samples with Gemini via the Google GenAI API."""

import io
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

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
                # Terminal instruction; omit from timeline.
                continue
            elif len(item) >= 3 and item[1:3] == ". ":
                candidates.append(item.strip())
            else:
                timeline.append(f"Subtitle: {item.strip()}")
        else:
            frames.append(np.array(item))
            timeline.append(f"Frame {len(frames)}")

    return frames, timeline, question, candidates


def _build_prompt(
    timeline: List[str], question: str, candidates: List[str], frame_count: int
) -> str:
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


def _to_pil_frames(frames: List[np.ndarray]) -> List[Image.Image]:
    if not frames:
        frames = [np.zeros((336, 336, 3), dtype=np.uint8)]
    return [Image.fromarray(frame.astype(np.uint8)) for frame in frames]


def _image_part(image: Image.Image) -> types.Part:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")


def _prepare_contents(prompt: str, frames: List[np.ndarray]) -> List[Any]:
    contents: List[Any] = [prompt]

    for image in _to_pil_frames(frames):
        contents.append(_image_part(image))

    return contents


def _run_sample(client: genai.Client, sample: Dict[str, Any]) -> str:
    frames, timeline, question, candidates = _split_inputs(sample["inputs"])
    prompt = _build_prompt(timeline, question, candidates, len(frames))
    contents = _prepare_contents(prompt, frames)

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
