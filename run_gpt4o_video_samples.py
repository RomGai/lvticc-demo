"""Run LongVideoBench samples with GPT-4o via the OpenRouter API."""

import base64
import io
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests
from PIL import Image

from longvideobench_dataset import LongVideoBenchDataset

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = os.getenv("GPT4O_MODEL_NAME", "openai/gpt-4o")
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
X_TITLE = os.getenv("OPENROUTER_X_TITLE", "")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "64"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "9999"))
RESULT_LOG_PATH = os.getenv(
    "GPT4O_RESULT_LOG", os.path.join("output", "gpt4o_video_samples.json")
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


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _prepare_messages(prompt: str, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

    for image in _to_pil_frames(frames):
        contents.append({"type": "image_url", "image_url": {"url": _image_data_url(image)}})

    return [{"role": "user", "content": contents}]


def _extract_reply(choice: Dict[str, Any]) -> str:
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                part_text = part.get("text")
                if isinstance(part_text, str):
                    text_parts.append(part_text)
        if text_parts:
            return "\n".join(text_parts)

    return ""


def _request_headers() -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    if HTTP_REFERER:
        headers["HTTP-Referer"] = HTTP_REFERER
    if X_TITLE:
        headers["X-Title"] = X_TITLE

    return headers


def _run_sample(session: requests.Session, sample: Dict[str, Any]) -> str:
    frames, timeline, question, candidates = _split_inputs(sample["inputs"])
    prompt = _build_prompt(timeline, question, candidates, len(frames))
    messages = _prepare_messages(prompt, frames)

    response = session.post(
        API_URL,
        headers=_request_headers(),
        json={"model": MODEL_NAME, "messages": messages},
        timeout=120,
    )

    response.raise_for_status()
    payload = response.json()

    choices = payload.get("choices", []) if isinstance(payload, dict) else []
    if not choices:
        return ""

    return _extract_reply(choices[0])


def run_samples() -> None:
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY is required to call the API.")

    session = requests.Session()
    dataset = LongVideoBenchDataset(output_root="./output", max_num_frames=MAX_FRAMES)

    for idx in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[idx]
        reply = _run_sample(session, sample)

        print(f"\n=== Sample {idx} (ID: {sample.get('id')}) ===")
        print("Correct choice:", sample.get("correct_choice", "@"))
        print("Model reply:", reply)
        _append_sample_result(RESULT_LOG_PATH, idx, sample, reply)


if __name__ == "__main__":
    run_samples()
