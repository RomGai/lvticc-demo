from io import BytesIO
import base64
import os
from typing import Any, Dict, List

from openai import OpenAI
from PIL import Image

from longvideobench_dataset import LongVideoBenchDataset

API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "sk-dGZZdFacB6dZ79DtPLy4rQd9mq0dxgxUBZ2xa4PIHTsNKZdh",
)
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ai.juguang.chat/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gemini-2.0-flash")
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "16"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "3"))

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def to_content(item: Any) -> Dict[str, Any]:
    """Convert dataset items to OpenAI chat content blocks."""
    if isinstance(item, str):
        return {"type": "text", "text": item}

    if isinstance(item, Image.Image):
        buf = BytesIO()
        item.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

    raise TypeError(f"Unsupported content type: {type(item)}")


def run_samples() -> None:
    ds = LongVideoBenchDataset(output_root="./output", max_num_frames=MAX_FRAMES)

    for idx in range(min(NUM_SAMPLES, len(ds))):
        sample = ds[idx]
        content: List[Dict[str, Any]] = [to_content(x) for x in sample["inputs"]]

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            max_tokens=5,
        )

        print("\n=== Sample", idx, "(ID:", sample.get("id"), ") ===")
        print("Correct choice:", sample.get("correct_choice", "@"))
        print("Model reply:", response.choices[0].message.content)


if __name__ == "__main__":
    run_samples()
