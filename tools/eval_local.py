from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    script_path = (
        Path(__file__).resolve().parents[1]
        / "vision_service"
        / "scripts"
        / "eval_price_tag_predictions.py"
    )
    runpy.run_path(str(script_path), run_name="__main__")

