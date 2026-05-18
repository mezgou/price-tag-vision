from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentation_assets"

BG = "#f3efe7"
PANEL = "#fffbf5"
PANEL_ALT = "#f8f1e6"
BORDER = "#d9cdbd"
TEXT = "#53483f"
TEXT_STRONG = "#1f1812"
MUTED = "#7f6e61"
GREEN = "#2f8f5b"
GREEN_SOFT = "#d7eadf"
AMBER = "#d98f2b"
AMBER_SOFT = "#f4e0bf"
RED = "#d4584b"
RED_SOFT = "#f5d9d4"
BLUE = "#3f7ca8"
BLUE_SOFT = "#dbe9f4"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    dataset_stats = compute_dataset_stats(ROOT / "my_dataset_yolo")
    train_stats = compute_training_stats(ROOT / "runs" / "detect" / "train" / "results.csv")
    pipeline_stats = compute_pipeline_stats(
        ROOT / "artifacts" / "v5c" / "outputs" / "v5c" / "result.csv",
        ROOT / "artifacts" / "v5c" / "outputs" / "v5c" / "smoke_summary.json",
    )

    copy_real_images()
    create_system_architecture()
    create_cv_pipeline()
    create_dataset_overview(dataset_stats, train_stats)
    create_training_metrics(train_stats)
    create_ui_mock(pipeline_stats)
    create_csv_fragment()
    return 0


def compute_dataset_stats(dataset_root: Path) -> dict[str, float | int]:
    def count_images(split: str) -> int:
        return len(list((dataset_root / "images" / split).glob("*.jpg")))

    def count_boxes(split: str) -> int:
        total = 0
        for path in (dataset_root / "labels" / split).glob("*.txt"):
            with path.open("r", encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        return total

    train_images = count_images("train")
    val_images = count_images("val")
    test_images = count_images("test")
    train_boxes = count_boxes("train")
    val_boxes = count_boxes("val")
    test_boxes = count_boxes("test")
    total_images = train_images + val_images + test_images
    total_boxes = train_boxes + val_boxes + test_boxes
    avg_boxes = total_boxes / total_images if total_images else 0.0

    return {
        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,
        "train_boxes": train_boxes,
        "val_boxes": val_boxes,
        "test_boxes": test_boxes,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "avg_boxes_per_image": avg_boxes,
    }


def compute_training_stats(results_csv: Path) -> dict[str, float | int]:
    df = pd.read_csv(results_csv)
    best_map_idx = df["metrics/mAP50-95(B)"].idxmax()
    best_map50_idx = df["metrics/mAP50(B)"].idxmax()
    last_row = df.iloc[-1]

    return {
        "epochs": int(df["epoch"].max()),
        "best_map_epoch": int(df.loc[best_map_idx, "epoch"]),
        "best_map5095": float(df.loc[best_map_idx, "metrics/mAP50-95(B)"]),
        "best_precision": float(df.loc[best_map_idx, "metrics/precision(B)"]),
        "best_recall": float(df.loc[best_map_idx, "metrics/recall(B)"]),
        "best_map50": float(df.loc[best_map_idx, "metrics/mAP50(B)"]),
        "peak_map50_epoch": int(df.loc[best_map50_idx, "epoch"]),
        "peak_map50": float(df.loc[best_map50_idx, "metrics/mAP50(B)"]),
        "final_precision": float(last_row["metrics/precision(B)"]),
        "final_recall": float(last_row["metrics/recall(B)"]),
        "final_map50": float(last_row["metrics/mAP50(B)"]),
        "final_map5095": float(last_row["metrics/mAP50-95(B)"]),
    }


def compute_pipeline_stats(result_csv: Path, summary_json: Path) -> dict[str, int | float | list[dict[str, str]]]:
    import json

    with result_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with summary_json.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    selected_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("product_name") and row.get("price_card"):
            selected_rows.append(row)
        if len(selected_rows) == 3:
            break

    return {
        "frames_processed": int(summary["stats"]["frames_processed"]),
        "detections_total": int(summary["stats"]["detections_total"]),
        "final_rows": int(summary["stats"]["final_rows"]),
        "duration_ms": int(summary["stats"]["duration_ms"]),
        "rows": selected_rows,
    }


def copy_real_images() -> None:
    copies = {
        ROOT / "artifacts" / "v5_showcase" / "frame_overlay.jpg": OUT / "05_detection_tracking_showcase.jpg",
        ROOT / "artifacts" / "v5_showcase" / "contact_sheet.jpg": OUT / "06_recognition_showcase.jpg",
        ROOT / "artifacts" / "datasets" / "price_tag_yolo_v1" / "qa" / "contact_sheets" / "train_gt_overlays.jpg": OUT / "raw_dataset_overlays.jpg",
        ROOT / "artifacts" / "datasets" / "price_tag_yolo_v1" / "qa" / "contact_sheets" / "crops_gt_examples.jpg": OUT / "raw_dataset_crops.jpg",
        ROOT / "runs" / "detect" / "train" / "results.png": OUT / "raw_yolo_results.png",
        ROOT / "runs" / "detect" / "train" / "confusion_matrix_normalized.png": OUT / "raw_confusion_matrix.png",
    }
    for source, target in copies.items():
        if source.exists():
            shutil.copy2(source, target)


def create_system_architecture() -> None:
    img = Image.new("RGB", (1800, 1020), BG)
    draw = ImageDraw.Draw(img)

    draw_gradient_blobs(draw, img.size)
    title(draw, 80, 60, "Price Tag Vision: product architecture")
    subtitle(
        draw,
        80,
        128,
        "Working asynchronous stack already covers upload, processing, preview, artifact storage, and CSV download.",
    )

    top_boxes = [
        ("Robot shelf video", 80, 230, 220, 120, AMBER_SOFT, AMBER),
        ("Frontend\nReact + Vite", 360, 230, 240, 120, PANEL, GREEN),
        ("Backend API\nFastAPI", 680, 230, 220, 120, PANEL, BLUE),
        ("Worker\nRQ + Redis", 980, 230, 220, 120, PANEL, GREEN),
        ("Vision Service\nFastAPI pipelines", 1280, 230, 280, 120, PANEL, RED),
    ]
    bottom_boxes = [
        ("PostgreSQL\njob state", 650, 470, 220, 110, PANEL_ALT, BLUE),
        ("Redis queue", 940, 470, 180, 110, PANEL_ALT, GREEN),
        ("MinIO\ninputs + outputs", 1260, 470, 240, 110, PANEL_ALT, RED),
        ("Result artifacts\nCSV, preview, crops", 840, 720, 340, 120, GREEN_SOFT, GREEN),
        ("Frontend result view\npreview + CSV download", 1260, 720, 360, 120, PANEL, GREEN),
    ]

    for box in top_boxes + bottom_boxes:
        labeled_box(draw, *box)

    arrow(draw, (300, 290), (360, 290), GREEN)
    arrow(draw, (600, 290), (680, 290), BLUE)
    arrow(draw, (900, 290), (980, 290), GREEN)
    arrow(draw, (1200, 290), (1280, 290), RED)

    arrow(draw, (790, 350), (760, 470), BLUE)
    arrow(draw, (1090, 350), (1030, 470), GREEN)
    arrow(draw, (1420, 350), (1380, 470), RED)

    arrow(draw, (1030, 525), (1030, 720), GREEN)
    arrow(draw, (1380, 525), (1180, 720), RED)
    arrow(draw, (1180, 780), (1260, 780), GREEN)

    paragraph(
        draw,
        80,
        890,
        1600,
        "Presentation message: the team already has a product contour, not just isolated CV experiments. "
        "The vision block can evolve internally while UI, job lifecycle, and output delivery remain stable.",
    )

    img.save(OUT / "01_system_architecture.png", quality=95)


def create_cv_pipeline() -> None:
    img = Image.new("RGB", (1800, 1040), BG)
    draw = ImageDraw.Draw(img)
    draw_gradient_blobs(draw, img.size)

    title(draw, 80, 60, "Canonical CV pipeline for the presentation")
    subtitle(
        draw,
        80,
        128,
        "Recommended story: many internal iterations happened, but the final presentation should show one clean pipeline with proven detection/tracking and catalog-anchored recognition.",
    )

    stages = [
        ("1. Video input", 80, 250, 190, 100, AMBER_SOFT, AMBER),
        ("2. Undistort +\nupright frame space", 310, 250, 220, 100, PANEL, BLUE),
        ("3. YOLOv8m\nprice-tag detection", 570, 250, 200, 100, GREEN_SOFT, GREEN),
        ("4. ByteTrack\ndedup by tag", 810, 250, 180, 100, GREEN_SOFT, GREEN),
        ("5. Best crop\nselection", 1030, 250, 180, 100, PANEL, BLUE),
        ("6. One-pass OCR\nper crop", 1250, 250, 180, 100, RED_SOFT, RED),
        ("7. db_hack catalog\nidentity resolver", 1470, 250, 220, 100, RED_SOFT, RED),
    ]
    for box in stages:
        labeled_box(draw, *box)
    for start, end in zip(stages, stages[1:]):
        arrow(draw, (start[1] + start[3], 300), (end[1], 300), GREEN)

    labeled_box(draw, "8. Row fusion\n+ best frame anchor", 370, 520, 260, 110, PANEL_ALT, BLUE)
    labeled_box(draw, "9. Full 29-column CSV", 690, 520, 220, 110, GREEN_SOFT, GREEN)
    labeled_box(draw, "10. Preview + crops\nfor the UI", 970, 520, 240, 110, PANEL_ALT, BLUE)
    labeled_box(draw, "11. Demo download\nand review", 1270, 520, 230, 110, GREEN_SOFT, GREEN)
    arrow(draw, (630, 575), (690, 575), BLUE)
    arrow(draw, (910, 575), (970, 575), GREEN)
    arrow(draw, (1210, 575), (1270, 575), BLUE)

    section_header(draw, 80, 710, "Proven base")
    pill(draw, 80, 760, 290, 52, "YOLOv8m detector trained locally", GREEN_SOFT, GREEN)
    pill(draw, 390, 760, 290, 52, "ByteTrack removes duplicate rows", GREEN_SOFT, GREEN)
    pill(draw, 700, 760, 310, 52, "Frame geometry mapped back to raw video", BLUE_SOFT, BLUE)

    section_header(draw, 80, 840, "Ideal evolution to emphasize")
    pill(draw, 80, 890, 350, 52, "Catalog-driven product identity via db_hack.csv", RED_SOFT, RED)
    pill(draw, 450, 890, 300, 52, "Fast OCR on crops, not on full 4K frames", RED_SOFT, RED)
    pill(draw, 770, 890, 300, 52, "One clean row per unique shelf tag", GREEN_SOFT, GREEN)

    img.save(OUT / "02_cv_pipeline_target.png", quality=95)


def create_dataset_overview(dataset_stats: dict[str, float | int], train_stats: dict[str, float | int]) -> None:
    img = Image.new("RGB", (1800, 1100), BG)
    draw = ImageDraw.Draw(img)
    draw_gradient_blobs(draw, img.size)
    title(draw, 60, 50, "my_dataset_yolo: detector dataset overview")
    subtitle(
        draw,
        60,
        118,
        "This is the local YOLO-format dataset used to train the shelf price-tag detector that powers the pipeline.",
    )

    left = rounded_panel(img, 60, 200, 500, 820)
    dleft = ImageDraw.Draw(left)
    small_title(dleft, 28, 24, "Dataset facts")
    key_value(dleft, 28, 90, "Class count", "1 class: price tag")
    key_value(dleft, 28, 148, "Images", f"{dataset_stats['total_images']} total")
    key_value(
        dleft,
        28,
        206,
        "Split",
        f"train {dataset_stats['train_images']} / val {dataset_stats['val_images']} / test {dataset_stats['test_images']}",
    )
    key_value(dleft, 28, 264, "Boxes", f"{dataset_stats['total_boxes']} labeled boxes")
    key_value(dleft, 28, 322, "Density", f"{dataset_stats['avg_boxes_per_image']:.1f} boxes per image")
    key_value(dleft, 28, 380, "Training recipe", "YOLOv8m, 100 epochs, imgsz 640, batch 16")
    key_value(
        dleft,
        28,
        438,
        "Best validation point",
        f"mAP50-95 {train_stats['best_map5095']:.3f} at epoch {train_stats['best_map_epoch']}",
    )
    paragraph(
        dleft,
        28,
        530,
        444,
        "Use this slide to explain that the detector was trained on shelf-video frames already formatted for YOLO, then reused by all later pipeline iterations.",
    )
    img.paste(left, (60, 200))

    overlays = Image.open(OUT / "raw_dataset_overlays.jpg").convert("RGB")
    crops = Image.open(OUT / "raw_dataset_crops.jpg").convert("RGB")
    overlays = fit_cover(overlays, (1140, 380))
    crops = fit_cover(crops, (1140, 380))

    panel_top = rounded_panel(img, 620, 200, 1140, 400)
    panel_bottom = rounded_panel(img, 620, 620, 1140, 400)
    panel_top.paste(overlays, (20, 20))
    panel_bottom.paste(crops, (20, 20))
    dt = ImageDraw.Draw(panel_top)
    db = ImageDraw.Draw(panel_bottom)
    chip(dt, 20, 336, "Frames with GT boxes")
    chip(db, 20, 336, "Example cropped tags")
    img.paste(panel_top, (620, 200))
    img.paste(panel_bottom, (620, 620))

    img.save(OUT / "03_dataset_overview.png", quality=95)


def create_training_metrics(train_stats: dict[str, float | int]) -> None:
    results = pd.read_csv(ROOT / "runs" / "detect" / "train" / "results.csv")

    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, 0.9], hspace=0.32, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[:, 2])

    for ax in (ax1, ax2, ax3, ax4, ax5):
        ax.set_facecolor(PANEL)

    ax1.plot(results["epoch"], results["train/box_loss"], color=BLUE, lw=2, label="train box")
    ax1.plot(results["epoch"], results["val/box_loss"], color=AMBER, lw=2, label="val box")
    ax1.set_title("Localization loss", fontsize=14, color=TEXT_STRONG)
    ax1.grid(alpha=0.2)
    ax1.legend(frameon=False)

    ax2.plot(results["epoch"], results["metrics/precision(B)"], color=GREEN, lw=2, label="precision")
    ax2.plot(results["epoch"], results["metrics/recall(B)"], color=BLUE, lw=2, label="recall")
    ax2.set_title("Precision and recall", fontsize=14, color=TEXT_STRONG)
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=False)

    ax3.plot(results["epoch"], results["metrics/mAP50(B)"], color=GREEN, lw=2, label="mAP50")
    ax3.plot(results["epoch"], results["metrics/mAP50-95(B)"], color=RED, lw=2, label="mAP50-95")
    ax3.set_title("Validation quality", fontsize=14, color=TEXT_STRONG)
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(alpha=0.2)
    ax3.legend(frameon=False)

    ax4.plot(results["epoch"], results["train/cls_loss"], color=AMBER, lw=2, label="cls loss")
    ax4.plot(results["epoch"], results["train/dfl_loss"], color=RED, lw=2, label="dfl loss")
    ax4.set_title("Classification / DFL losses", fontsize=14, color=TEXT_STRONG)
    ax4.grid(alpha=0.2)
    ax4.legend(frameon=False)

    ax5.axis("off")
    ax5.text(0.0, 0.98, "YOLOv8m training summary", fontsize=22, fontweight="bold", color=TEXT_STRONG, va="top")
    summary_lines = [
        f"Epochs: {train_stats['epochs']}",
        f"Best mAP50-95: {train_stats['best_map5095']:.3f} @ epoch {train_stats['best_map_epoch']}",
        f"Peak mAP50: {train_stats['peak_map50']:.3f} @ epoch {train_stats['peak_map50_epoch']}",
        f"Best precision: {train_stats['best_precision']:.3f}",
        f"Best recall: {train_stats['best_recall']:.3f}",
        f"Final precision: {train_stats['final_precision']:.3f}",
        f"Final recall: {train_stats['final_recall']:.3f}",
        f"Final mAP50: {train_stats['final_map50']:.3f}",
        f"Final mAP50-95: {train_stats['final_map5095']:.3f}",
    ]
    y = 0.84
    for line in summary_lines:
        ax5.text(0.0, y, line, fontsize=14, color=TEXT, va="top")
        y -= 0.085
    ax5.text(
        0.0,
        0.14,
        "Slide message: detector quality is already stable enough to support the downstream pipeline. "
        "The main remaining challenge is not finding tags, but extracting identity from small, noisy text.",
        fontsize=12,
        color=MUTED,
        wrap=True,
        va="top",
    )

    fig.suptitle(
        "Detector training on my_dataset_yolo",
        fontsize=26,
        color=TEXT_STRONG,
        fontweight="bold",
        x=0.36,
        y=0.98,
    )
    fig.text(
        0.06,
        0.93,
        "Representative training curves for the YOLOv8m price-tag detector used by the pipeline.",
        fontsize=12,
        color=MUTED,
    )
    fig.savefig(OUT / "04_detector_training_metrics.png", bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def create_ui_mock(pipeline_stats: dict[str, int | float | list[dict[str, str]]]) -> None:
    img = Image.new("RGB", (1800, 1080), BG)
    draw = ImageDraw.Draw(img)
    draw_gradient_blobs(draw, img.size)

    title(draw, 60, 44, "Representative UI flow")
    subtitle(
        draw,
        60,
        112,
        "Presentation-safe mock built from the actual frontend language and real pipeline outputs.",
    )

    hero = rounded_panel(img, 60, 180, 1680, 250)
    dh = ImageDraw.Draw(hero)
    small_title(dh, 28, 22, "Async shelf-video processing")
    dh.text((28, 82), "Upload an MP4, create a job, poll status, inspect preview rows, then download CSV.", font=font(20), fill=TEXT)
    dh.text((28, 130), "The working stack already supports upload, background execution, crop previews, and artifact download.", font=font(18), fill=MUTED)
    pill(dh, 28, 176, 220, 42, "MP4 upload", BLUE_SOFT, BLUE)
    pill(dh, 270, 176, 220, 42, "Job polling", AMBER_SOFT, AMBER)
    pill(dh, 512, 176, 220, 42, "Preview rows", GREEN_SOFT, GREEN)
    pill(dh, 754, 176, 220, 42, "CSV download", GREEN_SOFT, GREEN)
    img.paste(hero, (60, 180))

    left = rounded_panel(img, 60, 470, 520, 520)
    dl = ImageDraw.Draw(left)
    small_title(dl, 28, 20, "Upload")
    dropzone(dl, 28, 78, 464, 230)
    info_card(dl, 28, 336, 220, 132, "Video", "clip_close.mp4")
    info_card(dl, 272, 336, 220, 132, "Pipeline", "price_tag_v5")
    info_card(dl, 28, 486, 220, 132, "Frames", str(pipeline_stats["frames_processed"]))
    info_card(dl, 272, 486, 220, 132, "Detections", str(pipeline_stats["detections_total"]))
    img.paste(left, (60, 470))

    middle = rounded_panel(img, 620, 470, 520, 520)
    dm = ImageDraw.Draw(middle)
    small_title(dm, 28, 20, "Current job")
    status_card(dm, 28, 78, 464, 178)
    info_card(dm, 28, 286, 220, 110, "Final rows", str(pipeline_stats["final_rows"]))
    info_card(dm, 272, 286, 220, 110, "Runtime", f"{int(pipeline_stats['duration_ms']) // 1000} s")
    preview_rows = pipeline_stats["rows"]
    preview_table(dm, 28, 418, 464, 70, preview_rows)
    img.paste(middle, (620, 470))

    right = rounded_panel(img, 1180, 470, 560, 520)
    dr = ImageDraw.Draw(right)
    small_title(dr, 28, 20, "Crop gallery")
    crop_panels = load_track_crops()
    x_positions = [28, 200, 372]
    for idx, crop in enumerate(crop_panels[:3]):
        thumb = fit_cover(crop, (150, 320))
        card = Image.new("RGB", (164, 360), PANEL_ALT)
        dcard = ImageDraw.Draw(card)
        rounded_rect(dcard, (0, 0, 163, 359), 24, outline=BORDER, fill=PANEL_ALT)
        card.paste(thumb, (7, 7))
        chip(dcard, 14, 325, f"track {idx + 1}")
        right.paste(card, (x_positions[idx], 88))
    img.paste(right, (1180, 470))

    img.save(OUT / "07_interface_mock.png", quality=95)


def create_csv_fragment() -> None:
    with (ROOT / "artifacts" / "v5c" / "outputs" / "v5c" / "result.csv").open(
        "r", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    curated = []
    for row in rows:
        if row.get("product_name") and row.get("price_card"):
            curated.append(
                [
                    row["filename"],
                    shorten(row["product_name"], 52),
                    row["price_card"] or "-",
                    row["discount_amount"] or "-",
                    row["color"] or "-",
                    row["frame_timestamp"] or "-",
                    f"{row['x_min']},{row['y_min']},{row['x_max']},{row['y_max']}",
                ]
            )
        if len(curated) == 4:
            break

    img = Image.new("RGB", (1800, 900), BG)
    draw = ImageDraw.Draw(img)
    draw_gradient_blobs(draw, img.size)
    title(draw, 60, 50, "Representative CSV fragment")
    subtitle(
        draw,
        60,
        118,
        "The real output schema has 29 columns. For the presentation, show a readable subset and mention that the full CSV also includes QR fields and raw-frame coordinates.",
    )

    panel = rounded_panel(img, 60, 190, 1680, 640)
    dp = ImageDraw.Draw(panel)
    headers = ["filename", "product_name", "price_card", "discount", "color", "frame_ts", "bbox"]
    widths = [180, 640, 150, 150, 120, 120, 260]
    x = 28
    for header, width in zip(headers, widths):
        rounded_rect(dp, (x, 30, x + width, 92), 18, fill=PANEL_ALT, outline=BORDER)
        dp.text((x + 16, 50), header, font=font(18, bold=True), fill=TEXT_STRONG)
        x += width + 12

    y = 116
    for row in curated:
        x = 28
        for value, width in zip(row, widths):
            rounded_rect(dp, (x, y, x + width, y + 94), 18, fill=PANEL, outline=BORDER)
            wrapped = wrap_lines(str(value), font(18), width - 24, max_lines=3)
            dp.multiline_text((x + 12, y + 14), "\n".join(wrapped), font=font(18), fill=TEXT, spacing=6)
            x += width + 12
        y += 108

    note = "Suggested slide wording: one unique row per shelf tag, final CSV is schema-correct and tied to a best timestamp and bounding box."
    dp.text((28, 560), note, font=font(18), fill=MUTED)
    img.paste(panel, (60, 190))
    img.save(OUT / "08_csv_fragment.png", quality=95)


def load_track_crops() -> list[Image.Image]:
    crops = []
    for name in ["trk016.jpg", "trk023.jpg", "trk086.jpg"]:
        path = ROOT / "artifacts" / "v5_showcase" / name
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        crops.append(image.crop((0, 0, min(360, image.width), image.height)))
    return crops


def draw_gradient_blobs(draw: ImageDraw.ImageDraw, size: tuple[int, int]) -> None:
    w, h = size
    draw.ellipse((-120, -100, 620, 460), fill="#ead9c2")
    draw.ellipse((w - 520, h - 440, w + 120, h + 80), fill="#d9e8df")


def rounded_panel(base: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    panel = Image.new("RGB", (w, h), PANEL)
    draw = ImageDraw.Draw(panel)
    rounded_rect(draw, (0, 0, w - 1, h - 1), 30, outline=BORDER, fill=PANEL)
    return panel


def title(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.text((x, y), text, font=font(42, bold=True), fill=TEXT_STRONG)


def subtitle(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.text((x, y), text, font=font(20), fill=MUTED)


def small_title(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.text((x, y), text, font=font(26, bold=True), fill=TEXT_STRONG)


def section_header(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.text((x, y), text, font=font(24, bold=True), fill=TEXT_STRONG)


def paragraph(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, text: str) -> None:
    lines = wrap_lines(text, font(18), width, max_lines=10)
    draw.multiline_text((x, y), "\n".join(lines), font=font(18), fill=TEXT, spacing=7)


def key_value(draw: ImageDraw.ImageDraw, x: int, y: int, key: str, value: str) -> None:
    draw.text((x, y), key, font=font(16, bold=True), fill=MUTED)
    draw.text((x, y + 24), value, font=font(22), fill=TEXT_STRONG)


def pill(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, text: str, fill: str, outline: str) -> None:
    rounded_rect(draw, (x, y, x + w, y + h), h // 2, outline=outline, fill=fill)
    tw = draw.textlength(text, font=font(18, bold=True))
    draw.text((x + (w - tw) / 2, y + 13), text, font=font(18, bold=True), fill=TEXT_STRONG)


def chip(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    w = int(draw.textlength(text, font=font(16, bold=True)) + 26)
    pill(draw, x, y, w, 34, text, GREEN_SOFT, GREEN)


def labeled_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    w: int,
    h: int,
    fill: str,
    outline: str,
) -> None:
    rounded_rect(draw, (x, y, x + w, y + h), 26, fill=fill, outline=outline)
    lines = text.split("\n")
    line_height = 28
    total = len(lines) * line_height
    cy = y + (h - total) / 2
    for line in lines:
        tw = draw.textlength(line, font=font(20, bold=True))
        draw.text((x + (w - tw) / 2, cy), line, font=font(20, bold=True), fill=TEXT_STRONG)
        cy += line_height


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    *,
    outline: str,
    fill: str,
) -> None:
    draw.rounded_rectangle(box, radius=radius, outline=outline, fill=fill, width=2)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
    draw.line([start, end], fill=color, width=6)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ang = math.atan2(dy, dx)
    head = 16
    p1 = (
        end[0] - head * math.cos(ang - math.pi / 6),
        end[1] - head * math.sin(ang - math.pi / 6),
    )
    p2 = (
        end[0] - head * math.cos(ang + math.pi / 6),
        end[1] - head * math.sin(ang + math.pi / 6),
    )
    draw.polygon([end, p1, p2], fill=color)


def dropzone(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), radius=24, outline=GREEN, fill="#f8fff8", width=3)
    draw.text((x + 34, y + 50), "Drop an MP4 here or click to browse", font=font(24, bold=True), fill=TEXT_STRONG)
    lines = wrap_lines(
        "The app creates a background job, polls status every 1.5 seconds, then exposes preview rows and the final CSV.",
        font(18),
        w - 60,
        max_lines=4,
    )
    draw.multiline_text((x + 34, y + 108), "\n".join(lines), font=font(18), fill=MUTED, spacing=7)


def info_card(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, label: str, value: str) -> None:
    rounded_rect(draw, (x, y, x + w, y + h), 22, outline=BORDER, fill=PANEL_ALT)
    draw.text((x + 18, y + 20), label.upper(), font=font(14, bold=True), fill=MUTED)
    lines = wrap_lines(value, font(22, bold=True), w - 30, max_lines=3)
    draw.multiline_text((x + 18, y + 48), "\n".join(lines), font=font(22, bold=True), fill=TEXT_STRONG, spacing=6)


def status_card(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    rounded_rect(draw, (x, y, x + w, y + h), 22, outline=BORDER, fill=PANEL_ALT)
    draw.text((x + 18, y + 16), "clip_close.mp4", font=font(26, bold=True), fill=TEXT_STRONG)
    pill(draw, x + w - 150, y + 18, 120, 38, "succeeded", GREEN_SOFT, GREEN)
    draw.text((x + 18, y + 70), "Pipeline: price_tag_v5 / 0.1.0", font=font(18), fill=MUTED)
    draw.text((x + 18, y + 104), "Stage: completed", font=font(18), fill=TEXT)
    draw.text((x + 18, y + 138), "Message: Vision pipeline completed", font=font(18), fill=TEXT)
    draw.rounded_rectangle((x + 18, y + 162, x + w - 18, y + 178), radius=8, fill="#e4efe8", outline=GREEN, width=2)
    draw.rounded_rectangle((x + 18, y + 162, x + w - 18, y + 178), radius=8, fill=GREEN, outline=GREEN, width=2)


def preview_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    row_h: int,
    rows: list[dict[str, str]],
) -> None:
    headers = ["product", "price card", "discount"]
    col_widths = [240, 100, 100]
    cx = x
    for header, width in zip(headers, col_widths):
        rounded_rect(draw, (cx, y, cx + width, y + row_h), 16, outline=BORDER, fill=PANEL)
        draw.text((cx + 12, y + 22), header, font=font(16, bold=True), fill=TEXT_STRONG)
        cx += width + 12
    yy = y + row_h + 12
    for row in rows[:3]:
        values = [
            shorten(row.get("product_name", ""), 32),
            row.get("price_card", "-"),
            row.get("discount_amount", "-"),
        ]
        cx = x
        for value, width in zip(values, col_widths):
            rounded_rect(draw, (cx, yy, cx + width, yy + row_h), 16, outline=BORDER, fill=PANEL_ALT)
            lines = wrap_lines(value, font(15), width - 20, max_lines=2)
            draw.multiline_text((cx + 10, yy + 16), "\n".join(lines), font=font(15), fill=TEXT, spacing=4)
            cx += width + 12
        yy += row_h + 10


def wrap_lines(text: str, text_font: ImageFont.FreeTypeFont, width: int, max_lines: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    temp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    for word in words[1:]:
        candidate = f"{current} {word}"
        if temp.textlength(candidate, font=text_font) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = shorten(lines[-1], max(8, len(lines[-1]) - 3))
    return lines


def shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def fit_cover(image: Image.Image, target: tuple[int, int]) -> Image.Image:
    tw, th = target
    scale = max(tw / image.width, th / image.height)
    resized = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - tw) // 2)
    top = max(0, (resized.height - th) // 2)
    return resized.crop((left, top, left + tw, top + th))


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    if bold:
        candidates = [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            *candidates,
        ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())
