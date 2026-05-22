from __future__ import annotations

from app.evaluation.price_tag_eval import evaluate_rows, field_matches


def test_field_matches_normalizes_digits_prices_and_text() -> None:
    assert field_matches(field="barcode", pred_value="4 607124 143901", gt_value="4607124143901")
    assert field_matches(field="price_card", pred_value="73,99 ₽", gt_value="73.99")
    assert field_matches(field="product_name", pred_value="Молоко питьевое", gt_value="Молоко, питьевое")


def test_evaluate_rows_matches_by_iou_and_field_accuracy() -> None:
    gt_row = {
        "filename": "video.mp4",
        "x_min": "10",
        "y_min": "10",
        "x_max": "110",
        "y_max": "80",
        "product_name": "Молоко питьевое",
        "price_card": "73.99",
        "barcode": "4607124143901",
    }
    pred_row = {
        "filename": "video.mp4",
        "x_min": "12",
        "y_min": "12",
        "x_max": "109",
        "y_max": "81",
        "product_name": "Молоко, питьевое",
        "price_card": "73,99",
        "barcode": "4 607124 143901",
    }

    report = evaluate_rows(pred_rows=[pred_row], gt_rows=[gt_row])

    assert report["matched_rows"] == 1
    assert report["correct_rows"] == 1
    assert report["score"] == 1.0


def test_evaluate_rows_ignores_catalog_guess_metadata() -> None:
    gt_row = {
        "filename": "video.mp4",
        "x_min": "10",
        "y_min": "10",
        "x_max": "110",
        "y_max": "80",
        "price_card": "1299.99",
    }
    pred_row = {
        "filename": "video.mp4",
        "x_min": "10",
        "y_min": "10",
        "x_max": "110",
        "y_max": "80",
        "price_card": "1299.99",
        "catalog_match_status": "catalog_guess",
        "catalog_guess_name": "Likely Wine",
    }

    report = evaluate_rows(pred_rows=[pred_row], gt_rows=[gt_row])

    assert report["matched_rows"] == 1
    assert report["score"] == 1.0
