import os
import cv2
import numpy as np
import warnings
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import pandas as pd

warnings.filterwarnings("ignore")

TARGET_DPI = 300


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Improve image quality before detection.
    Denoises, sharpens, and boosts contrast so faint text
    becomes detectable.
    """
    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    return sharpened


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Boost contrast via CLAHE on the L channel (LAB space)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def load_images(file_path: str, dpi: int = TARGET_DPI):
    """Load PDF pages or images, upscaling to target DPI."""
    if file_path.lower().endswith(".pdf"):
        # dpi= ensures small text isn't missed due to low resolution
        return convert_from_path(file_path, dpi=dpi)
    else:
        img = Image.open(file_path).convert("RGB")
        # Upscale small images so detector can find small text
        w, h = img.size
        if max(w, h) < 1500:
            scale = 1500 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return [img]


def boxes_iou(b1, b2) -> float:
    """Compute IoU between two bounding boxes (list of 4 [x,y] points)."""
    def to_rect(b):
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        return min(xs), min(ys), max(xs), max(ys)

    x1, y1, x2, y2 = to_rect(b1)
    x3, y3, x4, y4 = to_rect(b2)

    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def deduplicate_results(results: list, iou_thresh: float = 0.5) -> list:
    """
    Remove duplicate detections from multi-pass OCR.
    Keeps the higher-confidence result when boxes overlap significantly.
    """
    kept = []
    for r in sorted(results, key=lambda x: -x["confidence"]):
        duplicate = any(boxes_iou(r["bbox"], k["bbox"]) > iou_thresh for k in kept)
        if not duplicate:
            kept.append(r)
    return kept


class PaddleOCREngine:
    def __init__(self, use_gpu=False):
        print(f"Loading PaddleOCR (GPU: {use_gpu})...")

        # Pass 1 engine: Standard thresholds
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=use_gpu,
            ocr_version='PP-OCRv4',
            show_log=False,
            det_db_thresh=0.2,
            det_db_box_thresh=0.4,
            det_db_unclip_ratio=2.0,
        )

        # Pass 2 engine: High sensitivity
        self.ocr_sensitive = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=use_gpu,
            ocr_version='PP-OCRv4',
            show_log=False,
            det_db_thresh=0.15,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.5,
        )

        print("✅ PaddleOCR models loaded")

    def _ocr_image(self, img_array: np.ndarray, page_num: int) -> list:
        """Multi-pass OCR on one page image."""
        preprocessed = preprocess_image(img_array)
        enhanced = enhance_contrast(img_array)

        results = []

        # Pass 1
        res1 = self.ocr.ocr(preprocessed, cls=True)
        if res1 and res1[0]:
            for line in res1[0]:
                bbox, (text, conf) = line
                results.append({"page": page_num, "text": text.strip(),
                                 "confidence": float(conf), "bbox": bbox,
                                 "pass": 1})

        # Pass 2
        res2 = self.ocr_sensitive.ocr(enhanced, cls=True)
        if res2 and res2[0]:
            for line in res2[0]:
                bbox, (text, conf) = line
                results.append({"page": page_num, "text": text.strip(),
                                 "confidence": float(conf), "bbox": bbox,
                                 "pass": 2})

        return deduplicate_results(results)

    def _tile_scan(self, img_array: np.ndarray, existing: list,
                   page_num: int, tile_size: int = 800, overlap: int = 100) -> list:
        """Region tiling fallback for sparse text regions."""
        h, w = img_array.shape[:2]
        extra = []

        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
                tile = img_array[y:y2, x:x2]

                # Check existing coverage in this tile
                covered = sum(
                    1 for r in existing
                    if x <= r["bbox"][0][0] <= x2 and y <= r["bbox"][0][1] <= y2
                )

                tile_area = (x2 - x) * (y2 - y)
                coverage_ratio = covered / max(1, tile_area / 10000)

                if coverage_ratio < 0.5:
                    res_tile = self.ocr.ocr(tile, cls=True)
                    if res_tile and res_tile[0]:
                        for line in res_tile[0]:
                            bbox, (text, conf) = line
                            adjusted_bbox = [[p[0] + x, p[1] + y] for p in bbox]
                            extra.append({
                                "page": page_num,
                                "text": text.strip(),
                                "confidence": float(conf),
                                "bbox": adjusted_bbox,
                                "pass": 3,
                            })

        return deduplicate_results(existing + extra)

    def extract_text_with_confidence(self, file_path: str) -> list:
        print(f"Running OCR on: {file_path}")
        images = load_images(file_path)
        all_results = []

        for page_num, pil_img in enumerate(images, start=1):
            img_array = np.array(pil_img)

            page_results = self._ocr_image(img_array, page_num)
            print(f"  Page {page_num}: {len(page_results)} boxes after multi-pass")

            page_results = self._tile_scan(img_array, page_results, page_num)
            print(f"  Page {page_num}: {len(page_results)} boxes after tile scan")

            all_results.extend(page_results)

        df = pd.DataFrame(all_results)
        df.to_csv("output.csv", index=False)
        print(f"\n✅ Total lines extracted: {len(all_results)}")
        return all_results