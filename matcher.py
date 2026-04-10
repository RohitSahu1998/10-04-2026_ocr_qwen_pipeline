import re
import cv2
import json
import csv
import copy
import difflib
import numpy as np
from PIL import Image

try:
    from ocr_engine import load_images
except ImportError:
    # Fallback if ocr_engine isn't available in current path during testing
    from pdf2image import convert_from_path
    def load_images(file_path):
        if file_path.lower().endswith(".pdf"):
            return convert_from_path(file_path, dpi=300)
        return [Image.open(file_path).convert("RGB")]

def clean_alphanumeric(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())

def get_center(bbox):
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (sum(xs) / 4.0, sum(ys) / 4.0)

def merge_bboxes(bboxes):
    if not bboxes: return None
    min_x = min([min(pt[0] for pt in bbox) for bbox in bboxes])
    min_y = min([min(pt[1] for pt in bbox) for bbox in bboxes])
    max_x = max([max(pt[0] for pt in bbox) for bbox in bboxes])
    max_y = max([max(pt[1] for pt in bbox) for bbox in bboxes])
    # Return plain Python floats, not numpy floats
    return [[float(min_x), float(min_y)], [float(max_x), float(min_y)], [float(max_x), float(max_y)], [float(min_x), float(max_y)]]

def group_boxes_by_line(boxes, line_tolerance=None):
    """
    Groups OCR boxes that are on the same visual line (similar vertical center).
    Returns a list of groups, where each group is a list of boxes on the same line.
    line_tolerance: max vertical distance between box centers to be on the same line.
                    Defaults to the average box height of the set.
    """
    if not boxes:
        return []
    
    # Compute average box height to set an adaptive tolerance
    heights = [abs(b['bbox'][2][1] - b['bbox'][0][1]) for b in boxes]
    avg_h = np.mean(heights) if heights else 20
    if line_tolerance is None:
        line_tolerance = avg_h * 0.75  # boxes within 75% of avg height are on same line
    
    # Sort boxes top-to-bottom by vertical center
    sorted_boxes = sorted(boxes, key=lambda b: get_center(b['bbox'])[1])
    
    groups = []
    current_group = [sorted_boxes[0]]
    current_y = get_center(sorted_boxes[0]['bbox'])[1]
    
    for box in sorted_boxes[1:]:
        cy = get_center(box['bbox'])[1]
        if abs(cy - current_y) <= line_tolerance:
            current_group.append(box)
        else:
            groups.append(current_group)
            current_group = [box]
            current_y = cy
    
    groups.append(current_group)
    return groups

def filter_spatial_outliers(boxes):
    """
    Instead of a loose median filter, group boxes by line and return
    only the dominant line group (the one with the most boxes, or the
    one whose combined text best represents the value).
    This prevents multi-line sprawl when a value accidentally matches
    tokens on several different lines.
    """
    if len(boxes) <= 1:
        return boxes
    
    line_groups = group_boxes_by_line(boxes)
    
    if len(line_groups) == 1:
        return boxes  # All on same line — no change needed
    
    # Pick the group with the most boxes (most OCR tokens matched on one line)
    best_group = max(line_groups, key=lambda g: len(g))
    return best_group

def extract_qwen_items(data, path=""):
    """
    Recursively extracts all semantic items. 
    Handles:
    1. Standard string format: {"field": "value"}
    2. Grounded Box format: {"field": {"value": "text", "bbox": [...]}}
    3. Confidence format: {"field": {"value": "text", "confidence": 0.99}}
    """
    results = []
    
    # Check if this item is a dictionary with a nested "value"
    if isinstance(data, dict) and "value" in data:
        val = str(data["value"]).strip()
        if val and val.lower() not in ["none", "-", "null", ""]:
            results.append({
                "field": path, 
                "value": val, 
                "clean": clean_alphanumeric(val), 
                "claimed_boxes": [],
                "qwen_bbox": data.get("bbox"),
                "ai_confidence": data.get("confidence", 0.0) # Capture AI logprob score
            })
        return results

    if isinstance(data, dict):
        for k, v in data.items():
            new_path = f"{path}.{k}" if path else k
            results.extend(extract_qwen_items(v, new_path))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]"
            results.extend(extract_qwen_items(item, new_path))
    else:
        val = str(data).strip()
        if val and val.lower() not in ["none", "-", "null", ""]:
            results.append({
                "field": path, 
                "value": val, 
                "clean": clean_alphanumeric(val), 
                "claimed_boxes": [],
                "qwen_bbox": None,
                "ai_confidence": 0.0
            })
    return results

def get_match_weight(ocr_text, qwen_text):
    ocr_clean = clean_alphanumeric(ocr_text)
    q_clean = clean_alphanumeric(qwen_text)
    
    if not ocr_clean or not q_clean: return False, 0
    if ocr_clean == q_clean: return True, 4 
    
    ocr_words = str(ocr_text).split()
    for w in ocr_words:
        w_clean = clean_alphanumeric(w)
        if w_clean and w_clean == q_clean: 
            return True, 3 
            
    if ocr_clean in q_clean and len(ocr_clean) >= 2: return True, 2  
    if q_clean in ocr_clean and len(q_clean) >= 3: return True, 2 
    
    for w in ocr_words:
        w_clean = clean_alphanumeric(w)
        if w_clean and len(w_clean) >= 3 and w_clean in q_clean:
            return True, 1
            
    if len(q_clean) >= 5 and len(ocr_clean) >= 4:
        ratio = difflib.SequenceMatcher(None, ocr_clean, q_clean).ratio()
        if ratio > 0.75: 
            return True, 0.5 
            
    return False, 0

def match_single_page(qwen_page_dict, ocr_page_list):
    """
    Matches Qwen extracted items to OCR boxes.
    IMPORTANT: Deep-copies the OCR list so the original data is not mutated.
    """
    qwen_items = extract_qwen_items(qwen_page_dict)
    
    # Deep copy to avoid mutating the original OCR data
    ocr_working = copy.deepcopy(ocr_page_list)
    
    for i, box in enumerate(ocr_working):
        box['id'] = i
        box['candidates_raw'] = []
        box['candidates'] = []
        
    for box in ocr_working:
        for q in qwen_items:
            matched, weight = get_match_weight(box.get('text', ''), q['value'])
            if matched:
                box['candidates_raw'].append((q, weight))
                
        if box['candidates_raw']:
            max_w = max(c[1] for c in box['candidates_raw'])
            box['candidates'] = [c[0] for c in box['candidates_raw'] if c[1] == max_w]
                
    for box in ocr_working:
        if len(box['candidates']) == 1:
            q = box['candidates'][0]
            q['claimed_boxes'].append(box)
            box['assigned'] = q
        else:
            box['assigned'] = None
            
    for box in ocr_working:
        if len(box['candidates']) > 1 and box['assigned'] is None:
            hungry_candidates = [q for q in box['candidates'] if not q['claimed_boxes']]
            search_pool = hungry_candidates if hungry_candidates else box['candidates']
            
            best_q = None
            min_dist = float('inf')
            b_center = get_center(box['bbox'])
            
            for q in search_pool:
                if q['claimed_boxes']:
                    dist = min(((b_center[0] - get_center(ab['bbox'])[0])**2 + (b_center[1] - get_center(ab['bbox'])[1])**2)**0.5 for ab in q['claimed_boxes'])
                    if dist < min_dist:
                        min_dist = dist
                        best_q = q
            
            if best_q is None:
                best_q = search_pool[0]
                    
            best_q['claimed_boxes'].append(box)
            box['assigned'] = best_q
            
    final_output = []
    for q in qwen_items:
        if q['claimed_boxes']:
            sane_boxes = filter_spatial_outliers(q['claimed_boxes'])
            bboxes = [b['bbox'] for b in sane_boxes]
            final_bbox = merge_bboxes(bboxes)
            
            # Calculate average OCR confidence for these matched boxes
            conf_scores = [b.get('confidence', 0) for b in sane_boxes]
            avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0
            
            final_output.append({
                "field": q['field'],
                "qwen_value": q['value'],
                "bbox": final_bbox,
                "ocr_confidence": avg_conf,
                "ai_confidence": q.get('ai_confidence', 0.0),
                "matched_ocr_text": " | ".join([b['text'] for b in sane_boxes]),
                "all_line_bboxes": _compute_per_line_bboxes(q['claimed_boxes']),
                "qwen_native_bbox": q["qwen_bbox"] # Passthrough Native Box if it exists
            })
        else:
            final_output.append({
                "field": q['field'], 
                "qwen_value": q['value'], 
                "bbox": None, 
                "ocr_confidence": 0.0,
                "ai_confidence": q.get('ai_confidence', 0.0),
                "matched_ocr_text": None,
                "all_line_bboxes": [],
                "qwen_native_bbox": q["qwen_bbox"]
            })
            
    return final_output


def _compute_per_line_bboxes(claimed_boxes):
    """Helper: returns a list of merged bboxes, one per visual line."""
    if not claimed_boxes:
        return []
    groups = group_boxes_by_line(claimed_boxes)
    result = []
    for grp in groups:
        merged = merge_bboxes([b['bbox'] for b in grp])
        if merged:
            result.append(merged)
    return result

def export_to_csv(all_matched_results, output_csv_path="matched_data.csv"):
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Page", "Field", "Qwen_Value", "AI_Confidence", "OCR_Confidence", "OCR_Matched_Text", "OCR_Bounding_Box", "Qwen_Native_Bounding_Box"])
        for res in all_matched_results:
            writer.writerow([
                res.get('page', 'Unknown'),
                res['field'],
                res['qwen_value'],
                f"{res.get('ai_confidence', 0):.4f}",
                f"{res.get('ocr_confidence', 0):.4f}",
                res['matched_ocr_text'] if res['matched_ocr_text'] else "NO MATCH",
                str(res['bbox']) if res['bbox'] else "None",
                str(res['qwen_native_bbox']) if res['qwen_native_bbox'] else "None"
            ])

def highlight_and_save_pdf(input_document_path, qwen_full_data, ocr_full_data, output_pdf_path="highlighted_output.pdf"):
    print(f"Loading document: {input_document_path}")
    pil_images = load_images(input_document_path)
        
    annotated_pil_images = []
    all_pages_results = []
    
    for page_index, pil_img in enumerate(pil_images):
        page_num = page_index + 1
        print(f"\n--- Processing Page {page_num} ---")
        
        qwen_page_dict = qwen_full_data.get(f"page_{page_num}", qwen_full_data) 
        ocr_page_list = [box for box in ocr_full_data if box.get('page') == page_num]
        
        if not ocr_page_list: ocr_page_list = ocr_full_data 
            
        matched_results = match_single_page(qwen_page_dict, ocr_page_list)
        
        for res in matched_results:
            res['page'] = page_num
            status = "[MATCH]" if res['bbox'] else "[MISS ]"
            print(f"{status} | Field: {res['field']} | Qwen: '{res['qwen_value']}' -> OCR: '{res['matched_ocr_text']}'")
            
        all_pages_results.extend(matched_results)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        h_img, w_img = cv_img.shape[:2]
        pad = 4

        for res in matched_results:
            if not res['bbox']:
                continue

            # Use per-line bboxes to avoid one giant merged rectangle
            line_bboxes = res.get('all_line_bboxes') or [res['bbox']]

            first_rx1, first_ry1 = None, None

            for line_bbox in line_bboxes:
                lx1 = max(0, int(line_bbox[0][0]) - pad)
                ly1 = max(0, int(line_bbox[0][1]) - pad)
                lx2 = min(w_img, int(line_bbox[2][0]) + pad)
                ly2 = min(h_img, int(line_bbox[2][1]) + pad)

                if first_rx1 is None:
                    first_rx1, first_ry1 = lx1, ly1

                # Semi-transparent green fill
                overlay = cv_img.copy()
                cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (0, 255, 100), -1)
                cv2.addWeighted(overlay, 0.22, cv_img, 0.78, 0, cv_img)

                # Solid green border
                cv2.rectangle(cv_img, (lx1, ly1), (lx2, ly2), (0, 200, 0), 2)

            # Label drawn once above the topmost line box
            if first_rx1 is not None:
                label = res['field']
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                yd = max(th + 4, first_ry1 - 2)
                cv2.rectangle(cv_img, (first_rx1, yd - th - 4), (first_rx1 + tw + 6, yd + 2), (255, 255, 255), -1)
                cv2.rectangle(cv_img, (first_rx1, yd - th - 4), (first_rx1 + tw + 6, yd + 2), (0, 180, 0), 1)
                cv2.putText(cv_img, label, (first_rx1 + 3, yd - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

        annotated_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        annotated_pil_images.append(annotated_pil)

    csv_path = output_pdf_path.replace(".pdf", ".csv").replace(".jpg", ".csv")
    export_to_csv(all_pages_results, csv_path)

    if annotated_pil_images:
        annotated_pil_images[0].save(output_pdf_path, save_all=True, append_images=annotated_pil_images[1:])
        print(f"\n✅ Final visual document written to -> {output_pdf_path}")

    return all_pages_results


def highlight_single_field(pil_image, field_result):
    """
    Takes a clean PIL image and ONE matched result dict.
    Draws tight per-line highlighted bounding boxes around ONLY that field's text.
    Uses all_line_bboxes (computed at match time) — one box per visual line —
    to avoid a single giant rectangle spanning the entire card/block.
    Falls back to the merged bbox if per-line data is unavailable.
    """
    bbox = field_result.get('bbox')
    if not bbox:
        return pil_image.copy()

    # Convert PIL -> OpenCV BGR
    rgb_img = pil_image.convert("RGB")
    cv_img = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

    # Prefer per-line bboxes to avoid one giant merged box
    line_bboxes = field_result.get('all_line_bboxes') or [bbox]

    pad = 6
    h_img, w_img = cv_img.shape[:2]

    first_rx1, first_ry1 = None, None
    last_ry2 = 0

    for line_bbox in line_bboxes:
        x1, y1 = int(line_bbox[0][0]), int(line_bbox[0][1])
        x2, y2 = int(line_bbox[2][0]), int(line_bbox[2][1])

        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w_img, x2 + pad)
        ry2 = min(h_img, y2 + pad)

        # Track the topmost left corner for the label
        if first_rx1 is None:
            first_rx1, first_ry1 = rx1, ry1
        last_ry2 = max(last_ry2, ry2)

        # --- Semi-transparent green fill per line ---
        overlay = cv_img.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 100), -1)
        cv2.addWeighted(overlay, 0.25, cv_img, 0.75, 0, cv_img)

        # --- Solid bright green border per line ---
        cv2.rectangle(cv_img, (rx1, ry1), (rx2, ry2), (0, 200, 0), 3)

    # --- Field label drawn once above the TOPMOST box ---
    label = field_result.get('field', '')
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    lx = first_rx1
    label_y = max(th + baseline + 10, first_ry1 - 10)
    label_bg_y1 = label_y - th - baseline - 4
    label_bg_y2 = label_y + 4 
    label_bg_x2 = lx + tw + 12

    cv2.rectangle(cv_img, (lx, label_bg_y1), (label_bg_x2, label_bg_y2), (255, 255, 255), -1)
    cv2.rectangle(cv_img, (lx, label_bg_y1), (label_bg_x2, label_bg_y2), (0, 200, 0), 2)
    cv2.putText(cv_img, label, (lx + 6, label_y - 2), font, font_scale, (0, 0, 200), thickness)

    # --- Value annotation below the BOTTOMMOST box ---
    value = field_result.get('qwen_value', '')
    if value:
        val_font_scale = 0.55
        val_thickness = 1
        (vw, vh), vbaseline = cv2.getTextSize(value, font, val_font_scale, val_thickness)
        val_y = last_ry2 + vh + vbaseline + 8

        if val_y + 4 < h_img:
            cv2.rectangle(cv_img, (lx, val_y - vh - 4), (lx + vw + 8, val_y + vbaseline + 4), (255, 255, 230), -1)
            cv2.rectangle(cv_img, (lx, val_y - vh - 4), (lx + vw + 8, val_y + vbaseline + 4), (0, 180, 0), 1)
            cv2.putText(cv_img, value, (lx + 4, val_y), font, val_font_scale, (0, 100, 0), val_thickness)

    result_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return result_pil
