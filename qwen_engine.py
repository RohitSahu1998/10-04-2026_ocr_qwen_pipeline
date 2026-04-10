import torch
import json
import math
import re
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_path
from PIL import Image


# ============================================================
# LOAD IMAGES
# ============================================================
def load_images(file_path):
    """
    Converts input into list of images.
    - PDF → list of page images
    - Image → single image in list
    """
    if file_path.lower().endswith(".pdf"):
        return convert_from_path(file_path)
    else:
        return [Image.open(file_path).convert("RGB")]


# ============================================================
# MAIN CLASS
# ============================================================
class QwenExtractor:

    def __init__(self, model_path="/home/rohit.sahu/Qwen_model/qwen_models/Qwen2.5-VL-3B-Instruct"):

        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen model on {self.device}...")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True
        )

        print("✅ Qwen Model loaded successfully")

        # Prompt
        self.prompt = """
        You are a highly accurate OCR data extraction system.

        RULES:
        1. Extract ONLY structured data.
        2. CPT codes must be code only (no description).
        3. Charges must be numeric values only.
        4. Return ONLY valid JSON.

        {
          "claimant_name": "",
          "claimant_number": "",
          "tax_id": "",
          "practice_address": "",
          "billing_address": "",
          "diagnosis_codes": [],
          "date_of_service": "",
          "cpt_codes": [],
          "charges": [],
          "units": [],
          "invoice_date": "",
          "invoice_number": "",
          "taxonomy": "",
          "total_amount": ""
        }
        """

    # ============================================================
    # CPT CLEANING
    # ============================================================
    def extract_cpt_code(self, text):
        """Extract clean CPT/HCPCS code from noisy string"""
        text = str(text).strip()

        patterns = [
            r"\b\d{5}\b",        # Category I
            r"\b\d{4}F\b",       # Category II
            r"\b\d{4}T\b",       # Category III
            r"\b[A-Z]\d{4}\b"    # HCPCS
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()

        return None

    # ============================================================
    # CPT VALIDATION
    # ============================================================
    def is_valid_cpt(self, code):
        """Return True/False (JSON safe)"""

        if not code:
            return False

        return bool(
            re.match(r"^\d{5}$", code) or
            re.match(r"^\d{4}F$", code) or
            re.match(r"^\d{4}T$", code) or
            re.match(r"^[A-Z]\d{4}$", code)
        )

    # ============================================================
    # AMOUNT VALIDATION
    # ============================================================
    def is_valid_amount(self, value):
        """Check if value is numeric"""
        try:
            float(str(value).replace(",", "").strip())
            return True
        except:
            return False

    # ============================================================
    # MODEL INFERENCE
    # ============================================================
    def extract_with_logprobs(self, image):

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt}
            ]
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )

        generated_ids = output.sequences[0]
        input_len = inputs["input_ids"].shape[-1]

        result_text = self.processor.decode(
            generated_ids[input_len:],
            skip_special_tokens=True
        )

        # TOKEN PROBS
        scores = output.scores
        tokens = generated_ids[input_len:]
        decoded_tokens = [self.processor.decode([t]) for t in tokens]

        token_data = []

        for i, score in enumerate(scores):
            log_probs = F.log_softmax(score, dim=-1)
            token_id = tokens[i]
            logprob = log_probs[0, token_id].item()

            token_data.append({
                "token": decoded_tokens[i],
                "prob": math.exp(logprob)
            })

        return result_text, token_data

    # ============================================================
    # CONFIDENCE
    # ============================================================
    def compute_field_confidence(self, value, token_data):
        """
        Calculates confidence score for a specific extracted value by finding its
        corresponding tokens and averaging their probabilities.
        """
        if not value:
            return 0.0

        # Standardize for comparison
        value_str = str(value).strip().lower()
        if not value_str:
            return 0.0

        best_score = 0.0
        
        # Sliding window search through the generated tokens
        for start in range(len(token_data)):
            reconstructed_raw = ""
            for end in range(start, min(start + 40, len(token_data))):
                reconstructed_raw += token_data[end]["token"]
                
                # Check if the reconstructed sequence contains our target value
                # Using lower() to handle casing differences and strip() for whitespace
                cleaned = reconstructed_raw.strip().lower()

                if value_str in cleaned or cleaned in value_str:
                    # Found a potential match, calculate geometric mean of probabilities
                    probs = [
                        token_data[k]["prob"]
                        for k in range(start, end + 1)
                        if token_data[k]["token"].strip() # Ignore pure whitespace tokens
                    ]

                    if not probs:
                        continue

                    # Geometric mean calculation
                    log_sum = sum(math.log(p + 1e-12) for p in probs)
                    geo_mean = math.exp(log_sum / len(probs))

                    # Penalty factor for length mismatch to avoid partial matches being overconfident
                    len_ratio = min(len(cleaned), len(value_str)) / max(len(cleaned), len(value_str))
                    score = geo_mean * len_ratio

                    best_score = max(best_score, score)
                    
                    # If we found an exact match, we can stop the inner loop early
                    if cleaned == value_str:
                        break

        # Sanity cap for very short values which are prone to false positives
        if len(value_str) <= 2:
            best_score = min(best_score, 0.75)

        return round(float(best_score), 4)

    # ============================================================
    # MAIN PIPELINE
    # ============================================================
    def extract_data(self, file_path):

        images = load_images(file_path)
        final_output = {}

        for i, image in enumerate(images):
            print(f"Processing page {i+1}")

            result_text, token_data = self.extract_with_logprobs(image)

            cleaned = result_text.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(cleaned)
            except:
                final_output[f"page_{i+1}"] = {"raw_output": cleaned}
                continue

            structured_data = {}

            for key, value in data.items():

                # LIST FIELDS
                if isinstance(value, list):

                    field_list = []

                    for v in value:

                        # CPT FIX
                        if key == "cpt_codes":
                            extracted = self.extract_cpt_code(v)

                            if extracted:
                                conf = self.compute_field_confidence(extracted, token_data)

                                item = {
                                    "value": extracted,
                                    "raw_text": v,
                                    "confidence": conf,
                                    "valid": self.is_valid_cpt(extracted),
                                    "review_required": conf < 0.80
                                }
                            else:
                                item = {
                                    "value": None,
                                    "raw_text": v,
                                    "confidence": 0.0,
                                    "valid": False,
                                    "review_required": True
                                }

                            field_list.append(item)
                            continue

                        # CHARGES FIX
                        if key == "charges":
                            if not self.is_valid_amount(v):
                                item = {
                                    "value": v,
                                    "confidence": 0.0,
                                    "valid": False,
                                    "review_required": True
                                }
                            else:
                                conf = self.compute_field_confidence(v, token_data)
                                item = {
                                    "value": v,
                                    "confidence": conf,
                                    "valid": True,
                                    "review_required": conf < 0.80
                                }

                            field_list.append(item)
                            continue

                        # DEFAULT LIST
                        conf = self.compute_field_confidence(v, token_data)

                        field_list.append({
                            "value": v,
                            "confidence": conf,
                            "review_required": conf < 0.80
                        })

                    structured_data[key] = field_list

                # SINGLE FIELD
                else:
                    conf = self.compute_field_confidence(value, token_data)

                    structured_data[key] = {
                        "value": value,
                        "confidence": conf,
                        "review_required": conf < 0.80
                    }

            final_output[f"page_{i+1}"] = structured_data

        return final_output


if __name__ == "__main__":
    extractor = QwenExtractor()
    result = extractor.extract_data("/home/rohit.sahu/Qwen_model/samples_nonstandard_data/Document_1.pdf")
    print(json.dumps(result, indent=2))