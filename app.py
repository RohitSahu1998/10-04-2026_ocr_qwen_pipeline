import streamlit as st
import os
import tempfile
import pandas as pd
import json
from PIL import Image

from ocr_engine import PaddleOCREngine, load_images
from qwen_engine import QwenExtractor
from matcher import highlight_and_save_pdf, highlight_single_field

try:
    from pdf2image import convert_from_path
except ImportError:
    pass

# UI Configuration
st.set_page_config(page_title="Document AI Extractor", layout="wide", page_icon="📄")

st.title("📄 Intelligent Document Extraction Pipeline")
st.markdown("Upload a Document (PDF/Image) to instantly extract structured semantic fields, match them precisely to OCR coordinates, and generate a highlighted verification PDF.")

# Cache the AI models so they don't reload every time the user clicks a button!
@st.cache_resource(show_spinner=False)
def load_ai_models():
    with st.spinner("Loading Vision-Language Model and OCR Engines (First run only)..."):
        qwen = QwenExtractor()
        
        # NOTE: If your local machine doesn't have an Nvidia GPU installed, you may need to set use_gpu=False
        try:
            ocr = PaddleOCREngine(use_gpu=True) 
        except Exception:
            ocr = PaddleOCREngine(use_gpu=False)
            
        return qwen, ocr


def load_document_images(file_path):
    """Load document pages using the shared OCR engine resolution (300 DPI)."""
    return load_images(file_path)


uploaded_file = st.file_uploader("Upload an Invoice, Claim, or Form", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    # Save the uploaded file temporarily so the backend engines can read it from a path
    file_bytes = uploaded_file.read()
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
        
    st.success(f"**{uploaded_file.name}** uploaded safely to memory!")
    
    if st.button("🚀 Run AI Extraction Pipeline", use_container_width=True, type="primary"):
        try:
            # 1. Load Models
            qwen_extractor, ocr_engine = load_ai_models()
            
            # 2. Extract Logic
            st.markdown("### Pipeline Execution Steps:")
            
            with st.spinner("🧠 Step 1/3: Running Qwen 2.5 Vision-Language Model..."):
                qwen_data = qwen_extractor.extract_data(temp_path)
                st.success("✅ Step 1: Semantic understanding complete!")
                
            with st.spinner("🔍 Step 2/3: Running PaddleOCR engine across all pages..."):
                ocr_data = ocr_engine.extract_text_with_confidence(temp_path)
                st.success("✅ Step 2: Pixel-level word extraction complete!")
                
            with st.spinner("🔗 Step 3/3: Running Anchor & Spatial Matching to link Qwen with OCR..."):
                output_pdf = temp_path + "_highlighted.pdf"
                output_csv = output_pdf.replace(".pdf", ".csv").replace(".jpg", ".csv")
                
                # Run the matcher which draws the boxes, saves the PDF, and outputs the CSV
                all_matched = highlight_and_save_pdf(temp_path, qwen_data, ocr_data, output_pdf)
                st.success("✅ Step 3: Visual highlighting and alignment CSV generated!")
                
            # --- Store results in session state for interactive highlighting ---
            original_images = load_document_images(temp_path)
            st.session_state['matched_results'] = all_matched if all_matched else []
            st.session_state['original_images'] = original_images
            st.session_state['qwen_data'] = qwen_data
            st.session_state['output_pdf'] = output_pdf
            st.session_state['output_csv'] = output_csv
            st.session_state['uploaded_name'] = uploaded_file.name
            st.session_state['selected_field_idx'] = None  # Reset selection
            st.session_state['pipeline_done'] = True
                
        except Exception as e:
            st.error(f"Pipeline crashed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# =========================================================================
# RESULTS DASHBOARD — Side-by-Side Interactive View
# =========================================================================
if st.session_state.get('pipeline_done'):
    
    matched_results = st.session_state.get('matched_results', [])
    original_images = st.session_state.get('original_images', [])
    qwen_data = st.session_state.get('qwen_data', {})
    output_pdf = st.session_state.get('output_pdf', '')
    output_csv = st.session_state.get('output_csv', '')
    uploaded_name = st.session_state.get('uploaded_name', 'document')

    st.divider()
    st.header("🔍 Interactive Extraction Workspace")
    
    # Side-by-side layout: Table | Document Preview
    col_table, col_preview = st.columns([1.8, 2.2])
    
    with col_table:
        st.subheader("Field Inventory")
        st.markdown("Click **View** in the Action column to highlight the field on the document.")
        
        # --- Custom Table Header ---
        h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns([1.2, 1.3, 0.6, 0.6, 0.5])
        h_col1.markdown("**Field Name**")
        h_col2.markdown("**Extracted Value**")
        h_col3.markdown("**AI Conf.**")
        h_col4.markdown("**OCR Vis.**")
        h_col5.markdown("**Action**")
        st.divider()
        
        # --- Interactive Table Rows ---
        for i, res in enumerate(matched_results):
            r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns([1.2, 1.3, 0.6, 0.6, 0.5])
            
            with r_col1:
                st.write(f"**{res['field']}**")
            
            with r_col2:
                val = res['qwen_value']
                st.write(val if len(val) <= 45 else val[:42] + "...")
            
            with r_col3:
                # AI Prediction Confidence (Logprobs)
                ai_conf = res.get('ai_confidence', 0)
                ai_color = "green" if ai_conf > 0.85 else "orange" if ai_conf > 0.6 else "red"
                st.markdown(f":{ai_color}[{ai_conf:.0%}]")
            
            with r_col4:
                # OCR Visual Readability Confidence
                ocr_conf = res.get('ocr_confidence', 0)
                ocr_color = "green" if ocr_conf > 0.8 else "orange" if ocr_conf > 0.5 else "red"
                st.markdown(f":{ocr_color}[{ocr_conf:.0%}]")
            
            with r_col5:
                # View button triggers session state update
                if st.button("View", key=f"btn_{i}", use_container_width=True):
                    st.session_state['selected_field_idx'] = i
                    st.rerun()

    with col_preview:
        st.subheader("Document Viewer")
        selected_idx = st.session_state.get('selected_field_idx')
        
        if selected_idx is not None and selected_idx < len(matched_results):
            selected = matched_results[selected_idx]
            page_num = selected.get('page', 1)
            
            # Show specific field indicator
            st.info(f"Viewing: **{selected['field']}** (Page {page_num})")
            
            if selected.get('bbox') and page_num <= len(original_images):
                clean_image = original_images[page_num - 1]
                highlighted_img = highlight_single_field(clean_image, selected)
                
                st.image(
                    highlighted_img,
                    caption=f"Verified View: {selected['field']}",
                    use_container_width=True
                )
            else:
                st.warning("No visual match available for this field.")
                # Show plain image if no bbox
                if page_num <= len(original_images):
                    st.image(original_images[page_num-1], use_container_width=True)
        else:
            # Default state: Show first page if nothing selected
            st.info("Select a field from the table to see it highlighted here.")
            if original_images:
                st.image(original_images[0], caption="Page 1 (Normal View)", use_container_width=True)

    # --- Download Buttons ---
    st.divider()
    st.subheader("📥 Export Data")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        if os.path.exists(output_pdf):
            with open(output_pdf, "rb") as f:
                st.download_button(
                    label="Download Verified PDF",
                    data=f,
                    file_name=f"Verified_{uploaded_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                    
    with d_col2:
        if os.path.exists(output_csv):
            with open(output_csv, "rb") as f:
                st.download_button(
                    label="Download Result CSV",
                    data=f,
                    file_name=f"Data_{uploaded_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
