"""Streamlit Web Dashboard for Intelligent Document Information Extractor.

Supports extracting and strictly validating Invoice and Resume documents
into structured JSON schemas using Google Gemini and Pydantic v2.
Engineered for local execution and Streamlit Community Cloud deployment.
"""

import os
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from pydantic import ValidationError

# Load local environment variables if available
load_dotenv()

from app.invoice_models import Invoice
from app.resume_models import Resume
from app.llm import GeminiExtractor
from app.prompts import get_invoice_prompt, get_resume_prompt
from app.utils import parse_and_validate, extract_text_from_pdf, read_document_file

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Extractor | LLM JSON Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# POLISHED MODERN THEME (DARK & LIGHT MODE COMPATIBLE)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Gradient Hero Title */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #60A5FA 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1rem;
        font-weight: 400;
    }

    /* Pill Badges */
    .badge-bar {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.12);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    /* Section Cards */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Skill Tags for Resume */
    .skill-tag {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.82rem;
        font-weight: 600;
        border-radius: 6px;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin: 3px;
    }

    /* Primary Action Button Styling */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border-radius: 8px !important;
        padding: 0.65rem 1.2rem !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# FILE SIZE LIMITS (BOUNDED TO 20MB)
# -----------------------------------------------------------------------------
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS (CLEAN & MINIMALIST)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")

    # Secure silent API Key handling (No key leakage or green badges)
    env_key = ""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            env_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    if not env_key:
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""

    user_api_key = env_key
    if not user_api_key:
        user_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Enter API key if not set in .env or st.secrets"
        )

    if "active_doc_type" not in st.session_state:
        st.session_state.active_doc_type = "Invoice"

    doc_type = st.selectbox(
        "Target Schema",
        options=["Invoice", "Resume"],
        index=0 if st.session_state.active_doc_type == "Invoice" else 1,
        help="Select schema or allow smart auto-detection"
    )
    st.session_state.active_doc_type = doc_type

    selected_model = st.selectbox(
        "LLM Model",
        options=["gemini-2.0-flash", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"],
        index=0,
        help="gemini-2.0-flash provides ultra-fast (1-3 seconds), highly reliable structured extraction."
    )

    st.markdown("---")

    # Clean, collapsible information block (No wall of text)
    with st.expander("ℹ️ About Architecture", expanded=False):
        st.markdown("""
        **Pipeline Overview:**
        - **Engine:** Google Gemini API with fallback cascade
        - **Validation:** Pydantic v2 strict typing & math reconciliation
        - **Ingestion:** Native PDF parsing via `pypdf` (up to 20MB)
        - **Security:** Anti-prompt-injection untrusted delimiters
        - **Test Suite:** 34 passing automated tests

        *Built for Datheon Internship (Task 1).*
        """)


# -----------------------------------------------------------------------------
# MAIN HERO HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="hero-title">📄 Intelligent Document Extractor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Parse complex, unstructured text & PDF documents into strictly validated JSON schemas.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="badge-bar">
    <span class="pill-badge">⚡ Gemini 2.0 Flash</span>
    <span class="pill-badge">🛡️ Pydantic v2</span>
    <span class="pill-badge">📑 Native PDF (Max 20MB)</span>
    <span class="pill-badge">🔒 Math Reconciled</span>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TWO-COLUMN WORKSPACE LAYOUT
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="medium")

examples_dir = Path("examples")
sample_invoice_pdf = examples_dir / "sample_invoice.pdf"
sample_invoice_txt = examples_dir / "sample_invoice.txt"
sample_resume_pdf = examples_dir / "sample_resume.pdf"
sample_resume_txt = examples_dir / "sample_resume.txt"

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# -----------------
# COLUMN 1: INPUT
# -----------------
with col_left:
    st.markdown('<div class="section-header">📥 1. Document Input (PDF / Text)</div>', unsafe_allow_html=True)

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("📄 Sample Invoice (PDF)", use_container_width=True):
            st.session_state.active_doc_type = "Invoice"
            if sample_invoice_pdf.exists():
                st.session_state.input_text = extract_text_from_pdf(sample_invoice_pdf)
            elif sample_invoice_txt.exists():
                st.session_state.input_text = sample_invoice_txt.read_text(encoding="utf-8")
            st.rerun()

    with btn_col2:
        if st.button("👤 Sample Resume (PDF)", use_container_width=True):
            st.session_state.active_doc_type = "Resume"
            if sample_resume_pdf.exists():
                st.session_state.input_text = extract_text_from_pdf(sample_resume_pdf)
            elif sample_resume_txt.exists():
                st.session_state.input_text = sample_resume_txt.read_text(encoding="utf-8")
            st.rerun()

    uploaded_file = st.file_uploader(
        f"Upload Document (PDF only, max {MAX_FILE_SIZE_MB}MB)",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error(f"⚠️ File size ({uploaded_file.size / (1024*1024):.2f} MB) exceeds maximum allowed limit of {MAX_FILE_SIZE_MB} MB.")
        else:
            try:
                pdf_bytes = uploaded_file.read()
                extracted_pdf_text = extract_text_from_pdf(pdf_bytes)
                st.session_state.input_text = extracted_pdf_text
                st.success(f"✅ Extracted text from '{uploaded_file.name}'")
            except Exception as e:
                st.error(f"❌ Error extracting text from PDF: {e}")

    document_content = st.text_area(
        label="Document Content:",
        value=st.session_state.input_text,
        height=320,
        placeholder="Upload a PDF above, load a sample, or paste raw text here..."
    )

    extract_button = st.button("✨ Extract & Validate JSON", type="primary", use_container_width=True)


# -----------------
# COLUMN 2: OUTPUT
# -----------------
with col_right:
    st.markdown('<div class="section-header">📊 2. Validated Schema Output</div>', unsafe_allow_html=True)

    if extract_button:
        content_size_bytes = len(document_content.encode("utf-8"))
        if not user_api_key.strip():
            st.error("⚠️ Please provide a Gemini API Key in the sidebar or via .env.")
        elif not document_content.strip():
            st.warning("⚠️ Please provide document text or upload a PDF.")
        elif content_size_bytes > MAX_FILE_SIZE_BYTES:
            st.error(f"⚠️ Document size ({content_size_bytes / (1024*1024):.2f} MB) exceeds limit of {MAX_FILE_SIZE_MB} MB.")
        else:
            # Smart Auto-Detection
            text_lower = document_content.lower()
            is_resume_like = any(kw in text_lower for kw in ["summary", "education", "experience", "technical skills", "certifications", "gpa"]) and not any(kw in text_lower for kw in ["invoice", "total due", "bill to", "tax id", "amount due", "remit payment"])
            is_invoice_like = any(kw in text_lower for kw in ["invoice", "subtotal", "bill to", "due date", "remit payment", "grand total"]) and not any(kw in text_lower for kw in ["education", "work experience", "professional summary"])

            active_doc_type = st.session_state.active_doc_type
            if active_doc_type == "Invoice" and is_resume_like:
                active_doc_type = "Resume"
                st.session_state.active_doc_type = "Resume"
            elif active_doc_type == "Resume" and is_invoice_like:
                active_doc_type = "Invoice"
                st.session_state.active_doc_type = "Invoice"

            with st.spinner(f"Extracting & validating {active_doc_type} with {selected_model}..."):
                try:
                    extractor = GeminiExtractor(
                        api_key=user_api_key.strip(),
                        model_name=selected_model
                    )

                    if active_doc_type == "Invoice":
                        prompt = get_invoice_prompt(document_content)
                        raw_llm_output = extractor.generate_extraction(prompt)
                        validated_data = parse_and_validate(raw_llm_output, Invoice)
                    else:
                        prompt = get_resume_prompt(document_content)
                        raw_llm_output = extractor.generate_extraction(prompt)
                        validated_data = parse_and_validate(raw_llm_output, Resume)

                    st.session_state["extraction_result"] = {
                        "type": active_doc_type,
                        "data": validated_data,
                        "json_str": validated_data.model_dump_json(indent=2)
                    }

                    # Automatically save to output folder
                    out_dir = Path("output")
                    out_dir.mkdir(exist_ok=True)
                    out_file = out_dir / f"streamlit_{active_doc_type.lower()}_output.json"
                    out_file.write_text(validated_data.model_dump_json(indent=2), encoding="utf-8")

                    st.success(f"✅ Successfully validated against {active_doc_type} Pydantic schema!")

                except ValidationError as val_err:
                    st.error(
                        f"❌ Schema Validation Error: The document does not conform to '{active_doc_type}'. "
                        f"Please ensure target schema matches document type.\n\n"
                        f"Errors: {val_err.error_count()} field issue(s)."
                    )
                except Exception as err:
                    st.error(f"❌ Extraction failed: {err}")

    # Render results if available
    if "extraction_result" in st.session_state:
        res = st.session_state["extraction_result"]
        doc_res = res["data"]
        json_output = res["json_str"]

        tab_json, tab_preview = st.tabs(["📄 Validated JSON", "📊 Structured Breakdown"])

        with tab_json:
            st.download_button(
                label="📥 Download Validated JSON",
                data=json_output,
                file_name=f"extracted_{res['type'].lower()}.json",
                mime="application/json",
                use_container_width=True
            )
            st.code(json_output, language="json")

        with tab_preview:
            if res["type"] == "Invoice":
                inv: Invoice = doc_res
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Invoice #", inv.invoice_number)
                col_m2.metric("Grand Total", f"{inv.currency or '$'} {inv.grand_total:,.2f}")
                col_m3.metric("Line Items", len(inv.items))

                st.markdown(f"**Vendor:** {inv.vendor.name} · `{inv.vendor.email or 'N/A'}`")
                st.markdown(f"**Customer:** {inv.customer.name} · `{inv.customer.email or 'N/A'}`")

                if inv.items:
                    st.markdown("#### Line Items Breakdown")
                    table_md = [
                        "| Description | Qty | Unit Price | Total |",
                        "| :--- | :---: | :---: | :---: |"
                    ]
                    for item in inv.items:
                        table_md.append(f"| {item.description} | {item.quantity} | ${item.unit_price:,.2f} | ${item.total:,.2f} |")
                    st.markdown("\n".join(table_md))

            else:
                cv: Resume = doc_res
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Candidate Name", cv.full_name)
                col_m2.metric("Identified Skills", len(cv.technical_skills))

                st.markdown(f"**Contact:** `{cv.email or 'N/A'}` | `{cv.phone_number or 'N/A'}` | `{cv.location or 'N/A'}`")

                if cv.professional_summary:
                    st.info(cv.professional_summary)

                if cv.technical_skills:
                    st.markdown("#### Technical Skills")
                    tags_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in cv.technical_skills])
                    st.markdown(tags_html, unsafe_allow_html=True)

                if cv.work_experience:
                    st.markdown("#### Work Experience")
                    for exp in cv.work_experience:
                        st.markdown(f"- **{exp.position}** at **{exp.company}** ({exp.start_date or ''} – {exp.end_date or 'Present'})")

    else:
        st.info("💡 Load a sample document on the left or upload a PDF, then click **'Extract & Validate JSON'**.")
