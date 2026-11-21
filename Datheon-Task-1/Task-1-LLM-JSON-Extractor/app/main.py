"""Main entry point and CLI for the Intelligent Document Information Extractor."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from app.invoice_models import Invoice
from app.llm import GeminiExtractor
from app.prompts import get_invoice_prompt, get_resume_prompt
from app.resume_models import Resume
from app.utils import parse_and_validate, read_document_file, save_json, setup_logger

logger = setup_logger("app.main")


def extract_invoice(
    text: str,
    extractor: Optional[GeminiExtractor] = None,
    model_name: Optional[str] = None
) -> Invoice:
    """
    Reusable pipeline function: Extract structured invoice data from unstructured text.

    Args:
        text: Unstructured invoice document text.
        extractor: Optional existing GeminiExtractor instance.
        model_name: Optional model override.

    Returns:
        Validated Invoice Pydantic model instance.
    """
    if extractor is None:
        extractor = GeminiExtractor(model_name=model_name)

    logger.info("Formatting invoice prompt with parameterized templates and few-shot examples...")
    prompt = get_invoice_prompt(text)

    logger.info("Invoking Gemini model for invoice extraction...")
    raw_response = extractor.generate_extraction(prompt)

    logger.info("Parsing and validating invoice against Pydantic schema...")
    validated_invoice = parse_and_validate(raw_response, Invoice)
    return validated_invoice


def extract_resume(
    text: str,
    extractor: Optional[GeminiExtractor] = None,
    model_name: Optional[str] = None
) -> Resume:
    """
    Reusable pipeline function: Extract structured resume data from unstructured text.

    Args:
        text: Unstructured resume document text.
        extractor: Optional existing GeminiExtractor instance.
        model_name: Optional model override.

    Returns:
        Validated Resume Pydantic model instance.
    """
    if extractor is None:
        extractor = GeminiExtractor(model_name=model_name)

    logger.info("Formatting resume prompt with parameterized templates and few-shot examples...")
    prompt = get_resume_prompt(text)

    logger.info("Invoking Gemini model for resume extraction...")
    raw_response = extractor.generate_extraction(prompt)

    logger.info("Parsing and validating resume against Pydantic schema...")
    validated_resume = parse_and_validate(raw_response, Resume)
    return validated_resume


def extract_document(
    doc_type: str,
    text: str,
    extractor: Optional[GeminiExtractor] = None,
    model_name: Optional[str] = None
) -> Union[Invoice, Resume]:
    """
    General entrypoint for extracting any supported document type.

    Args:
        doc_type: "invoice" or "resume"
        text: Raw document text
        extractor: Optional existing GeminiExtractor instance
        model_name: Optional model override

    Returns:
        Validated Pydantic model
    """
    normalized_type = doc_type.strip().lower()
    if normalized_type == "invoice":
        return extract_invoice(text, extractor=extractor, model_name=model_name)
    elif normalized_type == "resume":
        return extract_resume(text, extractor=extractor, model_name=model_name)
    else:
        raise ValueError(f"Unsupported document type: '{doc_type}'. Supported types: 'invoice', 'resume'")


def parse_arguments(args=None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Extract structured JSON from unstructured text documents using Google Gemini LLM."
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["invoice", "resume"],
        help="Type of document to extract: 'invoice' or 'resume'."
    )
    parser.add_argument(
        "--file",
        required=True,
        type=str,
        help="Path to the unstructured text document to process."
    )
    parser.add_argument(
        "--output",
        required=False,
        type=str,
        default=None,
        help="Optional custom output path for the extracted JSON. Defaults to output/<type>_<filename>.json"
    )
    parser.add_argument(
        "--model",
        required=False,
        type=str,
        default=None,
        help="Optional Gemini model override (e.g. gemini-2.5-flash, gemini-1.5-flash)."
    )
    parser.add_argument(
        "--api-key",
        required=False,
        type=str,
        default=None,
        help="Optional Gemini API key (defaults to GEMINI_API_KEY environment variable)."
    )
    return parser.parse_args(args)


def main():
    """CLI execution workflow."""
    args = parse_arguments()

    try:
        input_path = Path(args.file)
        logger.info("Reading document from %s...", input_path.resolve())
        doc_text = read_document_file(input_path)

        extractor = GeminiExtractor(
            api_key=args.api_key,
            model_name=args.model
        )

        logger.info("Starting %s extraction pipeline...", args.type.upper())
        result_model = extract_document(
            doc_type=args.type,
            text=doc_text,
            extractor=extractor,
            model_name=args.model
        )

        # Determine output file path
        if args.output:
            out_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = input_path.stem
            out_path = Path("output") / f"{args.type}_{stem}_{timestamp}.json"

        saved_path = save_json(result_model, out_path)
        logger.info("Successfully extracted and validated %s data!", args.type.upper())
        logger.info("Saved validated JSON to: %s", saved_path)

        print("\n" + "=" * 60)
        print(f"VALIDATED {args.type.upper()} JSON OUTPUT:")
        print("=" * 60)
        print(result_model.model_dump_json(indent=2))
        print("=" * 60 + "\n")

    except Exception as err:
        logger.error("Extraction pipeline failed: %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
