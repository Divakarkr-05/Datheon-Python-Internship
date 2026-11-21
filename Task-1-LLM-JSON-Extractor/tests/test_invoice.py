"""Unit tests for Invoice Pydantic schema validation and extraction pipeline."""

import json
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

from app.invoice_models import Invoice, InvoiceItem, VendorDetails, CustomerDetails
from app.main import extract_invoice
from app.llm import GeminiExtractor


def get_valid_invoice_payload():
    return {
        "invoice_number": "INV-2025-001",
        "invoice_date": "2025-01-10",
        "due_date": "2025-02-10",
        "vendor": {
            "name": "Acme Solutions",
            "address": "123 Business Way, Chicago, IL",
            "phone": "+1-312-555-0199",
            "email": "billing@acme.com",
            "tax_id": "US-12345678",
            "website": "https://acme.com"
        },
        "customer": {
            "name": "Global Corp",
            "address": "456 Enterprise Rd, New York, NY",
            "phone": "+1-212-555-0122",
            "email": "ap@globalcorp.com",
            "tax_id": None
        },
        "items": [
            {
                "description": "Cloud Architecture Consulting",
                "quantity": 10.0,
                "unit_price": 150.0,
                "total": 1500.0
            },
            {
                "description": "DevOps Pipeline Setup",
                "quantity": 1.0,
                "unit_price": 2000.0,
                "total": 2000.0
            }
        ],
        "subtotal": 3500.0,
        "tax": 280.0,
        "grand_total": 3780.0,
        "currency": "USD",
        "notes": "Net 30 payment terms"
    }


class TestInvoiceModel:
    """Tests for Pydantic schema validation of Invoices."""

    def test_valid_invoice_schema(self):
        payload = get_valid_invoice_payload()
        invoice = Invoice.model_validate(payload)

        assert invoice.invoice_number == "INV-2025-001"
        assert invoice.vendor.name == "Acme Solutions"
        assert invoice.customer.name == "Global Corp"
        assert len(invoice.items) == 2
        assert invoice.items[0].quantity == 10.0
        assert invoice.items[0].unit_price == 150.0
        assert invoice.items[0].total == 1500.0
        assert invoice.subtotal == 3500.0
        assert invoice.tax == 280.0
        assert invoice.grand_total == 3780.0
        assert invoice.currency == "USD"

    def test_missing_optional_fields(self):
        """Ensure invoice succeeds with nullable/optional fields omitted or None."""
        payload = {
            "invoice_number": "INV-MINIMAL",
            "invoice_date": None,
            "due_date": None,
            "vendor": {
                "name": "Bare Bones Vendor"
            },
            "customer": {
                "name": "Bare Bones Customer"
            },
            "items": [
                {
                    "description": "One service",
                    "quantity": 1.0,
                    "unit_price": 100.0,
                    "total": 100.0
                }
            ],
            "subtotal": None,
            "tax": None,
            "grand_total": 100.0,
            "currency": None,
            "notes": None
        }
        invoice = Invoice.model_validate(payload)
        assert invoice.invoice_number == "INV-MINIMAL"
        assert invoice.vendor.name == "Bare Bones Vendor"
        assert invoice.vendor.email is None
        assert invoice.due_date is None
        assert invoice.tax is None
        # Subtotal auto-computed from items
        assert invoice.subtotal == 100.0

    def test_invalid_negative_numerical_values(self):
        """Reject negative quantity, unit_price, or grand_total."""
        payload = get_valid_invoice_payload()
        payload["items"][0]["quantity"] = -5.0

        with pytest.raises(ValidationError) as exc_info:
            Invoice.model_validate(payload)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_invalid_unit_price_negative(self):
        payload = get_valid_invoice_payload()
        payload["items"][0]["unit_price"] = -10.0

        with pytest.raises(ValidationError) as exc_info:
            Invoice.model_validate(payload)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_invalid_grand_total_negative(self):
        payload = get_valid_invoice_payload()
        payload["grand_total"] = -50.0

        with pytest.raises(ValidationError) as exc_info:
            Invoice.model_validate(payload)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_empty_invoice_number_fails(self):
        payload = get_valid_invoice_payload()
        payload["invoice_number"] = "   "

        with pytest.raises(ValidationError) as exc_info:
            Invoice.model_validate(payload)
        assert "Invoice number cannot be empty" in str(exc_info.value)

    def test_empty_item_description_fails(self):
        payload = get_valid_invoice_payload()
        payload["items"][0]["description"] = ""

        with pytest.raises(ValidationError) as exc_info:
            Invoice.model_validate(payload)
        assert "Item description cannot be empty" in str(exc_info.value)


class TestInvoiceExtractionPipeline:
    """Tests for the complete invoice extraction pipeline with mocked LLM."""

    def test_mocked_llm_invoice_extraction(self):
        payload = get_valid_invoice_payload()
        mock_response_json = f"```json\n{json.dumps(payload)}\n```"

        mock_extractor = MagicMock(spec=GeminiExtractor)
        mock_extractor.generate_extraction.return_value = mock_response_json

        raw_document_text = "Sample invoice text for testing..."
        invoice = extract_invoice(raw_document_text, extractor=mock_extractor)

        mock_extractor.generate_extraction.assert_called_once()
        assert isinstance(invoice, Invoice)
        assert invoice.invoice_number == "INV-2025-001"
        assert invoice.grand_total == 3780.0
        assert len(invoice.items) == 2
