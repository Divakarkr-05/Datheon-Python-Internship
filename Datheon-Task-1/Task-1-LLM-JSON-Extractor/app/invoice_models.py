"""Pydantic models for structured invoice information extraction."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class VendorDetails(BaseModel):
    """Details about the vendor/seller issuing the invoice."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(description="Name of the vendor/company issuing the invoice")
    address: Optional[str] = Field(default=None, description="Physical or billing address of the vendor")
    phone: Optional[str] = Field(default=None, description="Contact phone number of the vendor")
    email: Optional[str] = Field(default=None, description="Contact email address of the vendor")
    tax_id: Optional[str] = Field(default=None, description="Vendor tax ID, VAT number, GSTIN, or registration number")
    website: Optional[str] = Field(default=None, description="Vendor website URL")


class CustomerDetails(BaseModel):
    """Details about the client/customer being billed."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(description="Name of the customer or client company")
    address: Optional[str] = Field(default=None, description="Billing or shipping address of the customer")
    phone: Optional[str] = Field(default=None, description="Customer phone number")
    email: Optional[str] = Field(default=None, description="Customer email address")
    tax_id: Optional[str] = Field(default=None, description="Customer tax ID or VAT number if applicable")


class InvoiceItem(BaseModel):
    """Line item within an invoice."""
    model_config = ConfigDict(str_strip_whitespace=True)

    description: str = Field(description="Description or name of the item or service")
    quantity: float = Field(description="Quantity purchased", ge=0)
    unit_price: float = Field(description="Price per single unit", ge=0)
    total: float = Field(description="Total line amount for this item", ge=0)

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Item description cannot be empty")
        return v.strip()


class Invoice(BaseModel):
    """Validated structured schema for complete invoice extraction."""
    model_config = ConfigDict(str_strip_whitespace=True)

    invoice_number: str = Field(description="Unique identifier/number of the invoice")
    invoice_date: Optional[str] = Field(default=None, description="Date the invoice was issued (YYYY-MM-DD or as listed)")
    due_date: Optional[str] = Field(default=None, description="Payment due date")
    vendor: VendorDetails = Field(description="Vendor details")
    customer: CustomerDetails = Field(description="Customer details")
    items: List[InvoiceItem] = Field(default_factory=list, description="List of invoiced items/services")
    subtotal: Optional[float] = Field(default=None, description="Subtotal amount before taxes and discounts", ge=0)
    tax: Optional[float] = Field(default=None, description="Tax or VAT amount", ge=0)
    grand_total: float = Field(description="Final total payable amount", ge=0)
    currency: Optional[str] = Field(default="USD", description="Currency code or symbol (e.g. USD, EUR, INR, $)")
    notes: Optional[str] = Field(default=None, description="Additional notes, payment terms, or bank transfer details")

    @field_validator("invoice_number")
    @classmethod
    def invoice_number_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Invoice number cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_totals(self) -> "Invoice":
        # If subtotal is missing, compute it from line items if available
        if self.subtotal is None and self.items:
            self.subtotal = round(sum(item.total for item in self.items), 2)
        return self
