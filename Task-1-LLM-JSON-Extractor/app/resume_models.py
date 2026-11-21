"""Pydantic models for structured resume information extraction."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class EducationItem(BaseModel):
    """Educational qualification details."""
    model_config = ConfigDict(str_strip_whitespace=True)

    institution: str = Field(description="Name of the university, college, or school")
    degree: Optional[str] = Field(default=None, description="Degree earned (e.g. Bachelor of Science, Master of Science, High School)")
    field_of_study: Optional[str] = Field(default=None, description="Major, concentration, or specialization")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY, MM/YYYY, or as written)")
    end_date: Optional[str] = Field(default=None, description="Graduation/End date or 'Present'")
    gpa: Optional[str] = Field(default=None, description="Grade Point Average or academic percentage if provided")

    @field_validator("institution")
    @classmethod
    def institution_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Institution name cannot be empty")
        return v.strip()


class WorkExperienceItem(BaseModel):
    """Professional employment history item."""
    model_config = ConfigDict(str_strip_whitespace=True)

    company: str = Field(description="Company or employer name")
    position: str = Field(description="Job title or role held")
    start_date: Optional[str] = Field(default=None, description="Start date of employment")
    end_date: Optional[str] = Field(default=None, description="End date or 'Present'")
    location: Optional[str] = Field(default=None, description="Job location (City, State/Country, or Remote)")
    responsibilities: List[str] = Field(default_factory=list, description="List of key responsibilities, bullet points, or achievements")

    @field_validator("company")
    @classmethod
    def company_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Company name cannot be empty")
        return v.strip()

    @field_validator("position")
    @classmethod
    def position_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Position cannot be empty")
        return v.strip()


class ProjectItem(BaseModel):
    """Personal or professional project details."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(description="Title or name of the project")
    description: Optional[str] = Field(default=None, description="Brief summary of the project scope and results")
    technologies: List[str] = Field(default_factory=list, description="Technologies, tools, or frameworks used")
    link: Optional[str] = Field(default=None, description="URL or repository link to the project")

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Project name cannot be empty")
        return v.strip()


class CertificationItem(BaseModel):
    """Professional certifications or licenses."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(description="Name or title of certification")
    issuing_organization: Optional[str] = Field(default=None, description="Issuing body (e.g. AWS, Google Cloud, Coursera)")
    issue_date: Optional[str] = Field(default=None, description="Date issued")
    credential_id: Optional[str] = Field(default=None, description="Credential ID or verification URL")

    @field_validator("name")
    @classmethod
    def cert_name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Certification name cannot be empty")
        return v.strip()


class Resume(BaseModel):
    """Validated structured schema for complete resume extraction."""
    model_config = ConfigDict(str_strip_whitespace=True)

    full_name: str = Field(description="Candidate's full legal or preferred name")
    email: Optional[str] = Field(default=None, description="Primary email address")
    phone_number: Optional[str] = Field(default=None, description="Contact phone number")
    location: Optional[str] = Field(default=None, description="Current city, state, or country")
    professional_summary: Optional[str] = Field(default=None, description="Executive summary, career objective, or bio")
    technical_skills: List[str] = Field(default_factory=list, description="List of technical skills, languages, tools, frameworks")
    education: List[EducationItem] = Field(default_factory=list, description="List of education qualifications")
    work_experience: List[WorkExperienceItem] = Field(default_factory=list, description="List of professional work experiences")
    projects: List[ProjectItem] = Field(default_factory=list, description="List of projects")
    certifications: List[CertificationItem] = Field(default_factory=list, description="List of certifications")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(default=None, description="GitHub profile URL")

    @field_validator("full_name")
    @classmethod
    def full_name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Candidate full name cannot be empty")
        return v.strip()
