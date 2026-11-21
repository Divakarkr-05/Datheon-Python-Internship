"""Unit tests for Resume Pydantic schema validation and extraction pipeline."""

import json
import pytest
from unittest.mock import MagicMock
from pydantic import ValidationError

from app.resume_models import Resume, EducationItem, WorkExperienceItem, ProjectItem, CertificationItem
from app.main import extract_resume
from app.llm import GeminiExtractor


def get_valid_resume_payload():
    return {
        "full_name": "Jordan Smith",
        "email": "jordan.smith@example.com",
        "phone_number": "+1 (555) 019-2834",
        "location": "Boston, MA",
        "professional_summary": "Full Stack Software Engineer with 5+ years of experience specializing in Python, React, and cloud applications.",
        "technical_skills": ["Python", "FastAPI", "React", "PostgreSQL", "Docker", "AWS"],
        "education": [
            {
                "institution": "Boston University",
                "degree": "Bachelor of Science",
                "field_of_study": "Computer Engineering",
                "start_date": "2015",
                "end_date": "2019",
                "gpa": "3.80"
            }
        ],
        "work_experience": [
            {
                "company": "Tech Innovations Inc.",
                "position": "Software Engineer II",
                "start_date": "June 2019",
                "end_date": "Present",
                "location": "Boston, MA",
                "responsibilities": [
                    "Engineered RESTful microservices using Python and FastAPI.",
                    "Improved CI/CD build speeds by 40% using Docker multi-stage builds."
                ]
            }
        ],
        "projects": [
            {
                "name": "Smart-Extractor",
                "description": "An open source document extraction tool using LLMs.",
                "technologies": ["Python", "Pydantic", "Gemini API"],
                "link": "https://github.com/jordansmith/smart-extractor"
            }
        ],
        "certifications": [
            {
                "name": "AWS Certified Developer",
                "issuing_organization": "Amazon Web Services",
                "issue_date": "2023",
                "credential_id": "AWS-DEV-998811"
            }
        ],
        "linkedin_url": "https://linkedin.com/in/jordansmith",
        "github_url": "https://github.com/jordansmith"
    }


class TestResumeModel:
    """Tests for Pydantic schema validation of Resumes."""

    def test_valid_resume_schema(self):
        payload = get_valid_resume_payload()
        resume = Resume.model_validate(payload)

        assert resume.full_name == "Jordan Smith"
        assert resume.email == "jordan.smith@example.com"
        assert len(resume.technical_skills) == 6
        assert len(resume.education) == 1
        assert resume.education[0].institution == "Boston University"
        assert len(resume.work_experience) == 1
        assert resume.work_experience[0].company == "Tech Innovations Inc."
        assert len(resume.projects) == 1
        assert len(resume.certifications) == 1
        assert resume.linkedin_url == "https://linkedin.com/in/jordansmith"

    def test_missing_optional_fields(self):
        """Verify resume succeeds when optional fields are omitted or None."""
        payload = {
            "full_name": "Minimalist Dev",
            "email": None,
            "phone_number": None,
            "location": None,
            "professional_summary": None,
            "technical_skills": ["Python"],
            "education": [],
            "work_experience": [],
            "projects": [],
            "certifications": [],
            "linkedin_url": None,
            "github_url": None
        }
        resume = Resume.model_validate(payload)
        assert resume.full_name == "Minimalist Dev"
        assert resume.email is None
        assert resume.linkedin_url is None
        assert resume.education == []
        assert resume.technical_skills == ["Python"]

    def test_empty_full_name_fails(self):
        payload = get_valid_resume_payload()
        payload["full_name"] = "  "

        with pytest.raises(ValidationError) as exc_info:
            Resume.model_validate(payload)
        assert "Candidate full name cannot be empty" in str(exc_info.value)

    def test_empty_education_institution_fails(self):
        payload = get_valid_resume_payload()
        payload["education"][0]["institution"] = ""

        with pytest.raises(ValidationError) as exc_info:
            Resume.model_validate(payload)
        assert "Institution name cannot be empty" in str(exc_info.value)

    def test_empty_work_company_fails(self):
        payload = get_valid_resume_payload()
        payload["work_experience"][0]["company"] = ""

        with pytest.raises(ValidationError) as exc_info:
            Resume.model_validate(payload)
        assert "Company name cannot be empty" in str(exc_info.value)

    def test_empty_work_position_fails(self):
        payload = get_valid_resume_payload()
        payload["work_experience"][0]["position"] = ""

        with pytest.raises(ValidationError) as exc_info:
            Resume.model_validate(payload)
        assert "Position cannot be empty" in str(exc_info.value)


class TestResumeExtractionPipeline:
    """Tests for the resume extraction pipeline with mocked LLM."""

    def test_mocked_llm_resume_extraction(self):
        payload = get_valid_resume_payload()
        mock_response_json = json.dumps(payload)

        mock_extractor = MagicMock(spec=GeminiExtractor)
        mock_extractor.generate_extraction.return_value = mock_response_json

        raw_document_text = "Sample resume text for testing..."
        resume = extract_resume(raw_document_text, extractor=mock_extractor)

        mock_extractor.generate_extraction.assert_called_once()
        assert isinstance(resume, Resume)
        assert resume.full_name == "Jordan Smith"
        assert "FastAPI" in resume.technical_skills
        assert resume.education[0].degree == "Bachelor of Science"
