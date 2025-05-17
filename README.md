# ATS Resume Checker

A personal resume checker app that helps optimize resumes for Applicant Tracking Systems (ATS).

## Features

- Resume parsing (PDF, DOCX)
- Job description analysis
- Keyword matching and scoring
- Skills gap analysis
- Formatting recommendations

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Download spaCy model: `python -m spacy download en_core_web_md`
6. Run the app: `python app.py`

## Project Structure

- `app/` - Web application code
- `models/` - Data processing and analysis models
- `utils/` - Helper functions
- `data/` - Sample data and resources 