# ATS Resume Checker

A powerful application that analyzes resumes against job descriptions to improve your chances with Applicant Tracking Systems.

## Features

- **Resume Analysis**: Analyze how well your resume matches a specific job description
- **Keyword Optimization**: Identify missing and matching keywords from job descriptions
- **Industry-Specific Scoring**: Specialized scoring for tech, finance, healthcare, marketing, education, and sales
- **Resume Format Validation**: Check if your resume's format is ATS-friendly
- **Resume Optimization**: Automatically improve your resume based on the job description
- **Visual Heatmap**: See where keywords appear in your resume with an interactive heatmap
- **DOCX Export**: Download optimized resumes in DOCX format
- **Multi-format Support**: Upload resumes in PDF, DOCX, DOC, TXT, or RTF format

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ats-resume-checker.git
   cd ats-resume-checker
   ```

2. Run the installation script:
   ```
   python install.py
   ```
   
   This will install all required dependencies, including NLP models.

3. Start the application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. **Upload Resume**: Upload your resume file (PDF, DOCX, DOC, TXT, or RTF)
2. **Enter Job Description**: Paste the complete job description
3. **Select Industry**: Choose the relevant industry for specialized scoring
4. **Submit**: Get a detailed analysis of your resume
5. **Optimize**: Automatically improve your resume for better ATS compatibility
6. **Download**: Save your optimized resume as a DOCX file

## Technical Details

### Technologies Used

- **Backend**: Flask, Python
- **NLP**: SpaCy, NLTK, TensorFlow, Gensim
- **Document Processing**: PyMuPDF, python-docx, textract
- **Data Storage**: SQLite (via database.py)
- **Frontend**: Bootstrap 5, JavaScript

### Architecture

- **models/**: Core analysis and optimization logic
  - `analyzer.py`: Resume analysis functionality
  - `optimizer.py`: Resume optimization functionality
  - `industry_scorer.py`: Industry-specific scoring
- **utils/**: Utility functions
  - `parser.py`: Document parsing with multiple fallbacks
- **templates/**: HTML templates
- **static/**: CSS, JavaScript, and assets
- **database.py**: Data persistence and management
- **app.py**: Main Flask application
- **tests/**: Automated tests

## Testing

Run the test suite to verify functionality:

```
python run_tests.py
```

For verbose output:

```
python run_tests.py -v
```

## Reliability Features

- **Multi-Parser System**: Fallback parsers when primary parser fails
- **Structured Error Handling**: Comprehensive logging and user feedback
- **Database Storage**: Persistent data storage with SQLite
- **Automated Testing**: Unit tests for core functionality
- **File Validation**: Input validation to prevent processing errors 

## Security Features

- **Input Sanitization**: Proper handling of user inputs
- **Secure File Handling**: Safe file uploads with type validation
- **Temporary Storage**: Automatic data cleanup after 24 hours
- **Session Management**: Secure session handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SpaCy and NLTK for natural language processing capabilities
- Bootstrap for the responsive frontend design
- Flask for the web framework 