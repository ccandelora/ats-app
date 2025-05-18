import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import analyzer module
from models.analyzer import ResumeAnalyzer

class TestResumeAnalyzer(unittest.TestCase):
    """Test cases for ResumeAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ResumeAnalyzer()
        
        # Sample resume and job description for testing
        self.sample_resume = """
John Doe
john.doe@example.com
(123) 456-7890

SUMMARY
Experienced software engineer with expertise in Python, JavaScript, and cloud technologies.

EXPERIENCE
Senior Software Engineer, ABC Tech Inc.
• Developed scalable web applications using Python and Flask
• Implemented RESTful APIs for mobile applications
• Optimized database queries resulting in 40% performance improvement

Software Developer, XYZ Solutions
• Created responsive web interfaces using React
• Maintained CI/CD pipelines with Jenkins
• Collaborated with UX team to improve user experience

EDUCATION
Bachelor of Science in Computer Science, University of Technology

SKILLS
Python, JavaScript, React, Flask, SQL, Git, Jenkins, AWS, Docker
"""

        self.sample_job_description = """
Senior Software Engineer

Requirements:
- 5+ years of experience in software development
- Strong knowledge of Python and web frameworks (Flask or Django)
- Experience with front-end technologies (JavaScript, React)
- Database design and optimization experience
- CI/CD and DevOps experience
- Cloud infrastructure experience (AWS or Azure)
- Good communication and teamwork skills

Responsibilities:
- Develop and maintain web applications
- Optimize application performance
- Collaborate with cross-functional teams
- Mentor junior developers
- Participate in code reviews
"""

    def test_extract_keywords(self):
        """Test keyword extraction functionality."""
        # Test keywords are extracted from text
        keywords = self.analyzer.extract_keywords(self.sample_job_description, 10)
        
        # Check that we get expected keywords
        self.assertEqual(len(keywords), 10)
        
        # Keywords should be tuples of (word, score)
        self.assertEqual(len(keywords[0]), 2)
        
        # Check some expected keywords are in the results
        extracted_words = [word for word, _ in keywords]
        expected_keywords = ['python', 'experience', 'developers']
        
        for keyword in expected_keywords:
            found = any(keyword in word for word in extracted_words)
            self.assertTrue(found, f"Expected keyword '{keyword}' not found in extracted keywords")

    def test_analyze_resume(self):
        """Test complete resume analysis."""
        # Run analysis with sample data
        results = self.analyzer.analyze_resume(self.sample_resume, self.sample_job_description)
        
        # Check that the analysis returns expected structure
        self.assertIn('match_results', results)
        self.assertIn('contact_info', results)
        self.assertIn('sections', results)
        
        # Check specific result components
        match_results = results['match_results']
        self.assertIn('combined_score', match_results)
        self.assertIn('matched_keywords', match_results)
        self.assertIn('missing_keywords', match_results)
        
        # Score should be between 0 and 100
        self.assertTrue(0 <= match_results['combined_score'] <= 100)
        
        # Contact info should be extracted
        contact_info = results['contact_info']
        self.assertEqual(contact_info['name'], 'John Doe')
        self.assertEqual(contact_info['email'], 'john.doe@example.com')
        self.assertEqual(contact_info['phone'], '(123) 456-7890')

    def test_identify_resume_sections(self):
        """Test resume section identification."""
        sections = self.analyzer.identify_resume_sections(self.sample_resume)
        
        # Check that common sections are identified
        expected_sections = ['summary', 'experience', 'education', 'skills']
        for section in expected_sections:
            self.assertIn(section, sections)
            self.assertIn('content', sections[section])
            self.assertNotEqual(sections[section]['content'], '')

    def test_check_action_verbs(self):
        """Test action verb detection."""
        result = self.analyzer.check_action_verbs(self.sample_resume)
        
        self.assertIn('has_action_verbs', result)
        self.assertIn('count', result)
        self.assertIn('found', result)
        
        # Sample resume has action verbs
        self.assertTrue(result['has_action_verbs'])
        self.assertGreater(result['count'], 0)
        self.assertGreater(len(result['found']), 0)

    def test_check_quantifiable_results(self):
        """Test quantifiable results detection."""
        result = self.analyzer.check_quantifiable_results(self.sample_resume)
        
        self.assertIn('has_quantifiable_results', result)
        self.assertIn('ratio', result)
        
        # Sample resume has a quantifiable result (40% improvement)
        self.assertTrue(result['has_quantifiable_results'])
        self.assertGreater(result['ratio'], 0)

    @patch('models.analyzer.calculate_semantic_similarity')
    def test_calculate_match_score(self, mock_similarity):
        """Test match score calculation with mocked similarity function."""
        # Mock the semantic similarity function to return predictable values
        mock_similarity.return_value = 0.8
        
        result = self.analyzer.calculate_match_score(self.sample_resume, self.sample_job_description, True)
        
        self.assertIn('combined_score', result)
        self.assertIn('exact_match_score', result)
        self.assertIn('semantic_match_score', result)
        self.assertIn('matched_keywords', result)
        self.assertIn('missing_keywords', result)
        
        # With our mocked similarity always returning 0.8, score should be high
        self.assertGreater(result['combined_score'], 50)

    def test_edge_cases(self):
        """Test edge cases like empty inputs."""
        # Test with empty resume
        empty_resume_result = self.analyzer.analyze_resume("", self.sample_job_description)
        self.assertIn('match_results', empty_resume_result)
        self.assertEqual(empty_resume_result['match_results']['combined_score'], 0)
        
        # Test with empty job description
        empty_job_result = self.analyzer.analyze_resume(self.sample_resume, "")
        self.assertIn('match_results', empty_job_result)
        # Score should be 0 or very low with no job keywords to match against
        self.assertLess(empty_job_result['match_results']['combined_score'], 10)

if __name__ == '__main__':
    unittest.main() 