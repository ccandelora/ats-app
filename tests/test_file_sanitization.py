import os
import sys
import unittest
import tempfile
from pathlib import Path

# Add parent directory to path to import application modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from utils.file_sanitizer import (
    validate_file_extension,
    validate_file_size,
    check_for_malicious_content,
    sanitize_file,
    generate_secure_filename,
    safe_filename
)

class TestFileSanitization(unittest.TestCase):
    def setUp(self):
        """Set up test files and directories."""
        self.test_dir = tempfile.mkdtemp()
        self.allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.rtf'}
        
        # Create test files
        self.valid_txt = os.path.join(self.test_dir, 'valid.txt')
        with open(self.valid_txt, 'w') as f:
            f.write('This is a valid text file for testing.')
        
        self.small_file = os.path.join(self.test_dir, 'small.txt')
        with open(self.small_file, 'w') as f:
            f.write('Small')
            
        self.malicious_file = os.path.join(self.test_dir, 'malicious.txt')
        with open(self.malicious_file, 'w') as f:
            f.write('This file contains <script>alert("XSS")</script> script tags.')
            
        # Create a PDF-like file with header
        self.fake_pdf = os.path.join(self.test_dir, 'fake.pdf')
        with open(self.fake_pdf, 'wb') as f:
            f.write(b'%PDF-1.5\n%Something to make it look like a PDF')
            f.write(b'\n' * 100)
            f.write(b'Some content to make it larger than 100 bytes')
    
    def tearDown(self):
        """Clean up test files."""
        for file in [self.valid_txt, self.small_file, self.malicious_file, self.fake_pdf]:
            if os.path.exists(file):
                os.remove(file)
        
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def test_validate_file_extension(self):
        """Test file extension validation."""
        # Valid extension
        valid, _ = validate_file_extension('test.pdf', self.allowed_extensions)
        self.assertTrue(valid)
        
        # Invalid extension
        valid, _ = validate_file_extension('test.exe', self.allowed_extensions)
        self.assertFalse(valid)
        
        # No extension
        valid, _ = validate_file_extension('noextension', self.allowed_extensions)
        self.assertFalse(valid)
    
    def test_validate_file_size(self):
        """Test file size validation."""
        # Valid size
        valid, _ = validate_file_size(self.valid_txt, 'test.txt')
        self.assertTrue(valid)
        
        # Too small
        valid, _ = validate_file_size(self.small_file, 'small.txt')
        self.assertFalse(valid)
    
    def test_check_for_malicious_content(self):
        """Test malicious content detection."""
        # Clean file
        valid, _ = check_for_malicious_content(self.valid_txt, 'valid.txt')
        self.assertTrue(valid)
        
        # Malicious file with script tags
        valid, _ = check_for_malicious_content(self.malicious_file, 'malicious.txt')
        self.assertFalse(valid)
    
    def test_sanitize_file(self):
        """Test the complete sanitization process."""
        # Valid file
        success, path, _ = sanitize_file(self.valid_txt, 'valid.txt', self.allowed_extensions)
        self.assertTrue(success)
        self.assertEqual(path, self.valid_txt)
        
        # Malicious file
        success, path, _ = sanitize_file(self.malicious_file, 'malicious.txt', self.allowed_extensions)
        self.assertFalse(success)
        
        # File with incorrect extension
        success, path, _ = sanitize_file(self.valid_txt, 'valid.exe', self.allowed_extensions)
        self.assertFalse(success)
    
    def test_filename_utils(self):
        """Test filename utility functions."""
        # Test safe_filename
        safe_name = safe_filename('test file & symbols!.pdf')
        self.assertEqual(safe_name, 'test_file___symbols_.pdf')
        
        # Test generate_secure_filename - should contain original name and random part
        secure_name = generate_secure_filename('test.pdf')
        self.assertTrue(secure_name.startswith('test_'))
        self.assertTrue(secure_name.endswith('.pdf'))
        self.assertGreater(len(secure_name), 16)  # Ensure it has the random part

if __name__ == '__main__':
    unittest.main() 