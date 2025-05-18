import unittest
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.security import ContentSecurityPolicy

class TestContentSecurityPolicy(unittest.TestCase):
    """Test suite for Content Security Policy implementation"""

    def setUp(self):
        """Set up test environment"""
        self.csp = ContentSecurityPolicy()

    def test_csp_initialization(self):
        """Test that CSP initializes with correct directives"""
        self.assertIn('default-src', self.csp.directives)
        self.assertIn('script-src', self.csp.directives)
        self.assertIn('style-src', self.csp.directives)
        self.assertIn('img-src', self.csp.directives)
        self.assertIn('connect-src', self.csp.directives)
        self.assertIn('frame-src', self.csp.directives)
        self.assertIn('object-src', self.csp.directives)
        
    def test_nonce_generation(self):
        """Test that nonces are generated correctly"""
        # Store the original nonces
        original_script_nonce = self.csp.script_nonce
        original_style_nonce = self.csp.style_nonce
        
        # Verify initial nonces are valid
        self.assertIsNotNone(original_script_nonce)
        self.assertIsNotNone(original_style_nonce)
        self.assertEqual(len(original_script_nonce), 64)  # sha256 hexdigest is 64 chars
        
        # Generate new nonces
        self.csp.generate_nonces()
        
        # Verify new nonces are different from original ones
        self.assertNotEqual(original_script_nonce, self.csp.script_nonce)
        self.assertNotEqual(original_style_nonce, self.csp.style_nonce)
    
    def test_csp_header_generation(self):
        """Test that CSP header is generated correctly"""
        header = self.csp.get_csp_header()
        
        # Verify the header contains expected directives
        self.assertIn("default-src 'self'", header)
        self.assertIn("script-src 'self'", header)
        self.assertIn("https://cdn.jsdelivr.net/", header)
        self.assertIn("frame-ancestors 'none'", header)
        
        # In development mode (the default in tests), unsafe-inline should be allowed
        self.assertIn("'unsafe-inline'", header)
    
    def test_security_headers(self):
        """Test that all security headers are properly generated"""
        headers = self.csp.get_security_headers()
        
        # Verify required headers are present
        self.assertIn('Content-Security-Policy', headers)
        self.assertIn('X-Content-Type-Options', headers)
        self.assertIn('X-Frame-Options', headers)
        self.assertIn('X-XSS-Protection', headers)
        self.assertIn('Referrer-Policy', headers)
        self.assertIn('Permissions-Policy', headers)
        
        # HSTS should not be enabled in development mode
        self.assertNotIn('Strict-Transport-Security', headers)
        
        # Verify specific header values
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(headers['X-Frame-Options'], 'DENY')
        self.assertEqual(headers['X-XSS-Protection'], '1; mode=block')

if __name__ == '__main__':
    unittest.main() 