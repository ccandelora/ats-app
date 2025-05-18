import logging
from flask import request, current_app, session
from urllib.parse import urlparse
import hashlib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/security.log'
)
logger = logging.getLogger('security')

class ContentSecurityPolicy:
    """
    Class to handle Content Security Policy implementation
    
    Provides methods to create and manage CSP headers for the application.
    """
    
    def __init__(self):
        """Initialize the CSP configuration"""
        # Store nonces for specific resources
        self.script_nonce = None
        self.style_nonce = None
        self.generate_nonces()
        
        # Default source directives for development vs production
        self.is_dev = os.environ.get('FLASK_ENV') == 'development'
        
        # Define the base policy
        self.directives = {
            'default-src': ["'self'"],
            'script-src': [
                "'self'", 
                "https://cdn.jsdelivr.net/",
                "https://cdnjs.cloudflare.com/",
            ],
            'style-src': [
                "'self'", 
                "https://cdn.jsdelivr.net/",
                "https://cdnjs.cloudflare.com/",
            ],
            'img-src': ["'self'", "data:"],
            'font-src': ["'self'", "https://cdn.jsdelivr.net/", "https://cdnjs.cloudflare.com/"],
            'connect-src': ["'self'"],
            'frame-src': ["'none'"],
            'object-src': ["'none'"],
            'base-uri': ["'self'"],
            'form-action': ["'self'"],
            'frame-ancestors': ["'none'"],
            'upgrade-insecure-requests': []
        }
        
        # If in development, add some relaxed rules
        if self.is_dev:
            self.directives['script-src'].append("'unsafe-eval'")  # For development tools
            self.directives['script-src'].append("'unsafe-inline'")  # For development convenience
            self.directives['style-src'].append("'unsafe-inline'")  # For development convenience
        
        # Add nonces to script-src and style-src
        self.add_nonces_to_directives()
        
    def generate_nonces(self):
        """Generate new nonces for this request"""
        # Create cryptographically secure random nonces
        self.script_nonce = hashlib.sha256(os.urandom(32)).hexdigest()
        self.style_nonce = hashlib.sha256(os.urandom(32)).hexdigest()
        
    def add_nonces_to_directives(self):
        """Add the generated nonces to the appropriate directives"""
        if not self.is_dev:  # In production, use nonces
            self.directives['script-src'].append(f"'nonce-{self.script_nonce}'")
            self.directives['style-src'].append(f"'nonce-{self.style_nonce}'")
    
    def get_csp_header(self):
        """
        Generate the Content-Security-Policy header value
        
        Returns:
            str: The properly formatted CSP header value
        """
        header_parts = []
        
        for directive, sources in self.directives.items():
            if sources:  # If there are sources for this directive
                formatted_sources = ' '.join(sources)
                header_parts.append(f"{directive} {formatted_sources}")
            else:  # For directives without sources (like upgrade-insecure-requests)
                header_parts.append(directive)
        
        return '; '.join(header_parts)
    
    def get_security_headers(self):
        """
        Get all security headers to include in the response
        
        Returns:
            dict: Dictionary of security headers
        """
        headers = {
            'Content-Security-Policy': self.get_csp_header(),
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=(), interest-cohort=()'
        }
        
        # Add HSTS in production
        if not self.is_dev:
            headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            
        return headers
    
    def get_nonces(self):
        """
        Get the current nonces
        
        Returns:
            dict: Dictionary containing script_nonce and style_nonce
        """
        return {
            'script_nonce': self.script_nonce,
            'style_nonce': self.style_nonce
        }


# Function to initialize CSP in Flask app
def init_security(app):
    """
    Initialize security features for the Flask app
    
    Args:
        app: Flask application instance
    """
    @app.before_request
    def create_csp():
        """Create a CSP instance for each request"""
        session['csp'] = ContentSecurityPolicy()
    
    @app.after_request
    def add_security_headers(response):
        """Add security headers to the response"""
        if 'csp' in session:
            csp = session['csp']
            headers = csp.get_security_headers()
            
            # Add headers to the response
            for header, value in headers.items():
                response.headers[header] = value
                
        return response 