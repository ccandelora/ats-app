import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/csp.log'
)
logger = logging.getLogger('csp')

class ContentSecurityPolicy:
    """
    Content Security Policy (CSP) implementation for the ATS Resume Checker application.
    
    This class helps to define and manage CSP headers to mitigate XSS and other security risks.
    """
    
    def __init__(self, report_only=False, report_uri=None):
        """
        Initialize the CSP configuration.
        
        Args:
            report_only (bool): Whether to use report-only mode (doesn't enforce, just reports violations)
            report_uri (str): URI to report CSP violations to (optional)
        """
        self.report_only = report_only
        self.report_uri = report_uri
        self.directives = {}
        
        # Set default directives
        self._setup_default_directives()
    
    def _setup_default_directives(self):
        """
        Set up default CSP directives for the ATS Resume Checker app
        based on the resources it uses.
        """
        # Default sources (fallback for other resource types)
        self.add_directive('default-src', ["'self'"])
        
        # Script sources - allow our own scripts and Bootstrap
        self.add_directive('script-src', [
            "'self'",  # Our own scripts
            "https://cdn.jsdelivr.net/",  # Bootstrap JS
            "'unsafe-inline'",  # For event handlers (consider removing when possible)
        ])
        
        # Style sources - allow our own styles, Bootstrap, and Font Awesome
        self.add_directive('style-src', [
            "'self'",  # Our own styles
            "https://cdn.jsdelivr.net/",  # Bootstrap CSS
            "https://cdnjs.cloudflare.com/",  # Font Awesome
            "'unsafe-inline'"  # For internal styles (consider removing when possible)
        ])
        
        # Font sources - allow our own fonts and Font Awesome
        self.add_directive('font-src', [
            "'self'",
            "https://cdnjs.cloudflare.com/"  # Font Awesome
        ])
        
        # Image sources - allow our own images
        self.add_directive('img-src', ["'self'", "data:"])
        
        # Connect sources - restrict to our own domain
        self.add_directive('connect-src', ["'self'"])
        
        # Frame sources - restrict to our own domain
        self.add_directive('frame-src', ["'none'"])
        
        # Object sources - disallow object, embed, applet
        self.add_directive('object-src', ["'none'"])
        
        # Block unexpected form submissions
        self.add_directive('form-action', ["'self'"])
        
        # Disallow embedding our site in iframes
        self.add_directive('frame-ancestors', ["'none'"])
        
        # Upgrade insecure requests
        self.add_directive('upgrade-insecure-requests', [])
        
        # Block mixed content
        self.add_directive('block-all-mixed-content', [])
    
    def add_directive(self, directive, values):
        """
        Add or update a CSP directive.
        
        Args:
            directive (str): The CSP directive to add/update
            values (list): List of values for the directive
        """
        self.directives[directive] = values
    
    def get_header_value(self):
        """
        Build the complete CSP header value from the configured directives.
        
        Returns:
            str: The CSP header value
        """
        parts = []
        
        # Add each directive
        for directive, values in self.directives.items():
            if values:
                parts.append(f"{directive} {' '.join(values)}")
            else:
                parts.append(directive)
        
        # Add report-uri if specified
        if self.report_uri:
            parts.append(f"report-uri {self.report_uri}")
        
        return "; ".join(parts)
    
    def get_header_name(self):
        """
        Get the appropriate CSP header name based on mode.
        
        Returns:
            str: The header name to use
        """
        if self.report_only:
            return 'Content-Security-Policy-Report-Only'
        return 'Content-Security-Policy'
    
    def get_header(self):
        """
        Get the complete CSP header (name and value).
        
        Returns:
            tuple: (header_name, header_value)
        """
        return (self.get_header_name(), self.get_header_value())


def get_csp_header(report_only=False, report_uri=None):
    """
    Helper function to get a CSP header for the application.
    
    Args:
        report_only (bool): Whether to use report-only mode
        report_uri (str): URI to report violations to (optional)
    
    Returns:
        tuple: (header_name, header_value)
    """
    csp = ContentSecurityPolicy(report_only=report_only, report_uri=report_uri)
    return csp.get_header()


def apply_csp_headers(response, report_only=False, report_uri=None):
    """
    Apply CSP headers to a Flask response object.
    
    Args:
        response: Flask response object
        report_only (bool): Whether to use report-only mode
        report_uri (str): URI to report violations to (optional)
    
    Returns:
        The modified Flask response object
    """
    header_name, header_value = get_csp_header(report_only, report_uri)
    response.headers[header_name] = header_value
    
    # Add additional security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    return response 