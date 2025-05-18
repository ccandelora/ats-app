# Content Security Policy in ATS Resume Checker

## Overview

The ATS Resume Checker implements a comprehensive Content Security Policy (CSP) to protect against Cross-Site Scripting (XSS) attacks, code injection, and other client-side vulnerabilities.

## Key Security Features

### 1. Content Security Policy Implementation

- A robust Content Security Policy that restricts which resources can be loaded and executed
- Nonce-based script and style protection for inline scripts and styles
- Strict source directives to prevent loading resources from unauthorized domains
- Defense-in-depth approach with multiple restrictive policies

### 2. CSP Directives

The application enforces the following CSP directives:

- **default-src**: Restricts to 'self' (same origin)
- **script-src**: Allows scripts only from:
  - Same origin ('self')
  - cdn.jsdelivr.net (for Bootstrap)
  - cdnjs.cloudflare.com (for Font Awesome)
  - Scripts with valid nonces (in production)
- **style-src**: Restricts styles to:
  - Same origin ('self')
  - cdn.jsdelivr.net (for Bootstrap)
  - cdnjs.cloudflare.com (for Font Awesome)
  - Styles with valid nonces (in production)
- **img-src**: Allows images from 'self' and data: URIs
- **font-src**: Restricts fonts to 'self' and CDN sources
- **connect-src**: Restricts AJAX/fetch calls to 'self'
- **frame-src**: Set to 'none' to prevent framing
- **object-src**: Set to 'none' to prevent embedding of plugins
- **base-uri**: Restricts to 'self' to prevent base tag hijacking
- **form-action**: Restricts to 'self' to prevent cross-domain form submissions
- **frame-ancestors**: Set to 'none' to prevent clickjacking
- **upgrade-insecure-requests**: Forces HTTPS for all requests

### 3. Nonce-Based Protection

- Each request generates unique cryptographic nonces for scripts and styles
- Nonces are applied to all inline and external scripts/styles
- Non-nonce scripts and styles are blocked in production
- Prevents attackers from injecting scripts even if they can modify the DOM

### 4. Additional Security Headers

The application also implements these security headers:

- **X-Content-Type-Options**: Set to 'nosniff' to prevent MIME type sniffing
- **X-Frame-Options**: Set to 'DENY' to prevent framing (legacy protection)
- **X-XSS-Protection**: Set to '1; mode=block' for additional XSS protection in older browsers
- **Referrer-Policy**: Set to 'strict-origin-when-cross-origin' to limit referrer information
- **Permissions-Policy**: Restricts browser features (camera, microphone, geolocation, FLoC)
- **Strict-Transport-Security**: Set to 'max-age=31536000; includeSubDomains' to enforce HTTPS (production only)

## Implementation Details

### Architecture

- Content Security Policy is implemented via a dedicated `ContentSecurityPolicy` class
- New nonces are generated for each request
- Flask middleware adds the CSP header to all responses
- Templates use conditional nonce attributes for scripts and styles

### Development vs. Production

- Development mode allows 'unsafe-inline' and 'unsafe-eval' for easier debugging
- Production mode enforces strict nonce-based policy
- Differentiation is automatic based on the FLASK_ENV environment variable

### Testing

CSP implementation includes automated testing to verify:
- Header presence in all responses
- Correct directives in the policy
- Nonce application in templates
- Blocking of unauthorized scripts

## Security Considerations

- CSP is one layer in a defense-in-depth security strategy
- Used alongside CSRF protection, input validation, and file sanitization
- Designed to be restrictive by default, following the principle of least privilege
- Balanced to ensure application functionality while maximizing security 