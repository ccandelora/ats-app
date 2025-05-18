# File Sanitization in ATS Resume Checker

## Overview

The ATS Resume Checker implements comprehensive file sanitization to ensure security when handling uploaded resume files. This document describes the file sanitization process and features implemented to protect against security threats.

## Key Security Features

### 1. File Extension Validation
- Validates that uploaded files have allowed extensions (.pdf, .docx, .doc, .txt, .rtf)
- Prevents upload of potentially dangerous file types like executables

### 2. MIME Type Verification
- Uses `python-magic` to check the actual file content against its claimed extension
- Prevents attacks where a malicious file is disguised with a safe extension
- Enforces allowed MIME types for each extension:
  - PDF: application/pdf
  - DOCX: application/vnd.openxmlformats-officedocument.wordprocessingml.document
  - DOC: application/msword
  - TXT: text/plain
  - RTF: text/rtf, application/rtf

### 3. File Size Validation
- Enforces minimum and maximum file size limits
- Prevents empty or suspiciously small files (under 100 bytes)
- Limits maximum file size based on file type:
  - PDF/DOCX/DOC: 16MB
  - TXT/RTF: 5MB

### 4. Content Scanning
- Scans file contents for malicious patterns:
  - JavaScript injections, script tags
  - Potential command injections
  - Macros in Office documents
  - Suspicious URL patterns
  - References to system files and paths

### 5. Magic Bytes/Signature Validation
- Checks file headers (first 16 bytes) to validate file type
- Blocks files with executable signatures (MZ, ELF)
- Detects shell scripts and other potentially dangerous file types

### 6. Office Document Security
- For DOCX files, examines the ZIP structure for:
  - VBA macros (blocked)
  - External references in document.xml.rels (blocked)

### 7. Secure File Handling
- Uses temporary files during validation to avoid filesystem exposure
- Implements secure file naming with randomization to prevent path traversal
- Securely cleans up temporary files even when errors occur

### 8. Comprehensive Logging
- Logs all file operations and security checks
- Records reasons for file rejections
- Maintains hash records of processed files for auditing

## Implementation Details

### Sanitization Process Flow

1. File upload received
2. Temporary file created
3. Extension validation
4. MIME type verification
5. Size validation
6. Content scanning
7. If all checks pass, generate secure filename
8. Move to final location
9. Clean up temporary files

### Error Handling

- Detailed error messages for administrators
- Generic security messages for users
- Proper cleanup of temporary files in case of errors
- Comprehensive logging of all rejected files

## Security Considerations

- All file validation is performed on the server side, never trusting client-side validation
- Defense-in-depth approach with multiple validation layers
- Content scanning with pattern matching to detect common attack vectors
- No execution of uploaded content at any point

## Testing

The file sanitization functionality includes automated testing to verify all security features:
- Extension validation tests
- Size validation tests
- Malicious content detection tests
- End-to-end sanitization workflow tests
- Filename sanitization tests 