import os
import re
import logging
import hashlib
import tempfile
import magic
import io
import zipfile
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/file_sanitizer.log'
)
logger = logging.getLogger('file_sanitizer')

# Define MIME types that are allowed for each extension
ALLOWED_MIME_TYPES = {
    '.pdf': ['application/pdf'],
    '.docx': [
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/zip'  # docx files are zip files with a special structure
    ],
    '.doc': [
        'application/msword'
    ],
    '.txt': [
        'text/plain',
        'text/x-log'
    ],
    '.rtf': [
        'text/rtf',
        'application/rtf'
    ]
}

# File size limits in bytes
FILE_SIZE_LIMITS = {
    '.pdf': 16 * 1024 * 1024,  # 16MB
    '.docx': 16 * 1024 * 1024,  # 16MB
    '.doc': 16 * 1024 * 1024,   # 16MB
    '.txt': 5 * 1024 * 1024,    # 5MB
    '.rtf': 5 * 1024 * 1024     # 5MB
}

# Known malicious patterns
MALICIOUS_PATTERNS = [
    # JavaScript injections
    r'<script.*?>.*?</script>',
    # Potential command injections
    r'&\s*\w+\s*`',
    # Macros in Office Documents
    r'Auto_Open',
    r'AutoOpen',
    r'AutoExec',
    r'Document_Open',
    r'DocumentOpen',
    # Suspicious URL patterns
    r'(https?://.+\.ru/)',
    r'(https?://.+\.cn/)',
    # Suspicious file path references
    r'(/etc/passwd)',
    r'(C:\\Windows\\System32)'
]

# Suspicious file signatures (magic bytes)
SUSPICIOUS_FILE_SIGNATURES = {
    b'MZ': 'Windows executable',
    b'#!/': 'Script file',
    b'\x7fELF': 'Unix executable',
    b'PK\x03\x04': 'ZIP archive (may contain macros)'
}

def get_file_mime_type(file_path):
    """
    Get the MIME type of a file using python-magic
    """
    try:
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
    except Exception as e:
        logger.error(f"Error determining MIME type: {str(e)}")
        return None
    
def validate_file_extension(filename, allowed_extensions):
    """
    Validate that the file has an allowed extension
    """
    if '.' not in filename:
        logger.warning(f"File {filename} has no extension")
        return False, "File has no extension"
    
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in allowed_extensions:
        logger.warning(f"File extension {ext} not in allowed extensions: {allowed_extensions}")
        return False, f"File extension {ext} not allowed. Allowed extensions: {', '.join(allowed_extensions)}"
    
    return True, ""

def validate_file_mime_type(file_path, filename):
    """
    Validate the file's MIME type matches its extension
    """
    ext = os.path.splitext(filename)[1].lower()
    
    # Get the file's actual MIME type
    mime_type = get_file_mime_type(file_path)
    
    if not mime_type:
        return False, "Could not determine file type"
    
    if ext in ALLOWED_MIME_TYPES:
        if mime_type not in ALLOWED_MIME_TYPES[ext]:
            logger.warning(f"File {filename} has MIME type {mime_type}, which doesn't match its extension {ext}")
            return False, f"File type mismatch. File claims to be {ext} but is actually {mime_type}"
    else:
        return False, f"Extension {ext} not supported"
    
    return True, ""

def validate_file_size(file_path, filename):
    """
    Validate the file size is within limits
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in FILE_SIZE_LIMITS:
        return False, f"Extension {ext} not supported"
    
    max_size = FILE_SIZE_LIMITS[ext]
    file_size = os.path.getsize(file_path)
    
    if file_size > max_size:
        logger.warning(f"File {filename} exceeds size limit. Size: {file_size}, Limit: {max_size}")
        return False, f"File exceeds size limit of {max_size / (1024 * 1024):.1f}MB"
    
    if file_size < 100:
        logger.warning(f"File {filename} is suspiciously small: {file_size} bytes")
        return False, "File is too small and might be corrupted"
    
    return True, ""

def check_for_malicious_content(file_path, filename):
    """
    Check for potentially malicious content in the file
    """
    ext = os.path.splitext(filename)[1].lower()
    
    # Check file signature (magic bytes)
    try:
        with open(file_path, 'rb') as f:
            header = f.read(16)  # Read the first 16 bytes
            
            for signature, description in SUSPICIOUS_FILE_SIGNATURES.items():
                if header.startswith(signature):
                    if (ext == '.docx' or ext == '.doc') and signature == b'PK\x03\x04':
                        # .docx files are ZIP archives, so this is normal
                        pass
                    else:
                        logger.warning(f"File {filename} has suspicious signature: {description}")
                        return False, f"File contains suspicious content: {description}"
    except Exception as e:
        logger.error(f"Error checking file signature: {str(e)}")
        return False, "Error checking file content"
    
    # For Office documents, check for macros
    if ext == '.docx':
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Check for VBA macros
                vba_files = [f for f in zip_ref.namelist() if "vbaproject" in f.lower()]
                if vba_files:
                    logger.warning(f"File {filename} contains VBA macros: {vba_files}")
                    return False, "File contains macros which are not allowed"
                
                # Check for external references in document.xml.rels
                for file_info in zip_ref.infolist():
                    if "document.xml.rels" in file_info.filename.lower():
                        content = zip_ref.read(file_info.filename).decode('utf-8', errors='ignore')
                        # Check for external targets
                        if 'Target="http' in content or 'Target="https' in content:
                            logger.warning(f"File {filename} contains external references")
                            return False, "File contains external references which are not allowed"
        except Exception as e:
            logger.error(f"Error checking DOCX file: {str(e)}")
            return False, "Error checking document content"
    
    # For text-based files, check for malicious patterns
    if ext in ['.txt', '.rtf'] or ext == '.pdf':  # PDF can also contain text
        try:
            # For PDFs, extract text first
            if ext == '.pdf':
                # Use the existing parser to extract text
                from parser import extract_text_from_pdf
                content = extract_text_from_pdf(file_path)
            else:
                # For text files, read directly
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
            
            # Check for malicious patterns
            for pattern in MALICIOUS_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    logger.warning(f"File {filename} contains suspicious pattern: {pattern}")
                    return False, "File contains potentially malicious content"
        except Exception as e:
            logger.error(f"Error checking file content: {str(e)}")
            # Don't fail here, as it might be a binary file or encoding issue
    
    return True, ""

def sanitize_file(file_path, original_filename, allowed_extensions):
    """
    Comprehensive file sanitization to ensure security
    
    Args:
        file_path: Path to the temporary file to validate
        original_filename: Original name of the uploaded file
        allowed_extensions: Set of allowed file extensions
        
    Returns:
        tuple: (success, sanitized_file_path, error_message)
            - success: Boolean indicating if sanitization was successful
            - sanitized_file_path: Path to the sanitized file (same as input if no sanitization needed)
            - error_message: Error message if sanitization failed
    """
    logger.info(f"Sanitizing file: {original_filename}")
    
    # Check file existence
    if not os.path.exists(file_path):
        return False, None, "File does not exist"
    
    # Validate file extension
    valid, error = validate_file_extension(original_filename, allowed_extensions)
    if not valid:
        return False, None, error
    
    # Validate file MIME type
    valid, error = validate_file_mime_type(file_path, original_filename)
    if not valid:
        return False, None, error
    
    # Validate file size
    valid, error = validate_file_size(file_path, original_filename)
    if not valid:
        return False, None, error
    
    # Check for malicious content
    valid, error = check_for_malicious_content(file_path, original_filename)
    if not valid:
        return False, None, error
    
    # Calculate file hash for reference
    file_hash = compute_file_hash(file_path)
    logger.info(f"File {original_filename} passed all security checks. Hash: {file_hash}")
    
    return True, file_path, ""

def compute_file_hash(file_path):
    """
    Compute SHA-256 hash of a file
    """
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

def safe_filename(filename):
    """
    Sanitize a filename to make it safe
    """
    # Get file extension
    root, ext = os.path.splitext(filename)
    
    # Clean the root name to remove unsafe characters
    safe_root = re.sub(r'[^\w\-.]', '_', root)
    
    # Combine with the original extension
    return safe_root + ext.lower()

def generate_secure_filename(original_filename):
    """
    Generate a secure filename with a random component
    """
    # Get file extension
    root, ext = os.path.splitext(original_filename)
    
    # Generate a UUID for uniqueness
    unique_id = hashlib.md5(os.urandom(16)).hexdigest()[:12]
    
    # Get a safe version of the original filename (first 10 chars)
    safe_root = re.sub(r'[^\w\-.]', '_', root)[:10]
    
    # Combine them
    return f"{safe_root}_{unique_id}{ext.lower()}" 