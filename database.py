import os
import sqlite3
import json
from datetime import datetime, timedelta
import logging
import uuid
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/database.log'
)
logger = logging.getLogger('database')

# Database configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'resume_data.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Ensure temp storage directory exists
TEMP_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_storage')
os.makedirs(TEMP_STORAGE_DIR, exist_ok=True)

def get_db_connection():
    """Create a database connection to SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        return None

def init_db():
    """Initialize the database schema if it doesn't exist."""
    try:
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to connect to database for initialization")
            return False
        
        logger.info("Initializing database schema...")
        
        # Create resume_data table for storing resume analysis data
        conn.execute('''
        CREATE TABLE IF NOT EXISTS resume_data (
            id TEXT PRIMARY KEY,
            resume_text TEXT NOT NULL,
            job_description TEXT,
            industry TEXT DEFAULT 'general',
            optimized_resume TEXT,
            analysis_results TEXT,
            original_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            last_accessed TIMESTAMP
        )
        ''')
        
        # Create a table for contact info
        conn.execute('''
        CREATE TABLE IF NOT EXISTS contact_info (
            resume_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            linkedin TEXT,
            FOREIGN KEY (resume_id) REFERENCES resume_data (id) ON DELETE CASCADE
        )
        ''')
        
        # Create index on expiry date
        conn.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON resume_data (expires_at)')
        
        conn.commit()
        conn.close()
        
        logger.info("Database schema initialized successfully")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        return False

def save_resume_data(resume_text, job_description, original_filename, industry='general'):
    """Save resume data to database and return a reference ID."""
    try:
        # Generate a unique ID
        data_id = str(uuid.uuid4())
        
        # Create database connection
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to connect to database for saving resume data")
            return None
        
        # Set expiry time (24 hours from now)
        expires_at = datetime.now() + timedelta(hours=24)
        
        # Insert resume data
        conn.execute(
            '''
            INSERT INTO resume_data 
            (id, resume_text, job_description, original_filename, industry, expires_at, last_accessed) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (data_id, resume_text, job_description, original_filename, industry, expires_at, datetime.now())
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved resume data with ID: {data_id}")
        return data_id
        
    except sqlite3.Error as e:
        logger.error(f"Error saving resume data: {e}")
        
        # Fallback to file-based storage
        logger.info("Falling back to file-based storage")
        return save_resume_data_to_file(resume_text, job_description, original_filename, industry)

def save_resume_data_to_file(resume_text, job_description, original_filename, industry='general'):
    """Legacy fallback method: Save resume data to a temporary file and return a reference ID."""
    try:
        # Generate a unique ID
        data_id = str(uuid.uuid4())
        
        # Create data object
        data = {
            'resume_text': resume_text,
            'job_description': job_description,
            'original_filename': original_filename,
            'industry': industry,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
        # Save to file
        file_path = os.path.join(TEMP_STORAGE_DIR, f"{data_id}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved resume data to file with ID: {data_id}")
        return data_id
    except Exception as e:
        logger.error(f"Error in file-based storage fallback: {e}")
        return None

def load_resume_data(data_id):
    """Load resume data from database."""
    try:
        # Create database connection
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to connect to database for loading resume data")
            return None
        
        # Query resume data
        cursor = conn.cursor()
        cursor.execute(
            '''
            SELECT * FROM resume_data WHERE id = ?
            ''',
            (data_id,)
        )
        
        row = cursor.fetchone()
        
        # Update last accessed timestamp
        if row:
            conn.execute(
                '''
                UPDATE resume_data SET last_accessed = ? WHERE id = ?
                ''',
                (datetime.now(), data_id)
            )
            conn.commit()
        
        conn.close()
        
        if row:
            # Convert Row to dict
            data = dict(row)
            
            # Parse analysis_results JSON if it exists
            if data.get('analysis_results'):
                try:
                    data['analysis_results'] = json.loads(data['analysis_results'])
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON
            
            logger.info(f"Loaded resume data with ID: {data_id}")
            return data
        else:
            # Try file-based storage as fallback
            return load_resume_data_from_file(data_id)
            
    except sqlite3.Error as e:
        logger.error(f"Error loading resume data: {e}")
        # Try file-based storage as fallback
        return load_resume_data_from_file(data_id)

def load_resume_data_from_file(data_id):
    """Legacy fallback method: Load resume data from a temporary file."""
    file_path = os.path.join(TEMP_STORAGE_DIR, f"{data_id}.json")
    if not os.path.exists(file_path):
        logger.error(f"No resume data file found for ID: {data_id}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded resume data from file with ID: {data_id}")
        return data
    except Exception as e:
        logger.error(f"Error loading resume data from file: {e}")
        return None

def save_analysis_results(data_id, analysis_results):
    """Save analysis results to database."""
    try:
        # Create database connection
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to connect to database for saving analysis results")
            return False
        
        # Convert analysis_results to JSON string
        analysis_json = json.dumps(analysis_results)
        
        # Update resume data
        conn.execute(
            '''
            UPDATE resume_data SET analysis_results = ? WHERE id = ?
            ''',
            (analysis_json, data_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved analysis results for resume ID: {data_id}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Error saving analysis results: {e}")
        
        # Fallback to file-based update
        return update_resume_data_file(data_id, {'analysis_results': analysis_results})

def save_optimized_resume(data_id, optimized_resume):
    """Save optimized resume to database."""
    try:
        # Create database connection
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to connect to database for saving optimized resume")
            return False
        
        # Update resume data
        conn.execute(
            '''
            UPDATE resume_data SET optimized_resume = ? WHERE id = ?
            ''',
            (optimized_resume, data_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved optimized resume for ID: {data_id}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Error saving optimized resume: {e}")
        
        # Fallback to file-based update
        return update_resume_data_file(data_id, {'optimized_resume': optimized_resume})

def save_contact_info(data_id, contact_info):
    """Save contact info to database."""
    try:
        # Create database connection
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to connect to database for saving contact info")
            return False
        
        # Check if contact info already exists
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM contact_info WHERE resume_id = ?', (data_id,))
        exists = cursor.fetchone() is not None
        
        if exists:
            # Update existing record
            conn.execute(
                '''
                UPDATE contact_info 
                SET name = ?, email = ?, phone = ?, linkedin = ?
                WHERE resume_id = ?
                ''',
                (
                    contact_info.get('name'), 
                    contact_info.get('email'), 
                    contact_info.get('phone'), 
                    contact_info.get('linkedin'),
                    data_id
                )
            )
        else:
            # Insert new record
            conn.execute(
                '''
                INSERT INTO contact_info 
                (resume_id, name, email, phone, linkedin) 
                VALUES (?, ?, ?, ?, ?)
                ''',
                (
                    data_id,
                    contact_info.get('name'), 
                    contact_info.get('email'), 
                    contact_info.get('phone'), 
                    contact_info.get('linkedin')
                )
            )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved contact info for resume ID: {data_id}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Error saving contact info: {e}")
        
        # Fallback to file-based update
        return update_resume_data_file(data_id, {'contact_info': contact_info})

def save_industry_info(data_id, industry):
    """Save industry selection to database."""
    try:
        # Create database connection
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to connect to database for saving industry info")
            return False
        
        # Update resume data
        conn.execute(
            '''
            UPDATE resume_data SET industry = ? WHERE id = ?
            ''',
            (industry, data_id)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved industry info for resume ID: {data_id}")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Error saving industry info: {e}")
        
        # Fallback to file-based update
        return update_resume_data_file(data_id, {'industry': industry})

def update_resume_data_file(data_id, updates):
    """Legacy fallback method: Update resume data file with new information."""
    file_path = os.path.join(TEMP_STORAGE_DIR, f"{data_id}.json")
    if not os.path.exists(file_path):
        logger.error(f"No resume data file found for ID: {data_id}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Update data with new values
        data.update(updates)
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Updated resume data file for ID: {data_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating resume data file: {e}")
        return False

def cleanup_expired_data():
    """Clean up expired resume data from database and temp storage."""
    now = datetime.now()
    
    # Clean up database
    try:
        conn = get_db_connection()
        if conn is not None:
            # Get IDs of expired data for logging
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM resume_data WHERE expires_at < ?', (now,))
            expired_ids = [row['id'] for row in cursor.fetchall()]
            
            if expired_ids:
                logger.info(f"Cleaning up {len(expired_ids)} expired entries: {', '.join(expired_ids)}")
                
                # Delete expired records
                conn.execute('DELETE FROM resume_data WHERE expires_at < ?', (now,))
                conn.commit()
            
            conn.close()
    except sqlite3.Error as e:
        logger.error(f"Error cleaning up database: {e}")
    
    # Clean up file storage
    try:
        for filename in os.listdir(TEMP_STORAGE_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(TEMP_STORAGE_DIR, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if 'expires_at' in data:
                        expires_at = datetime.fromisoformat(data['expires_at'])
                        if expires_at < now:
                            os.remove(file_path)
                            logger.info(f"Removed expired file: {filename}")
                except (json.JSONDecodeError, ValueError, OSError) as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    
                    # If file is older than 48 hours, remove it anyway
                    file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if now - file_mod_time > timedelta(hours=48):
                        os.remove(file_path)
                        logger.info(f"Removed old file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning up file storage: {e}")

# Initialize the database when module is imported
init_db() 