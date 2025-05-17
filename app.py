import os
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
import tempfile
import traceback
import io
import uuid
import json
from datetime import timedelta

# Import parser functions with error handling
try:
    from utils.parser import extract_text_from_file
except ImportError as e:
    print(f"Error importing parser: {e}")
    # Define a fallback function
    def extract_text_from_file(file_path):
        return f"Error: Parser module not available. Please install dependencies via install.py."

# Import analyzer with error handling
try:
    from models.analyzer import ResumeAnalyzer
    from models.optimizer import ResumeOptimizer
except ImportError as e:
    print(f"Error importing analyzer or optimizer: {e}")
    # Define fallback classes
    class ResumeAnalyzer:
        def analyze_resume(self, resume_text, job_description):
            return {
                "error": "Analyzer module not available. Please install dependencies via install.py.",
                "contact_info": {"name": None, "email": None, "phone": None, "linkedin": None},
                "match_results": {"combined_score": 0, "exact_match_score": 0, "semantic_match_score": 0, 
                                  "matched_keywords": [], "missing_keywords": []},
                "format_issues": ["Dependency installation error"],
                "sections": {},
                "action_verbs": {"has_action_verbs": False, "count": 0, "found": [], "missing": []},
                "quantifiable_results": {"has_quantifiable_results": False}
            }
    
    class ResumeOptimizer:
        def optimize_resume(self, resume_text, job_description):
            return resume_text
        
        def generate_docx(self, resume_text):
            return None

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'development-key-for-ats-app')
# Configure session to be permanent and last for 1 hour
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Make sessions permanent by default
@app.before_request
def make_session_permanent():
    session.permanent = True

# Create storage directory for temporary resume data
app.config['STORAGE_DIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_storage')
os.makedirs(app.config['STORAGE_DIR'], exist_ok=True)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_resume_data(resume_text, job_description, original_filename):
    """Save resume data to a temporary file and return a reference ID."""
    # Generate a unique ID
    data_id = str(uuid.uuid4())
    
    # Create data object
    data = {
        'resume_text': resume_text,
        'job_description': job_description,
        'original_filename': original_filename
    }
    
    # Save to file
    file_path = os.path.join(app.config['STORAGE_DIR'], f"{data_id}.json")
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    return data_id

def load_resume_data(data_id):
    """Load resume data from a temporary file."""
    file_path = os.path.join(app.config['STORAGE_DIR'], f"{data_id}.json")
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading resume data: {e}")
        return None

def save_optimized_resume(data_id, optimized_resume):
    """Save optimized resume to the existing data file."""
    file_path = os.path.join(app.config['STORAGE_DIR'], f"{data_id}.json")
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        data['optimized_resume'] = optimized_resume
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        return True
    except Exception as e:
        print(f"Error saving optimized resume: {e}")
        return False

def save_contact_info(data_id, contact_info):
    """Save manual contact info to the existing data file."""
    file_path = os.path.join(app.config['STORAGE_DIR'], f"{data_id}.json")
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Add or update contact info
        data['contact_info'] = contact_info
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        return True
    except Exception as e:
        print(f"Error saving contact info: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    # Check if a file was uploaded
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    
    # If no file was selected
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract text from file
            resume_text = extract_text_from_file(filepath)
            
            # Check if there was an error in text extraction
            if resume_text.startswith("ERROR:"):
                flash(resume_text)
                os.remove(filepath)
                return redirect(request.url)
            
            # Save resume data to file storage instead of session
            data_id = save_resume_data(resume_text, job_description, filename)
            
            # Store only the data ID in the session
            session['resume_data_id'] = data_id
            
            # Analyze resume
            analyzer = ResumeAnalyzer()
            analysis_results = analyzer.analyze_resume(resume_text, job_description)
            
            # Check if there was an error in analysis
            if "error" in analysis_results:
                flash(analysis_results["error"])
                os.remove(filepath)
                return redirect(request.url)
            
            # Clean up file after analysis
            os.remove(filepath)
            
            return render_template('results.html', results=analysis_results)
            
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error analyzing resume: {error_details}")
            flash(f'Error analyzing resume: {str(e)}')
            # Try to remove the file if it exists
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            return redirect(request.url)
    else:
        flash('File type not allowed. Please upload a PDF or DOCX file.')
        return redirect(request.url)

@app.route('/api/analyze', methods=['POST'])
def api_analyze_resume():
    """API endpoint for programmatic resume analysis."""
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Use a temporary file to avoid filesystem clutter
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            file.save(temp.name)
            filepath = temp.name
        
        try:
            # Extract text from file
            resume_text = extract_text_from_file(filepath)
            
            # Check if there was an error in text extraction
            if resume_text.startswith("ERROR:"):
                os.unlink(filepath)
                return jsonify({"error": resume_text}), 500
            
            # Analyze resume
            analyzer = ResumeAnalyzer()
            analysis_results = analyzer.analyze_resume(resume_text, job_description)
            
            # Clean up temporary file
            os.unlink(filepath)
            
            return jsonify(analysis_results)
            
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"API Error: {error_details}")
            # Clean up temporary file in case of error
            if os.path.exists(filepath):
                os.unlink(filepath)
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed. Please upload a PDF or DOCX file."}), 400

@app.route('/optimize', methods=['POST'])
def optimize_resume():
    """Optimize the resume based on job description and display the result."""
    # Check if we have a resume data ID in session
    if 'resume_data_id' not in session:
        flash('No resume to optimize. Please upload a resume first.')
        return redirect(url_for('index'))
    
    try:
        # Get resume data from file storage
        data_id = session['resume_data_id']
        data = load_resume_data(data_id)
        
        if not data:
            flash('Resume data not found. Please upload your resume again.')
            return redirect(url_for('index'))
        
        resume_text = data['resume_text']
        job_description = data['job_description']
        
        # Check if manual contact info was provided
        manual_contact_info = None
        if 'contact_info_data' in request.form:
            try:
                manual_contact_info = json.loads(request.form['contact_info_data'])
                print(f"DEBUG: Received manual contact info: {manual_contact_info}")
            except:
                print("Error parsing manual contact info")
                
        # Optimize the resume
        optimizer = ResumeOptimizer()
        optimized_resume = optimizer.optimize_resume(resume_text, job_description)
        
        # Save the optimized resume to file storage
        save_optimized_resume(data_id, optimized_resume)
        
        # If manual contact info was provided, save it as well
        if manual_contact_info:
            save_contact_info(data_id, manual_contact_info)
        
        # Analyze the optimized resume to show improvement
        analyzer = ResumeAnalyzer()
        analysis_results = analyzer.analyze_resume(optimized_resume, job_description)
        
        # If we have manual contact info, update the analysis results
        if manual_contact_info:
            for key, value in manual_contact_info.items():
                if value:  # Only update if value is not empty
                    analysis_results['contact_info'][key] = value
        
        return render_template('optimized.html', 
                              results=analysis_results, 
                              optimized_resume=optimized_resume)
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error optimizing resume: {error_details}")
        flash(f'Error optimizing resume: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download-optimized')
def download_optimized():
    """Download the optimized resume as a DOCX file."""
    # Check if we have a resume data ID in session
    if 'resume_data_id' not in session:
        flash('No optimized resume to download. Please optimize a resume first.')
        return redirect(url_for('index'))
    
    try:
        # Get resume data from file storage
        data_id = session['resume_data_id']
        data = load_resume_data(data_id)
        
        if not data or 'optimized_resume' not in data:
            flash('Optimized resume not found. Please optimize your resume first.')
            return redirect(url_for('index'))
        
        optimized_resume = data['optimized_resume']
        original_filename = data.get('original_filename', 'resume.docx')
        
        # Create a DOCX document
        optimizer = ResumeOptimizer()
        doc = optimizer.generate_docx(optimized_resume)
        
        # Create an in-memory file
        file_stream = io.BytesIO()
        doc.save(file_stream)
        file_stream.seek(0)
        
        # Get the base filename without extension
        base_filename = os.path.splitext(original_filename)[0]
        download_filename = f"{base_filename}_optimized.docx"
        
        return send_file(
            file_stream,
            as_attachment=True,
            download_name=download_filename,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error downloading optimized resume: {error_details}")
        flash(f'Error downloading optimized resume: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True) 