import os
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
import tempfile
import traceback
import io
import uuid
import json
from datetime import timedelta, datetime
import logging
import time
import functools
from flask_wtf.csrf import CSRFProtect, CSRFError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/app.log'
)
logger = logging.getLogger('app')

# Import database functions
try:
    from database import save_resume_data, load_resume_data, save_optimized_resume, save_contact_info, save_industry_info, save_analysis_results, cleanup_expired_data
    DB_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing database module: {e}")
    DB_AVAILABLE = False
    # Define fallback functions that will be replaced by the real implementations

# Import parser functions with error handling
try:
    from utils.parser import extract_text_from_file
except ImportError as e:
    logger.error(f"Error importing parser: {e}")
    # Define a fallback function
    def extract_text_from_file(file_path):
        return f"ERROR: Parser module not available. Please install dependencies via install.py."

# Import analyzer with error handling
try:
    from models.analyzer import ResumeAnalyzer
    from models.optimizer import ResumeOptimizer
    from models.industry_scorer import IndustryScorer  # Add industry scorer import
except ImportError as e:
    logger.error(f"Error importing analyzer or optimizer: {e}")
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
            
    class IndustryScorer:
        def apply_industry_scoring(self, score_data, job_description, industry):
            return score_data

# Add API rate limiting
class RateLimiter:
    def __init__(self, max_calls=10, time_frame=60):
        """
        Initialize rate limiter
        max_calls: Maximum number of calls allowed in the time frame
        time_frame: Time frame in seconds
        """
        self.max_calls = max_calls
        self.time_frame = time_frame  # in seconds
        self.call_history = {}  # IP address -> list of timestamps
    
    def is_rate_limited(self, ip_address):
        """Check if the IP address is rate limited."""
        current_time = time.time()
        
        # If this is a new IP, initialize its history
        if ip_address not in self.call_history:
            self.call_history[ip_address] = []
        
        # Clean up old history
        self.call_history[ip_address] = [
            timestamp for timestamp in self.call_history[ip_address]
            if current_time - timestamp < self.time_frame
        ]
        
        # Check if rate limit is exceeded
        if len(self.call_history[ip_address]) >= self.max_calls:
            return True
        
        # Record the new call
        self.call_history[ip_address].append(current_time)
        return False

# Create rate limiter instance
api_rate_limiter = RateLimiter(max_calls=30, time_frame=60)  # 30 calls per minute

# Rate limiting decorator
def rate_limit(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        ip_address = request.remote_addr
        
        # Skip rate limiting for local development
        if ip_address in ['127.0.0.1', 'localhost']:
            return f(*args, **kwargs)
        
        if api_rate_limiter.is_rate_limited(ip_address):
            logger.warning(f"Rate limit exceeded for IP: {ip_address}")
            return jsonify({
                'error': 'Rate limit exceeded. Please try again later.',
                'status': 429
            }), 429
        
        return f(*args, **kwargs)
    
    return decorated_function

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'development-key-for-ats-app')
# Configure session to be permanent and last for 1 hour
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# File upload settings
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'doc', 'txt', 'rtf'}

# Create necessary directories
for directory in ['uploads', 'cache', 'logs', 'temp_storage']:
    os.makedirs(directory, exist_ok=True)

# Custom Jinja filters
@app.template_filter('now')
def _jinja2_filter_now(format_string):
    """Return formatted current time."""
    return datetime.now().strftime(format_string)

# Clean up expired data periodically
@app.before_request
def before_request():
    # Make sessions permanent
    session.permanent = True
    
    # Clean up expired data once per hour (approx)
    cleanup_time_file = 'temp_storage/last_cleanup.txt'
    current_time = time.time()
    last_cleanup_time = 0
    
    if os.path.exists(cleanup_time_file):
        try:
            with open(cleanup_time_file, 'r') as f:
                last_cleanup_time = float(f.read().strip())
        except (ValueError, OSError) as e:
            logger.error(f"Error reading last cleanup time: {e}")
    
    # If it's been more than an hour since the last cleanup
    if current_time - last_cleanup_time > 3600:  # 3600 seconds = 1 hour
        logger.info("Running scheduled cleanup of expired data")
        if DB_AVAILABLE:
            cleanup_expired_data()
        
        # Update the last cleanup time
        try:
            with open(cleanup_time_file, 'w') as f:
                f.write(str(current_time))
        except OSError as e:
            logger.error(f"Error writing last cleanup time: {e}")

def allowed_file(filename):
    """Check if filename has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    """Process resume upload and analysis."""
    start_time = time.time()
    logger.info("Starting resume analysis request")
    
    # Check if a file was uploaded
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    industry = request.form.get('industry', 'general')  # Get industry selection
    
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
            logger.info(f"Extracting text from file: {filename}")
            resume_text = extract_text_from_file(filepath)
            
            # Check if there was an error in text extraction
            if resume_text.startswith("ERROR:"):
                flash(resume_text)
                os.remove(filepath)
                return redirect(request.url)
            
            # Save resume data
            if DB_AVAILABLE:
                data_id = save_resume_data(resume_text, job_description, filename, industry)
            else:
                # Legacy file-based storage as fallback
                data_id = legacy_save_resume_data(resume_text, job_description, filename)
                # Save industry selection
                legacy_save_industry_info(data_id, industry)
            
            # Store only the data ID in the session
            session['resume_data_id'] = data_id
            
            # Analyze resume
            logger.info(f"Analyzing resume with ID: {data_id}")
            analyzer = ResumeAnalyzer()
            analysis_results = analyzer.analyze_resume(resume_text, job_description)
            
            # Apply industry-specific scoring if industry is specified
            if industry != 'general':
                logger.info(f"Applying {industry} industry scoring")
                industry_scorer = IndustryScorer()
                analysis_results = industry_scorer.apply_industry_scoring(
                    analysis_results, job_description, industry
                )
            
            # Save analysis results if database is available
            if DB_AVAILABLE:
                save_analysis_results(data_id, analysis_results)
            
            # Check if there was an error in analysis
            if "error" in analysis_results:
                flash(analysis_results["error"])
                os.remove(filepath)
                return redirect(request.url)
            
            # Clean up file after analysis
            os.remove(filepath)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
            
            return render_template('results.html', results=analysis_results, industry=industry)
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error analyzing resume: {error_details}")
            flash(f'Error analyzing resume: {str(e)}')
            # Try to remove the file if it exists
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            return redirect(request.url)
    else:
        allowed_extensions = ', '.join(app.config['ALLOWED_EXTENSIONS'])
        flash(f'File type not allowed. Please upload one of these types: {allowed_extensions}')
        return redirect(request.url)

@csrf.exempt
@app.route('/api/analyze', methods=['POST'])
@rate_limit  # Apply rate limiting to the API endpoint
def api_analyze_resume():
    """API endpoint for programmatic resume analysis."""
    start_time = time.time()
    logger.info("Starting API resume analysis request")
    
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['resume']
    job_description = request.form.get('job_description', '')
    industry = request.form.get('industry', 'general')  # Get industry parameter
    
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
            logger.info(f"API extracting text from file: {filename}")
            resume_text = extract_text_from_file(filepath)
            
            # Check if there was an error in text extraction
            if resume_text.startswith("ERROR:"):
                os.unlink(filepath)
                return jsonify({"error": resume_text}), 500
            
            # Analyze resume
            logger.info(f"API analyzing resume: {filename}")
            analyzer = ResumeAnalyzer()
            analysis_results = analyzer.analyze_resume(resume_text, job_description)
            
            # Apply industry-specific scoring if industry is specified
            if industry != 'general':
                logger.info(f"API applying {industry} industry scoring")
                industry_scorer = IndustryScorer()
                analysis_results = industry_scorer.apply_industry_scoring(
                    analysis_results, job_description, industry
                )
            
            # Clean up temporary file
            os.unlink(filepath)
            
            elapsed_time = time.time() - start_time
            logger.info(f"API analysis completed in {elapsed_time:.2f} seconds")
            
            return jsonify(analysis_results)
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"API error: {error_details}")
            # Clean up temporary file
            try:
                os.unlink(filepath)
            except:
                pass
            return jsonify({"error": str(e)}), 500
    else:
        allowed_extensions = ', '.join(app.config['ALLOWED_EXTENSIONS'])
        return jsonify({"error": f"File type not allowed. Please upload one of these types: {allowed_extensions}"}), 400

@app.route('/get_resume_text')
@rate_limit  # Apply rate limiting
def get_resume_text():
    """API endpoint to retrieve resume text for heatmap visualization."""
    logger.info("Getting resume text for heatmap")
    
    # Get data ID from request
    data_id = request.args.get('id') or session.get('resume_data_id')
    if not data_id:
        return jsonify({"error": "No resume data ID provided"}), 400
    
    # Load resume data
    if DB_AVAILABLE:
        data = load_resume_data(data_id)
    else:
        data = legacy_load_resume_data(data_id)
        
    if not data:
        return jsonify({"error": "Resume data not found"}), 404
    
    return jsonify({
        "resume_text": data.get('resume_text', ''),
        "original_filename": data.get('original_filename', '')
    })

@app.route('/industry_analysis', methods=['POST'])
def industry_analysis():
    """Re-analyze a resume with industry-specific scoring."""
    logger.info("Starting industry re-analysis")
    
    # Get data ID from session
    data_id = session.get('resume_data_id')
    if not data_id:
        flash('No resume data found. Please upload your resume first.')
        return redirect(url_for('index'))
    
    # Load resume data
    if DB_AVAILABLE:
        data = load_resume_data(data_id)
    else:
        data = legacy_load_resume_data(data_id)
        
    if not data:
        flash('Resume data could not be loaded. Please upload your resume again.')
        return redirect(url_for('index'))
    
    # Get selected industry
    industry = request.form.get('industry', 'general')
    
    # Save industry selection
    if DB_AVAILABLE:
        save_industry_info(data_id, industry)
    else:
        legacy_save_industry_info(data_id, industry)
    
    try:
        # Analyze resume
        logger.info(f"Re-analyzing resume with industry: {industry}")
        analyzer = ResumeAnalyzer()
        analysis_results = analyzer.analyze_resume(data['resume_text'], data['job_description'])
        
        # Apply industry-specific scoring if industry is specified
        if industry != 'general':
            industry_scorer = IndustryScorer()
            analysis_results = industry_scorer.apply_industry_scoring(
                analysis_results, data['job_description'], industry
            )
        
        # Save analysis results if database is available
        if DB_AVAILABLE:
            save_analysis_results(data_id, analysis_results)
        
        return render_template('results.html', results=analysis_results, industry=industry)
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in industry analysis: {error_details}")
        flash(f'Error analyzing resume for industry: {str(e)}')
        return redirect(url_for('index'))

@app.route('/optimize', methods=['POST'])
def optimize_resume():
    """Optimize the resume based on job description."""
    logger.info("Starting resume optimization")
    
    # Get data ID from session
    data_id = session.get('resume_data_id')
    if not data_id:
        flash('No resume data found. Please upload your resume first.')
        return redirect(url_for('index'))
    
    # Load resume data
    if DB_AVAILABLE:
        data = load_resume_data(data_id)
    else:
        data = legacy_load_resume_data(data_id)
        
    if not data:
        flash('Resume data could not be loaded. Please upload your resume again.')
        return redirect(url_for('index'))
    
    # Get customizations from form if submitted, otherwise use empty dict
    customizations = {}
    
    # If contact info was submitted, update it
    if 'name' in request.form:
        contact_info = {
            'name': request.form.get('name', ''),
            'email': request.form.get('email', ''),
            'phone': request.form.get('phone', ''),
            'linkedin': request.form.get('linkedin', '')
        }
        
        if DB_AVAILABLE:
            save_contact_info(data_id, contact_info)
        else:
            legacy_save_contact_info(data_id, contact_info)
            
        customizations['contact_info'] = contact_info
    
    try:
        # Get industry
        industry = data.get('industry', 'general')
        
        # Create optimizer
        logger.info(f"Optimizing resume for industry: {industry}")
        optimizer = ResumeOptimizer()
        
        # Optimize resume
        optimized_resume = optimizer.optimize_resume(
            data['resume_text'], 
            data['job_description'],
            industry=industry,  # Pass industry for specialized optimization
            customizations=customizations
        )
        
        # Save optimized resume
        if DB_AVAILABLE:
            success = save_optimized_resume(data_id, optimized_resume)
        else:
            success = legacy_save_optimized_resume(data_id, optimized_resume)
            
        if not success:
            flash('Error saving optimized resume.')
            return redirect(url_for('index'))
        
        # Analyze optimized resume
        logger.info("Analyzing optimized resume")
        analyzer = ResumeAnalyzer()
        optimized_analysis = analyzer.analyze_resume(optimized_resume, data['job_description'])
        
        # Apply industry-specific scoring if industry is specified
        if industry != 'general':
            industry_scorer = IndustryScorer()
            optimized_analysis = industry_scorer.apply_industry_scoring(
                optimized_analysis, data['job_description'], industry
            )
        
        return render_template(
            'optimization_results.html',
            original_resume=data['resume_text'],
            optimized_resume=optimized_resume,
            results=optimized_analysis,
            industry=industry
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error optimizing resume: {error_details}")
        flash(f'Error optimizing resume: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download-optimized')
def download_optimized():
    """Download the optimized resume as a DOCX file."""
    logger.info("Downloading optimized resume")
    
    # Get data ID from session
    data_id = session.get('resume_data_id')
    if not data_id:
        flash('No resume data found. Please upload your resume first.')
        return redirect(url_for('index'))
    
    # Load resume data
    if DB_AVAILABLE:
        data = load_resume_data(data_id)
    else:
        data = legacy_load_resume_data(data_id)
        
    if not data or 'optimized_resume' not in data:
        flash('No optimized resume found. Please optimize your resume first.')
        return redirect(url_for('index'))
    
    try:
        # Create DOCX file
        logger.info("Generating DOCX file")
        optimizer = ResumeOptimizer()
        docx_bytes = optimizer.generate_docx(data['optimized_resume'])
        
        if not docx_bytes:
            flash('Error generating DOCX file.')
            return redirect(url_for('index'))
        
        # Get the original filename without extension
        original_name = os.path.splitext(data['original_filename'])[0]
        
        # Create response with DOCX file
        mem_file = io.BytesIO(docx_bytes)
        mem_file.seek(0)
        
        return send_file(
            mem_file,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            as_attachment=True,
            download_name=f"{original_name}-optimized.docx"
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error generating DOCX: {error_details}")
        flash(f'Error generating DOCX: {str(e)}')
        return redirect(url_for('index'))

@app.route('/compare-analysis')
def compare_analysis():
    """Compare original and optimized resume analysis."""
    logger.info("Starting resume comparison")
    
    # Get data ID from session
    data_id = session.get('resume_data_id')
    if not data_id:
        flash('No resume data found. Please upload your resume first.')
        return redirect(url_for('index'))
    
    # Load resume data
    if DB_AVAILABLE:
        data = load_resume_data(data_id)
    else:
        data = legacy_load_resume_data(data_id)
        
    if not data or 'optimized_resume' not in data:
        flash('No optimized resume found. Please optimize your resume first.')
        return redirect(url_for('index'))
    
    try:
        # Get industry
        industry = data.get('industry', 'general')
        
        # Analyze both resumes
        logger.info("Analyzing original and optimized resumes for comparison")
        analyzer = ResumeAnalyzer()
        original_analysis = analyzer.analyze_resume(data['resume_text'], data['job_description'])
        optimized_analysis = analyzer.analyze_resume(data['optimized_resume'], data['job_description'])
        
        # Apply industry-specific scoring if industry is specified
        if industry != 'general':
            industry_scorer = IndustryScorer()
            original_analysis = industry_scorer.apply_industry_scoring(
                original_analysis, data['job_description'], industry
            )
            optimized_analysis = industry_scorer.apply_industry_scoring(
                optimized_analysis, data['job_description'], industry
            )
        
        return render_template(
            'comparison.html',
            original=original_analysis,
            optimized=optimized_analysis,
            industry=industry
        )
        
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error comparing resumes: {error_details}")
        flash(f'Error comparing resumes: {str(e)}')
        return redirect(url_for('index'))

@app.route('/optimization_wizard')
def optimization_wizard():
    """Start the guided resume optimization wizard."""
    logger.info("Starting optimization wizard")
    
    # Get data ID from session
    data_id = session.get('resume_data_id')
    if not data_id:
        flash('No resume data found. Please upload your resume first.')
        return redirect(url_for('index'))
    
    # Load resume data
    if DB_AVAILABLE:
        data = load_resume_data(data_id)
    else:
        data = legacy_load_resume_data(data_id)
        
    if not data:
        flash('Resume data could not be loaded. Please upload your resume again.')
        return redirect(url_for('index'))
    
    # Initialize wizard session data if not exists
    if 'wizard_data' not in session:
        # Analyze resume to get initial data
        analyzer = ResumeAnalyzer()
        analysis_results = analyzer.analyze_resume(data['resume_text'], data['job_description'])
        
        # Apply industry-specific scoring if industry is specified
        industry = data.get('industry', 'general')
        if industry != 'general':
            industry_scorer = IndustryScorer()
            analysis_results = industry_scorer.apply_industry_scoring(
                analysis_results, data['job_description'], industry
            )
        
        # Initialize wizard data
        session['wizard_data'] = {
            'step': 'contact',
            'step_completed': {
                'contact': False,
                'sections': False,
                'keywords': False,
                'content': False,
                'format': False,
                'review': False
            },
            'contact_info': analysis_results.get('contact_info', {}),
            'sections': analysis_results.get('sections', {}),
            'matched_keywords': analysis_results.get('match_results', {}).get('matched_keywords', []),
            'missing_keywords': analysis_results.get('match_results', {}).get('missing_keywords', []),
            'format_issues': analysis_results.get('format_issues', []),
            'section_order': [
                'summary', 'experience', 'education', 'skills', 'projects', 
                'certifications', 'awards', 'publications', 'references'
            ],
            'optimization_data': {}
        }
    
    # Get current wizard data
    wizard_data = session['wizard_data']
    
    return render_template(
        'optimization_wizard.html',
        step=wizard_data['step'],
        step_completed=wizard_data['step_completed'],
        resume_id=data_id,
        contact_info=wizard_data['contact_info'],
        resume_sections=wizard_data['sections'],
        section_order=wizard_data['section_order'],
        matched_keywords=wizard_data['matched_keywords'],
        missing_keywords=wizard_data['missing_keywords'],
        format_issues=wizard_data['format_issues']
    )

@app.route('/process_optimization_step', methods=['POST'])
def process_optimization_step():
    """Process each step of the optimization wizard."""
    logger.info("Processing optimization wizard step")
    
    # Get current step and resume ID
    current_step = request.form.get('current_step')
    resume_id = request.form.get('resume_id')
    action = request.form.get('action', 'next')
    
    if not current_step or not resume_id:
        flash('Missing step information. Please try again.')
        return redirect(url_for('index'))
    
    # Load resume data
    if DB_AVAILABLE:
        data = load_resume_data(resume_id)
    else:
        data = legacy_load_resume_data(resume_id)
        
    if not data:
        flash('Resume data could not be loaded. Please upload your resume again.')
        return redirect(url_for('index'))
    
    # Get wizard data from session
    wizard_data = session.get('wizard_data', {})
    if not wizard_data:
        return redirect(url_for('optimization_wizard'))
    
    # Process "previous" action - go back to previous step
    if action == 'prev':
        prev_step = get_previous_step(current_step)
        wizard_data['step'] = prev_step
        session['wizard_data'] = wizard_data
        return redirect(url_for('optimization_wizard'))
    
    # Process "download" action from review step
    if action == 'download' and current_step == 'review':
        return process_final_optimization(wizard_data, data)
    
    # Process current step data
    if current_step == 'contact':
        process_contact_step(wizard_data, request.form)
    elif current_step == 'sections':
        process_sections_step(wizard_data, request.form)
    elif current_step == 'keywords':
        process_keywords_step(wizard_data, request.form)
    elif current_step == 'content':
        process_content_step(wizard_data, request.form)
    elif current_step == 'format':
        process_format_step(wizard_data, request.form)
    elif current_step == 'review':
        # Review step doesn't need processing, it's just a preview
        pass
    
    # Mark current step as completed
    wizard_data['step_completed'][current_step] = True
    
    # Move to next step if action is 'next'
    if action == 'next':
        next_step = get_next_step(current_step)
        wizard_data['step'] = next_step
        
        # If moving to review step, generate the optimized resume
        if next_step == 'review':
            optimized_data = generate_optimized_resume(wizard_data, data)
            wizard_data.update(optimized_data)
    
    # Save updated wizard data to session
    session['wizard_data'] = wizard_data
    
    return redirect(url_for('optimization_wizard'))

def get_next_step(current_step):
    """Get the next step in the wizard sequence."""
    steps = ['contact', 'sections', 'keywords', 'content', 'format', 'review']
    current_index = steps.index(current_step)
    next_index = min(current_index + 1, len(steps) - 1)
    return steps[next_index]

def get_previous_step(current_step):
    """Get the previous step in the wizard sequence."""
    steps = ['contact', 'sections', 'keywords', 'content', 'format', 'review']
    current_index = steps.index(current_step)
    prev_index = max(current_index - 1, 0)
    return steps[prev_index]

def process_contact_step(wizard_data, form_data):
    """Process contact information step data."""
    wizard_data['contact_info'] = {
        'name': form_data.get('name', ''),
        'email': form_data.get('email', ''),
        'phone': form_data.get('phone', ''),
        'linkedin': form_data.get('linkedin', '')
    }
    
    # Store in optimization data
    wizard_data['optimization_data']['contact_info'] = wizard_data['contact_info']

def process_sections_step(wizard_data, form_data):
    """Process resume sections step data."""
    # Get section order from form
    section_order_str = form_data.get('section_order', '')
    if section_order_str:
        wizard_data['section_order'] = section_order_str.split(',')
    
    # Get included sections
    included_sections = request.form.getlist('included_sections')
    
    # Filter sections to only include the selected ones
    selected_sections = {}
    for section_name in included_sections:
        if section_name in wizard_data['sections']:
            selected_sections[section_name] = wizard_data['sections'][section_name]
    
    # Store in optimization data
    wizard_data['optimization_data']['section_order'] = wizard_data['section_order']
    wizard_data['optimization_data']['included_sections'] = included_sections
    wizard_data['selected_sections'] = selected_sections

def process_keywords_step(wizard_data, form_data):
    """Process keywords step data."""
    # Get selected keywords to add
    add_keywords = request.form.getlist('add_keywords')
    
    # Get section assignments for each keyword
    keyword_sections = {}
    for keyword in add_keywords:
        section = form_data.get(f'keyword_section_{keyword}', '')
        if section:
            keyword_sections[keyword] = section
    
    # Store in optimization data
    wizard_data['optimization_data']['add_keywords'] = add_keywords
    wizard_data['optimization_data']['keyword_sections'] = keyword_sections

def process_content_step(wizard_data, form_data):
    """Process content enhancement step data."""
    # Get enhancement options
    enhance_action_verbs = form_data.get('enhance_action_verbs') == 'on'
    enhance_quantifiable = form_data.get('enhance_quantifiable') == 'on'
    
    # Store in optimization data
    wizard_data['optimization_data']['enhance_action_verbs'] = enhance_action_verbs
    wizard_data['optimization_data']['enhance_quantifiable'] = enhance_quantifiable

def process_format_step(wizard_data, form_data):
    """Process formatting step data."""
    # Get formatting options
    fix_format_issues = form_data.get('fix_format_issues') == 'on'
    
    # Store in optimization data
    wizard_data['optimization_data']['fix_format_issues'] = fix_format_issues

def generate_optimized_resume(wizard_data, data):
    """Generate the optimized resume based on wizard selections."""
    try:
        # Create optimizer
        optimizer = ResumeOptimizer()
        
        # Prepare optimization settings
        optimization_config = {
            'contact_info': wizard_data['optimization_data'].get('contact_info', {}),
            'section_order': wizard_data['optimization_data'].get('section_order', []),
            'included_sections': wizard_data['optimization_data'].get('included_sections', []),
            'add_keywords': wizard_data['optimization_data'].get('add_keywords', []),
            'keyword_sections': wizard_data['optimization_data'].get('keyword_sections', {}),
            'enhance_action_verbs': wizard_data['optimization_data'].get('enhance_action_verbs', True),
            'enhance_quantifiable': wizard_data['optimization_data'].get('enhance_quantifiable', True),
            'fix_format_issues': wizard_data['optimization_data'].get('fix_format_issues', True)
        }
        
        # Optimize resume with customizations
        optimized_resume = optimizer.optimize_resume(
            data['resume_text'], 
            data['job_description'],
            industry=data.get('industry', 'general'),
            customizations=optimization_config
        )
        
        # Extract optimization stats
        stats = optimizer.get_optimization_stats()
        
        # Get optimized sections
        analyzer = ResumeAnalyzer()
        optimized_sections = analyzer.identify_resume_sections(optimized_resume)
        
        # Sort sections according to configured order
        final_section_order = []
        for section in optimization_config['section_order']:
            if section in optimization_config['included_sections']:
                final_section_order.append(section)
        
        return {
            'optimized_resume': optimized_resume,
            'optimized_sections': {name: section['content'] for name, section in optimized_sections.items()},
            'final_section_order': final_section_order,
            'stats': stats
        }
    except Exception as e:
        logger.error(f"Error generating optimized resume: {str(e)}")
        return {
            'optimized_resume': data['resume_text'],
            'optimized_sections': wizard_data['sections'],
            'final_section_order': wizard_data['section_order'],
            'stats': {
                'keywords_added': 0,
                'action_verbs_enhanced': 0,
                'quantifiable_added': 0,
                'format_issues_fixed': 0
            }
        }

def process_final_optimization(wizard_data, data):
    """Process final optimization and generate downloadable resume."""
    try:
        optimized_resume = wizard_data.get('optimized_resume', data['resume_text'])
        
        # Save the optimized resume
        if DB_AVAILABLE:
            success = save_optimized_resume(data['id'], optimized_resume)
        else:
            success = legacy_save_optimized_resume(data['id'], optimized_resume)
            
        if not success:
            flash('Error saving optimized resume.')
            return redirect(url_for('index'))
        
        # Generate DOCX file
        optimizer = ResumeOptimizer()
        docx_bytes = optimizer.generate_docx(optimized_resume)
        
        if not docx_bytes:
            flash('Error generating DOCX file.')
            return redirect(url_for('optimization_wizard'))
        
        # Get the original filename without extension
        original_name = os.path.splitext(data['original_filename'])[0]
        
        # Create response with DOCX file
        mem_file = io.BytesIO(docx_bytes)
        mem_file.seek(0)
        
        return send_file(
            mem_file,
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            as_attachment=True,
            download_name=f"{original_name}-optimized.docx"
        )
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in final optimization: {error_details}")
        flash(f'Error optimizing resume: {str(e)}')
        return redirect(url_for('optimization_wizard'))

# Legacy file-based storage functions for fallback when DB is not available
def legacy_save_resume_data(resume_text, job_description, original_filename):
    """Legacy fallback: Save resume data to a temporary file and return a reference ID."""
    # Generate a unique ID
    data_id = str(uuid.uuid4())
    
    # Create data object
    data = {
        'resume_text': resume_text,
        'job_description': job_description,
        'original_filename': original_filename
    }
    
    # Save to file
    file_path = os.path.join('temp_storage', f"{data_id}.json")
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    return data_id

def legacy_load_resume_data(data_id):
    """Legacy fallback: Load resume data from a temporary file."""
    file_path = os.path.join('temp_storage', f"{data_id}.json")
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading resume data: {e}")
        return None

def legacy_save_optimized_resume(data_id, optimized_resume):
    """Legacy fallback: Save optimized resume to the existing data file."""
    file_path = os.path.join('temp_storage', f"{data_id}.json")
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
        logger.error(f"Error saving optimized resume: {e}")
        return False

def legacy_save_contact_info(data_id, contact_info):
    """Legacy fallback: Save manual contact info to the existing data file."""
    file_path = os.path.join('temp_storage', f"{data_id}.json")
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
        logger.error(f"Error saving contact info: {e}")
        return False

def legacy_save_industry_info(data_id, industry):
    """Legacy fallback: Save industry selection to the existing data file."""
    file_path = os.path.join('temp_storage', f"{data_id}.json")
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Add or update industry
        data['industry'] = industry
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        return True
    except Exception as e:
        logger.error(f"Error saving industry info: {e}")
        return False

# Add CSRF error handler
@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    logger.warning(f"CSRF error: {e}")
    if request.accept_mimetypes.accept_json:
        return jsonify({"error": "CSRF token is missing or invalid"}), 400
    flash("Your session has expired or the form has been tampered with. Please try again.")
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Log application startup
    logger.info("Starting ATS Resume Checker application")
    app.run(debug=True) 