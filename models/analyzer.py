import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("spaCy model not found. Please run: python -m spacy download en_core_web_md")
    nlp = None

class ResumeAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess text."""
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and short words
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        return ' '.join(tokens)
    
    def extract_keywords(self, text, n=30):
        """Extract important keywords using TF-IDF."""
        preprocessed_text = self.preprocess_text(text)
        
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
        
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores
        scores = zip(feature_names, np.asarray(tfidf_matrix.sum(axis=0)).ravel())
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return [item[0] for item in sorted_scores[:n]]
    
    def extract_skills(self, text):
        """Extract potential skills from text using spaCy's NER and noun chunks."""
        if nlp is None:
            return []
            
        doc = nlp(text)
        
        # Extract noun chunks (potential skills)
        noun_chunks = set([chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3])
        
        # Extract named entities that might be skills
        entities = set([ent.text.lower() for ent in doc.ents 
                       if ent.label_ in ["ORG", "PRODUCT"] and len(ent.text) > 2])
        
        return list(noun_chunks.union(entities))
    
    def calculate_match_score(self, resume_text, job_description):
        """Calculate how well the resume matches the job description."""
        if nlp is None:
            return {"score": 0, "matched_keywords": [], "missing_keywords": []}
            
        # Preprocess texts
        processed_resume = self.preprocess_text(resume_text)
        processed_jd = self.preprocess_text(job_description)
        
        # Extract keywords
        jd_keywords = self.extract_keywords(job_description)
        resume_keywords = self.extract_keywords(resume_text)
        
        # Find matching keywords
        matched_keywords = [keyword for keyword in jd_keywords if keyword in processed_resume]
        missing_keywords = [keyword for keyword in jd_keywords if keyword not in processed_resume]
        
        # Calculate simple match percentage
        match_percentage = (len(matched_keywords) / len(jd_keywords)) * 100 if jd_keywords else 0
        
        # Calculate semantic similarity using spaCy
        resume_doc = nlp(processed_resume[:1000000])  # Limit size to avoid memory issues
        jd_doc = nlp(processed_jd[:1000000])  # Limit size to avoid memory issues
        
        # Use a try-except block to handle potential errors during similarity calculation
        try:
            semantic_similarity = resume_doc.similarity(jd_doc) * 100
        except Exception:
            # Fallback to a basic similarity if spaCy's similarity fails
            semantic_similarity = match_percentage
        
        # Calculate combined score (50% keyword match, 50% semantic similarity)
        combined_score = (match_percentage + semantic_similarity) / 2
        
        return {
            "exact_match_score": round(match_percentage, 1),
            "semantic_match_score": round(semantic_similarity, 1),
            "combined_score": round(combined_score, 1),
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords
        }
    
    def identify_resume_sections(self, text):
        """Identify common resume sections."""
        sections = {}
        
        # Common section headers
        section_patterns = {
            "contact": r"(?i)(contact|personal) information",
            "summary": r"(?i)(summary|profile|objective)",
            "experience": r"(?i)(experience|work|employment|history)",
            "education": r"(?i)(education|academic|qualification)",
            "skills": r"(?i)(skills|expertise|competencies|technical)",
            "projects": r"(?i)(projects|portfolio)",
            "certifications": r"(?i)(certifications|certificates|licenses)",
            "languages": r"(?i)(languages|language skills)",
        }
        
        lines = text.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a section header
            for section, pattern in section_patterns.items():
                if re.search(pattern, line):
                    current_section = section
                    sections[section] = {"start": i}
                    # Close previous section if it exists
                    for s in sections:
                        if s != section and "start" in sections[s] and "end" not in sections[s]:
                            sections[s]["end"] = i
                    break
                    
        # Close the last section
        if current_section and "end" not in sections[current_section]:
            sections[current_section]["end"] = len(lines)
            
        # Extract section content
        for section in sections:
            if "start" in sections[section] and "end" in sections[section]:
                start = sections[section]["start"] + 1  # Skip header line
                end = sections[section]["end"]
                sections[section]["content"] = "\n".join(lines[start:end])
                
        return sections
    
    def analyze_resume(self, resume_text, job_description):
        """Perform comprehensive resume analysis."""
        # Match score
        match_results = self.calculate_match_score(resume_text, job_description)
        
        # Section identification
        sections = self.identify_resume_sections(resume_text)
        
        # Extract contact info
        from utils.parser import extract_contact_info
        contact_info = extract_contact_info(resume_text)
        
        # Format check
        format_issues = self.check_formatting(resume_text)
        
        # Action verbs check
        action_verb_results = self.check_action_verbs(resume_text)
        
        # Quantifiable results check
        quantifiable_results = self.check_quantifiable_results(resume_text)
        
        return {
            "contact_info": contact_info,
            "match_results": match_results,
            "sections": sections,
            "format_issues": format_issues,
            "action_verbs": action_verb_results,
            "quantifiable_results": quantifiable_results
        }
    
    def check_formatting(self, text):
        """Check for potential formatting issues."""
        issues = []
        
        # Check for potential table structures (simplified)
        if re.search(r"\|\s+\|", text):
            issues.append("Possible table structure detected")
            
        # Check for excessive line breaks
        if text.count('\n\n\n') > 3:
            issues.append("Multiple excessive line breaks detected")
            
        # Check length (assuming about 500 chars per page)
        if len(text) > 5000:  # Roughly 2+ pages
            issues.append("Resume may be too long (appears to be more than 2 pages)")
            
        # Check for potential header/footer text
        lines = text.split('\n')
        if len(lines) > 10:
            first_line = lines[0].strip()
            last_line = lines[-1].strip()
            if re.search(r"\d+", first_line) or re.search(r"page", first_line.lower()):
                issues.append("Possible header detected")
            if re.search(r"\d+", last_line) or re.search(r"page", last_line.lower()):
                issues.append("Possible footer detected")
                
        return issues
    
    def check_action_verbs(self, text):
        """Check for strong action verbs in experience sections."""
        common_action_verbs = [
            "achieved", "improved", "trained", "managed", "created", "resolved",
            "delivered", "increased", "decreased", "negotiated", "influenced",
            "led", "launched", "developed", "implemented", "coordinated",
            "produced", "founded", "initiated", "presented", "advanced",
            "established", "expanded", "optimized", "redesigned", "simplified"
        ]
        
        sections = self.identify_resume_sections(text)
        
        if "experience" not in sections:
            return {"has_action_verbs": False, "count": 0, "missing": common_action_verbs[:5]}
            
        experience_text = sections["experience"].get("content", "")
        experience_lines = [line.strip() for line in experience_text.split('\n') if line.strip()]
        
        found_verbs = []
        for verb in common_action_verbs:
            if re.search(r'\b' + verb + r'\b', experience_text.lower()):
                found_verbs.append(verb)
                
        bullet_points = [line for line in experience_lines if line.startswith('•') or line.startswith('-')]
        bullets_with_action_verbs = 0
        
        for bullet in bullet_points:
            has_action_verb = False
            for verb in common_action_verbs:
                if re.search(r'\b' + verb + r'\b', bullet.lower()):
                    has_action_verb = True
                    break
            if has_action_verb:
                bullets_with_action_verbs += 1
                
        missing_verbs = [verb for verb in common_action_verbs if verb not in found_verbs]
        
        return {
            "has_action_verbs": len(found_verbs) > 0,
            "count": len(found_verbs),
            "found": found_verbs,
            "missing": missing_verbs[:5],  # Just show first 5 missing verbs
            "bullet_percentage": round((bullets_with_action_verbs / len(bullet_points)) * 100, 1) if bullet_points else 0
        }
        
    def check_quantifiable_results(self, text):
        """Check for quantifiable results in experience section."""
        sections = self.identify_resume_sections(text)
        
        if "experience" not in sections:
            return {"has_quantifiable_results": False}
            
        experience_text = sections["experience"].get("content", "")
        experience_lines = [line.strip() for line in experience_text.split('\n') if line.strip()]
        
        # Look for numbers, percentages, and dollar amounts
        number_pattern = r'\b\d+[%]?\b'
        money_pattern = r'\$\s?\d+'
        
        has_numbers = bool(re.search(number_pattern, experience_text))
        has_money = bool(re.search(money_pattern, experience_text))
        
        bullet_points = [line for line in experience_lines if line.startswith('•') or line.startswith('-')]
        bullets_with_numbers = 0
        
        for bullet in bullet_points:
            if re.search(number_pattern, bullet) or re.search(money_pattern, bullet):
                bullets_with_numbers += 1
                
        return {
            "has_quantifiable_results": has_numbers or has_money,
            "bullet_percentage": round((bullets_with_numbers / len(bullet_points)) * 100, 1) if bullet_points else 0
        } 