import re
import spacy
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("spaCy model not found. Please run: python -m spacy download en_core_web_md")
    nlp = None

class ResumeAnalyzer:
    # Define common section headers
    COMMON_SECTION_HEADERS = [
        'experience', 'work experience', 'employment history',
        'education', 'skills', 'technical skills',
        'projects', 'certifications', 'awards',
        'summary', 'professional summary', 'objective',
        'languages', 'publications', 'references'
    ]
    
    # Define recommended section order
    RECOMMENDED_ORDER = [
        'summary', 'experience', 'education', 'skills', 'projects',
        'certifications', 'languages', 'publications', 'references'
    ]
    
    # Define essential sections
    ESSENTIAL_SECTIONS = ['experience', 'education', 'skills']
    
    # Scoring weights
    SCORING_WEIGHTS = {
        'keyword_match': 0.35,
        'keyword_position': 0.25,
        'keyword_density': 0.15,
        'section_organization': 0.15,
        'formatting': 0.10
    }
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text):
        """Clean and preprocess text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_keywords(self, text, n=30):
        """Extract important keywords using TF-IDF."""
        preprocessed_text = self.preprocess_text(text)
        
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
            
            # Get feature names (words)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            scores = zip(feature_names, np.asarray(tfidf_matrix.sum(axis=0)).ravel())
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Return top N keywords with scores
            return [(item[0], float(item[1])) for item in sorted_scores[:n]]
        except ValueError:
            # Handle empty text
            return []
    
    def get_top_keywords(self, text, n=20):
        """Get top n keywords from text"""
        keywords = self.extract_keywords(text, n)
        return [word for word, _ in keywords]
    
    def get_important_phrases(self, text):
        """Extract important phrases using spaCy"""
        if nlp is None:
            return []
            
        doc = nlp(text[:100000])  # Limit size to avoid memory issues
        phrases = []
        
        # Extract noun phrases and named entities
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to phrases with 3 or fewer words
                phrases.append(chunk.text.lower())
                
        # Add named entities
        for ent in doc.ents:
            if len(ent.text.split()) <= 3:
                phrases.append(ent.text.lower())
                
        return list(set(phrases))
    
    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts using spaCy"""
        if nlp is None:
            return 0.0
            
        doc1 = nlp(text1[:10000])  # Limit size
        doc2 = nlp(text2[:10000])  # Limit size
        
        try:
            return doc1.similarity(doc2)
        except:
            return 0.0
    
    def get_synonyms(self, word):
        """Get synonyms for a word using WordNet"""
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        
        return list(synonyms)
    
    def expand_keywords_with_synonyms(self, keywords):
        """Expand keywords with their synonyms"""
        expanded_keywords = set(keywords)
        
        for keyword in keywords:
            synonyms = self.get_synonyms(keyword)
            expanded_keywords.update(synonyms)
        
        return list(expanded_keywords)
    
    def calculate_keyword_similarities(self, resume_keywords, job_keywords):
        """Calculate similarities between resume keywords and job keywords"""
        similarities = []
        
        for resume_kw in resume_keywords:
            for job_kw in job_keywords:
                similarity = self.calculate_semantic_similarity(resume_kw, job_kw)
                
                if similarity > 0.6:  # Only consider significant similarities
                    similarities.append({
                        'resume_keyword': resume_kw,
                        'job_keyword': job_kw,
                        'similarity': float(similarity)
                    })
        
        return sorted(similarities, key=lambda x: x['similarity'], reverse=True)
    
    def calculate_position_score(self, resume_text, keyword):
        """Calculate how early the keyword appears in the resume"""
        # Split resume into sections
        sections = re.split(r'\n{2,}', resume_text)
        total_sections = len(sections)
        
        # Find which section contains the keyword
        for i, section in enumerate(sections):
            if keyword.lower() in section.lower():
                # Earlier sections get higher scores
                return 1 - (i / total_sections)
        
        return 0
    
    def calculate_keyword_density(self, resume_text, keywords):
        """Calculate keyword density in the resume"""
        resume_words = word_tokenize(resume_text.lower())
        resume_length = len(resume_words)
        
        if resume_length == 0:
            return 0
        
        # Count occurrences of keywords
        keyword_count = 0
        for keyword in keywords:
            keyword_count += resume_text.lower().count(keyword.lower())
        
        # Calculate density
        density = keyword_count / resume_length
        
        # Normalize density (ideal range is around 3-5%)
        if density > 0.10:  # Too many keywords (keyword stuffing)
            normalized_density = 0.5 + (0.10 / density) * 0.5  # Penalize stuffing
        else:
            normalized_density = min(1.0, density / 0.05)  # Ideal at 5%
            
        return normalized_density
    
    def detect_sections(self, resume_text):
        """Detect sections in the resume"""
        lines = resume_text.split('\n')
        sections = []
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if this line is a section header
            trimmed_line = line.strip().lower()
            is_header = any(
                trimmed_line == header or 
                trimmed_line.startswith(header + ':') or
                trimmed_line.startswith(header.upper()) or
                trimmed_line.startswith(header.title())
                for header in self.COMMON_SECTION_HEADERS
            )
            
            # Also check for headers with all caps or title case
            is_header = is_header or (
                trimmed_line.isupper() and len(trimmed_line) > 3 and len(trimmed_line) < 25
            )
            
            if is_header:
                # Save previous section if it exists
                if current_section:
                    sections.append({
                        'name': current_section,
                        'content': '\n'.join(current_content)
                    })
                
                # Start new section
                current_section = trimmed_line
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Add the final section
        if current_section:
            sections.append({
                'name': current_section,
                'content': '\n'.join(current_content)
            })
        
        return sections
    
    def analyze_section_structure(self, resume_text):
        """Analyze section structure for completeness and organization"""
        sections = self.detect_sections(resume_text)
        section_names = [s['name'].lower() for s in sections]
        
        # Check for essential sections
        has_essential_sections = all(
            any(essential in name for name in section_names)
            for essential in self.ESSENTIAL_SECTIONS
        )
        
        # Check for recommended order
        order_score = 0
        for i, recommended_section in enumerate(self.RECOMMENDED_ORDER):
            section_index = next(
                (idx for idx, name in enumerate(section_names) 
                 if recommended_section in name),
                -1
            )
            
            if section_index != -1:
                # Check if this section appears in proper order relative to previous sections
                previous_sections_in_order = self.RECOMMENDED_ORDER[:i]
                previous_section_indices = [
                    next((idx for idx, name in enumerate(section_names) 
                          if prev_section in name), -1)
                    for prev_section in previous_sections_in_order
                ]
                
                if all(prev_idx < section_index for prev_idx in previous_section_indices if prev_idx != -1):
                    order_score += 1
        
        # Normalize order score
        max_possible_score = sum(1 for section in self.RECOMMENDED_ORDER 
                                if any(section in name for name in section_names))
        normalized_order_score = order_score / max_possible_score if max_possible_score > 0 else 0
        
        return normalized_order_score
    
    def analyze_formatting(self, resume_text):
        """Detect formatting issues that could cause ATS problems"""
        issues = []
        score = 1.0  # Start with perfect score
        
        # Check for tables (approximation by detecting patterns)
        has_table_pattern = bool(re.search(r'\|.*\|', resume_text) or 
                                 re.search(r'\+[-+]+\+', resume_text))
        if has_table_pattern:
            issues.append('Potential table detected - tables may not parse correctly in ATS systems')
            score -= 0.2
        
        # Check for excessive bullet points
        bullet_point_count = len(re.findall(r'•|\*|\-\s', resume_text))
        if bullet_point_count > 30:
            issues.append('Excessive bullet points detected - consider consolidating some points')
            score -= 0.1
        
        # Check for complex formatting
        has_complex_formatting = bool(re.search(r'[^\x00-\x7F]', resume_text))  # Non-ASCII chars
        if has_complex_formatting:
            issues.append('Special characters detected - these may cause parsing issues')
            score -= 0.15
        
        # Check for headers/footers
        possible_header_footer = bool(re.search(r'page \d of \d|^\d+$|^\s*\d+\s*$', resume_text, re.MULTILINE))
        if possible_header_footer:
            issues.append('Possible header/footer detected - these may confuse ATS parsing')
            score -= 0.15
        
        # Check for contact info in header
        first_lines = '\n'.join(resume_text.split('\n')[:5])
        has_contact_info = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', first_lines) or
                               re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', first_lines))
        if not has_contact_info:
            issues.append('Contact information should be at the top of resume for proper ATS parsing')
            score -= 0.1
        
        return {
            'score': max(0, score),
            'issues': issues
        }
    
    def calculate_match_score(self, resume_text, job_description, synonym_expansion=True):
        """Calculate comprehensive ATS score"""
        # Extract keywords from job and resume
        job_keywords = [kw for kw, _ in self.extract_keywords(job_description, 20)]
        resume_keywords = [kw for kw, _ in self.extract_keywords(resume_text, 30)]
        
        # Also get important phrases
        job_phrases = self.get_important_phrases(job_description)
        
        # Expand with synonyms if enabled
        if synonym_expansion:
            job_keywords = self.expand_keywords_with_synonyms(job_keywords)
        
        # Track matches and their details
        matches = []
        missing_keywords = []
        
        # Process each job keyword
        for job_keyword in job_keywords:
            found = False
            
            # Check for direct matches
            if job_keyword.lower() in [kw.lower() for kw in resume_keywords]:
                position_score = self.calculate_position_score(resume_text, job_keyword)
                matches.append({
                    'keyword': job_keyword,
                    'match_type': 'direct',
                    'position_score': position_score,
                    'score': 1.0
                })
                found = True
            
            # Check for semantic matches using spaCy
            elif synonym_expansion and nlp is not None:
                for resume_kw in resume_keywords:
                    similarity = self.calculate_semantic_similarity(job_keyword, resume_kw)
                    if similarity > 0.75:  # High similarity threshold
                        position_score = self.calculate_position_score(resume_text, resume_kw)
                        matches.append({
                            'keyword': job_keyword,
                            'matched_with': resume_kw,
                            'match_type': 'semantic',
                            'position_score': position_score,
                            'score': similarity
                        })
                        found = True
                        break
            
            if not found:
                missing_keywords.append(job_keyword)
        
        # Calculate component scores
        match_score = len(matches) / len(job_keywords) if job_keywords else 0
        
        # Calculate position score average
        position_score = sum(match['position_score'] for match in matches) / len(matches) if matches else 0
        
        # Calculate sections organization score
        section_score = self.analyze_section_structure(resume_text)
        
        # Calculate formatting score
        formatting_analysis = self.analyze_formatting(resume_text)
        formatting_score = formatting_analysis['score']
        
        # Calculate keyword density score
        density_score = self.calculate_keyword_density(resume_text, job_keywords)
        
        # Calculate weighted total score
        total_score = (
            match_score * self.SCORING_WEIGHTS['keyword_match'] +
            position_score * self.SCORING_WEIGHTS['keyword_position'] +
            density_score * self.SCORING_WEIGHTS['keyword_density'] +
            section_score * self.SCORING_WEIGHTS['section_organization'] +
            formatting_score * self.SCORING_WEIGHTS['formatting']
        ) * 100
        
        return {
            "exact_match_score": round(match_score * 100, 1),
            "semantic_match_score": round(position_score * 100, 1),
            "combined_score": round(total_score, 1),
            "matched_keywords": [m['keyword'] for m in matches],
            "missing_keywords": missing_keywords,
            "keyword_matches": matches,
            "component_scores": {
                "keyword_match": match_score,
                "keyword_position": position_score,
                "keyword_density": density_score,
                "section_organization": section_score,
                "formatting": formatting_score
            },
            "format_issues": formatting_analysis['issues']
        }

    def identify_resume_sections(self, text):
        """Identify common resume sections."""
        # First try the new more robust section detection
        detected_sections = self.detect_sections(text)
        if detected_sections:
            sections_dict = {}
            for section in detected_sections:
                section_name = section['name'].lower()
                # Map to standard section names
                for std_name in ['contact', 'summary', 'experience', 'education', 
                                 'skills', 'projects', 'certifications', 'languages']:
                    if std_name in section_name:
                        sections_dict[std_name] = {'content': section['content']}
                        break
            return sections_dict
        
        # Fall back to original approach if the new one doesn't produce results
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
        # Match score with enhanced algorithm
        match_results = self.calculate_match_score(resume_text, job_description, True)
        
        # Section identification
        sections = self.identify_resume_sections(resume_text)
        
        # Extract contact info
        from utils.parser import extract_contact_info
        contact_info = extract_contact_info(resume_text)
        
        # Action verbs check
        action_verb_results = self.check_action_verbs(resume_text)
        
        # Quantifiable results check
        quantifiable_results = self.check_quantifiable_results(resume_text)
        
        # Get detailed feedback
        feedback = self.generate_detailed_feedback(
            resume_text, 
            job_description, 
            match_results, 
            sections
        )
        
        return {
            "contact_info": contact_info,
            "match_results": match_results,
            "sections": sections,
            "format_issues": match_results["format_issues"],
            "action_verbs": action_verb_results,
            "quantifiable_results": quantifiable_results,
            "feedback": feedback
        }
    
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
                if bullet.lower().startswith(f"• {verb}") or bullet.lower().startswith(f"- {verb}"):
                    has_action_verb = True
                    break
            if has_action_verb:
                bullets_with_action_verbs += 1
                
        bullet_ratio = bullets_with_action_verbs / len(bullet_points) if bullet_points else 0
                
        return {
            "has_action_verbs": len(found_verbs) >= 5 and bullet_ratio >= 0.5,
            "count": len(found_verbs),
            "found": found_verbs,
            "missing": [verb for verb in common_action_verbs if verb not in found_verbs][:5],
            "bullet_ratio": bullet_ratio
        }
    
    def check_quantifiable_results(self, text):
        """Check for quantifiable results in experience section."""
        # Check for numbers and percentages in bullet points
        sections = self.identify_resume_sections(text)
        
        if "experience" not in sections:
            return {"has_quantifiable_results": False}
            
        experience_text = sections["experience"].get("content", "")
        
        # Look for bullet points with numbers, percentages, or dollar amounts
        bullet_points = [line.strip() for line in experience_text.split('\n') 
                        if line.strip().startswith('•') or line.strip().startswith('-')]
        
        quantified_bullets = 0
        
        for bullet in bullet_points:
            # Check for percentages, numbers with units, or dollar amounts
            if (re.search(r'\d+%', bullet) or 
                re.search(r'\$\d+', bullet) or
                re.search(r'\d+ (users|customers|clients|people|employees|teams)', bullet) or
                re.search(r'(increased|decreased|reduced|improved|grew|raised) .* by \d+', bullet, re.IGNORECASE)):
                quantified_bullets += 1
                
        ratio = quantified_bullets / len(bullet_points) if bullet_points else 0
        
        return {
            "has_quantifiable_results": ratio >= 0.3,  # At least 30% of bullets should have quantifiable results
            "ratio": ratio,
            "quantified_count": quantified_bullets,
            "total_bullets": len(bullet_points)
        }
    
    def generate_detailed_feedback(self, resume_text, job_description, match_results, sections):
        """Generate detailed, actionable feedback based on analysis results."""
        feedback = {
            "summary": "",
            "keyword_feedback": [],
            "section_feedback": [],
            "formatting_feedback": [],
            "prioritized_actions": []
        }
        
        # Generate keyword feedback
        missing_keywords = match_results["missing_keywords"][:10]  # Focus on top 10 missing keywords
        for keyword in missing_keywords:
            feedback["keyword_feedback"].append({
                "type": "missing",
                "keyword": keyword,
                "suggestion": f"Add the keyword '{keyword}' to your resume. This is an important term in the job description."
            })
        
        # Keyword position feedback
        low_position_matches = [match for match in match_results.get("keyword_matches", []) 
                               if match.get("position_score", 0) < 0.5]
        for match in low_position_matches[:3]:  # Focus on top 3 issues
            feedback["keyword_feedback"].append({
                "type": "reposition",
                "keyword": match["keyword"],
                "suggestion": f"Move '{match['keyword']}' to a more prominent position in your resume (earlier sections)."
            })
        
        # Section feedback
        if len(sections) < 4:
            feedback["section_feedback"].append({
                "type": "missing-sections",
                "suggestion": "Your resume is missing essential sections. Ensure you have Summary, Experience, Education, and Skills sections."
            })
        
        if match_results["component_scores"]["section_organization"] < 0.7:
            feedback["section_feedback"].append({
                "type": "section-order",
                "suggestion": "Rearrange your sections to follow the recommended order: Summary, Experience, Education, Skills, followed by additional sections."
            })
        
        # Formatting feedback
        for issue in match_results["format_issues"]:
            feedback["section_feedback"].append({
                "type": "formatting",
                "suggestion": issue
            })
        
        # Prioritize actions based on impact
        all_feedback = (
            [{"priority": "High", "action": item["suggestion"]} for item in feedback["keyword_feedback"] if item["type"] == "missing"][:5] +
            [{"priority": "High", "action": item["suggestion"]} for item in feedback["section_feedback"] if item["type"] == "missing-sections"] +
            [{"priority": "Medium", "action": item["suggestion"]} for item in feedback["formatting_feedback"]] +
            [{"priority": "Medium", "action": item["suggestion"]} for item in feedback["keyword_feedback"] if item["type"] == "reposition"] +
            [{"priority": "Medium", "action": item["suggestion"]} for item in feedback["section_feedback"] if item["type"] == "section-order"] +
            [{"priority": "Low", "action": item["suggestion"]} for item in feedback["keyword_feedback"] if item["type"] == "missing"][5:]
        )
        
        feedback["prioritized_actions"] = all_feedback
        
        # Generate summary
        score = match_results["combined_score"]
        if score >= 85:
            feedback["summary"] = "Your resume is well-optimized for ATS systems. Make minor improvements for even better results."
        elif score >= 70:
            feedback["summary"] = "Your resume is moderately ATS-friendly. Address the top issues to improve your chances."
        else:
            feedback["summary"] = "Your resume needs significant optimization for ATS systems. Focus on the high-priority actions."
            
        return feedback 