import copy

class IndustryScorer:
    # Define industry-specific keyword importance mappings
    INDUSTRY_KEYWORD_WEIGHTS = {
        'tech': {
            'programming': 1.5,
            'development': 1.3,
            'software': 1.4,
            'agile': 1.2,
            'api': 1.3,
            'cloud': 1.4,
            'devops': 1.3,
            'frontend': 1.2,
            'backend': 1.2,
            'database': 1.3,
            'python': 1.4,
            'javascript': 1.4,
            'java': 1.3,
            'algorithms': 1.3,
            'security': 1.2
        },
        'finance': {
            'analysis': 1.4,
            'investment': 1.5,
            'portfolio': 1.3,
            'regulatory': 1.4,
            'compliance': 1.5,
            'risk': 1.4,
            'banking': 1.3,
            'financial': 1.5,
            'audit': 1.3,
            'accounting': 1.3,
            'revenue': 1.2,
            'budget': 1.2,
            'forecast': 1.3,
            'capital': 1.4,
            'trading': 1.3
        },
        'healthcare': {
            'patient': 1.5,
            'clinical': 1.4,
            'medical': 1.3,
            'healthcare': 1.4,
            'nursing': 1.3,
            'physician': 1.4,
            'therapy': 1.3,
            'diagnosis': 1.4,
            'treatment': 1.3,
            'hospital': 1.2,
            'care': 1.3,
            'health': 1.3,
            'records': 1.2,
            'compliance': 1.4,
            'insurance': 1.2
        },
        'marketing': {
            'marketing': 1.5,
            'campaign': 1.4,
            'social': 1.4,
            'content': 1.4,
            'brand': 1.5,
            'seo': 1.3,
            'analytics': 1.4,
            'digital': 1.4,
            'audience': 1.3,
            'strategy': 1.4,
            'advertising': 1.3,
            'media': 1.3,
            'engagement': 1.2,
            'conversion': 1.3,
            'roi': 1.3
        },
        'education': {
            'teaching': 1.5,
            'curriculum': 1.4,
            'education': 1.5,
            'learning': 1.4,
            'instruction': 1.3,
            'assessment': 1.4,
            'classroom': 1.3,
            'pedagogy': 1.4,
            'students': 1.3,
            'faculty': 1.3,
            'academic': 1.4,
            'research': 1.3,
            'development': 1.2,
            'training': 1.3,
            'program': 1.2
        },
        'sales': {
            'sales': 1.5,
            'revenue': 1.4,
            'customer': 1.4,
            'account': 1.3,
            'business': 1.3,
            'relationship': 1.2,
            'pipeline': 1.3,
            'negotiation': 1.4,
            'closing': 1.4,
            'target': 1.3,
            'quota': 1.5,
            'crm': 1.3,
            'client': 1.4,
            'growth': 1.3,
            'prospecting': 1.4
        }
    }
    
    def apply_industry_scoring(self, score_data, job_description, industry):
        """Apply industry-specific scoring adjustments"""
        if industry not in self.INDUSTRY_KEYWORD_WEIGHTS:
            return score_data  # No industry-specific scoring available
        
        # Create a deep copy to avoid modifying the original
        adjusted_data = copy.deepcopy(score_data)
        
        industry_keywords = self.INDUSTRY_KEYWORD_WEIGHTS[industry]
        
        # Adjust match scores based on industry importance
        if 'keyword_matches' in adjusted_data:
            adjusted_matches = []
            for match in adjusted_data['keyword_matches']:
                weight = industry_keywords.get(match['keyword'].lower(), 1.0)
                
                adjusted_match = match.copy()
                adjusted_match['score'] = min(1.0, match['score'] * weight)
                adjusted_match['industry_weight'] = weight
                adjusted_matches.append(adjusted_match)
            
            adjusted_data['keyword_matches'] = adjusted_matches
        
        # Recalculate keyword match score
        if 'component_scores' in adjusted_data and 'keyword_matches' in adjusted_data:
            matches = adjusted_data['keyword_matches']
            match_score = sum(match['score'] for match in matches) / len(matches) if matches else 0
            adjusted_data['component_scores']['keyword_match'] = match_score
        
        # Recalculate total score
        if 'component_scores' in adjusted_data:
            from models.analyzer import ResumeAnalyzer
            analyzer = ResumeAnalyzer()
            total_score = (
                adjusted_data['component_scores']['keyword_match'] * analyzer.SCORING_WEIGHTS['keyword_match'] +
                adjusted_data['component_scores']['keyword_position'] * analyzer.SCORING_WEIGHTS['keyword_position'] +
                adjusted_data['component_scores']['keyword_density'] * analyzer.SCORING_WEIGHTS['keyword_density'] +
                adjusted_data['component_scores']['section_organization'] * analyzer.SCORING_WEIGHTS['section_organization'] +
                adjusted_data['component_scores']['formatting'] * analyzer.SCORING_WEIGHTS['formatting']
            ) * 100
            
            adjusted_data['combined_score'] = round(total_score, 1)
        
        # Add industry-specific feedback
        if 'feedback' in adjusted_data:
            adjusted_data['feedback']['industry_specific'] = self.generate_industry_feedback(industry, adjusted_data)
        
        return adjusted_data
    
    def generate_industry_feedback(self, industry, analysis_results):
        """Generate industry-specific feedback based on analysis results"""
        industry_feedback = []
        
        # Get top industry keywords that are missing
        missing_keywords = analysis_results.get('missing_keywords', [])
        industry_keywords = self.INDUSTRY_KEYWORD_WEIGHTS[industry]
        
        # Find missing industry-specific keywords
        important_missing = []
        for keyword in missing_keywords:
            if keyword.lower() in industry_keywords and industry_keywords[keyword.lower()] >= 1.3:
                important_missing.append({
                    'keyword': keyword,
                    'importance': industry_keywords[keyword.lower()]
                })
        
        # Sort by importance
        important_missing.sort(key=lambda x: x['importance'], reverse=True)
        
        # Generate feedback
        for item in important_missing[:3]:  # Focus on top 3
            industry_feedback.append({
                'type': 'industry_keyword',
                'keyword': item['keyword'],
                'suggestion': f"Add '{item['keyword']}' to your resume - this is a highly valued keyword in the {industry} industry."
            })
        
        # Add general industry advice
        if industry == 'tech':
            industry_feedback.append({
                'type': 'industry_advice',
                'suggestion': "Tech resumes should emphasize specific technologies, programming languages, and technical frameworks. Quantify project outcomes."
            })
        elif industry == 'finance':
            industry_feedback.append({
                'type': 'industry_advice',
                'suggestion': "Finance resumes should highlight regulatory knowledge, analytical skills, and specific financial tools or systems you've used."
            })
        elif industry == 'healthcare':
            industry_feedback.append({
                'type': 'industry_advice',
                'suggestion': "Healthcare resumes should emphasize patient care outcomes, compliance knowledge, and specific medical technologies or protocols."
            })
        elif industry == 'marketing':
            industry_feedback.append({
                'type': 'industry_advice',
                'suggestion': "Marketing resumes should showcase campaign metrics, growth numbers, and specific marketing tools or platforms you've mastered."
            })
        elif industry == 'education':
            industry_feedback.append({
                'type': 'industry_advice',
                'suggestion': "Education resumes should highlight teaching methods, curriculum development, and student outcome improvements or achievements."
            })
        elif industry == 'sales':
            industry_feedback.append({
                'type': 'industry_advice',
                'suggestion': "Sales resumes should emphasize revenue generation, quota achievement percentages, and specific sales methodologies you've used successfully."
            })
            
        return industry_feedback 