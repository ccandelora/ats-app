import re
from .analyzer import ResumeAnalyzer
from docx import Document
import docx

class ResumeOptimizer:
    def __init__(self):
        self.analyzer = ResumeAnalyzer()
        
    def optimize_resume(self, resume_text, job_description):
        """Generate an optimized resume based on job description."""
        # Analyze the original resume
        analysis = self.analyzer.analyze_resume(resume_text, job_description)
        
        # Extract missing keywords from job description
        missing_keywords = analysis["match_results"]["missing_keywords"]
        
        # Identify resume sections
        sections = analysis["sections"]
        
        # Initialize optimized resume with original content
        optimized_sections = {}
        for section_name, section_data in sections.items():
            if "content" in section_data:
                optimized_sections[section_name] = section_data["content"]
        
        # If we couldn't identify sections properly, use the whole text
        if not optimized_sections:
            # Just add missing keywords to the summary/profile section
            lines = resume_text.split('\n')
            # Look for a summary/profile section
            summary_index = -1
            for i, line in enumerate(lines):
                if re.search(r'(?i)(summary|profile|objective)', line):
                    summary_index = i
                    break
            
            if summary_index >= 0 and summary_index + 1 < len(lines):
                # Add optimized content after the summary heading
                summary_text = lines[summary_index + 1]
                enriched_summary = self._enrich_text(summary_text, missing_keywords)
                lines[summary_index + 1] = enriched_summary
                return '\n'.join(lines)
            else:
                # No identifiable sections, return with minimal changes
                return resume_text
                
        # Enhance summary/profile section
        if "summary" in optimized_sections:
            optimized_sections["summary"] = self._enrich_text(
                optimized_sections["summary"], 
                missing_keywords[:5]  # Add top 5 missing keywords to summary
            )
            
        # Enhance skills section
        if "skills" in optimized_sections:
            # Add missing keywords as skills
            skills_text = optimized_sections["skills"]
            skills_lines = skills_text.split('\n')
            
            # Find where to insert new skills
            if len(skills_lines) > 0:
                new_skills = []
                added_keywords = set()
                
                for keyword in missing_keywords:
                    # Avoid duplicates and only add relevant skills
                    if keyword not in skills_text.lower() and keyword not in added_keywords and len(keyword) > 3:
                        new_skills.append(f"• {keyword.title()}")
                        added_keywords.add(keyword)
                
                if new_skills:
                    # Find a good insertion point (after existing bullet points)
                    insertion_point = 0
                    for i, line in enumerate(skills_lines):
                        if line.strip().startswith('•') or line.strip().startswith('-'):
                            insertion_point = i + 1
                    
                    # Insert new skills
                    skills_lines = skills_lines[:insertion_point] + new_skills + skills_lines[insertion_point:]
                    optimized_sections["skills"] = '\n'.join(skills_lines)
        
        # Enhance experience section to include relevant keywords
        if "experience" in optimized_sections:
            experience_text = optimized_sections["experience"]
            # Look for places to naturally integrate keywords
            experience_lines = experience_text.split('\n')
            
            # Process bullet points to include keywords
            for i in range(len(experience_lines)):
                line = experience_lines[i].strip()
                if (line.startswith('•') or line.startswith('-')) and len(line) > 10:
                    # This is a bullet point
                    experience_lines[i] = self._enhance_bullet_point(line, missing_keywords)
            
            optimized_sections["experience"] = '\n'.join(experience_lines)
        
        # Add action verbs if needed
        if "experience" in optimized_sections and not analysis["action_verbs"]["has_action_verbs"]:
            optimized_sections["experience"] = self._add_action_verbs(
                optimized_sections["experience"], 
                analysis["action_verbs"]["missing"]
            )
        
        # Add quantifiable results if needed
        if "experience" in optimized_sections and not analysis["quantifiable_results"]["has_quantifiable_results"]:
            optimized_sections["experience"] = self._suggest_quantifiable_results(
                optimized_sections["experience"]
            )
        
        # Reconstruct the resume
        reconstructed_resume = []
        for section_name, section_content in sections.items():
            if "start" in section_content and "end" in section_content:
                # Get the original section header
                original_lines = resume_text.split('\n')
                header_line = original_lines[section_content["start"]]
                reconstructed_resume.append(header_line)
                
                # Add optimized content if available, otherwise original content
                if section_name in optimized_sections:
                    reconstructed_resume.append(optimized_sections[section_name])
                else:
                    section_lines = original_lines[section_content["start"]+1:section_content["end"]]
                    reconstructed_resume.append('\n'.join(section_lines))
                
                reconstructed_resume.append('')  # Add blank line between sections
        
        return '\n'.join(reconstructed_resume)
    
    def _enrich_text(self, text, keywords, max_keywords=5):
        """Add keywords to text where appropriate."""
        if not text or not keywords:
            return text
            
        # Simple approach: add keywords to the end of the text
        keywords_to_add = [k for k in keywords[:max_keywords] if k.lower() not in text.lower()]
        
        if not keywords_to_add:
            return text
            
        enriched_text = text.rstrip()
        
        # Check if the last sentence ends with proper punctuation
        if not enriched_text.endswith('.') and not enriched_text.endswith('!') and not enriched_text.endswith('?'):
            enriched_text += '.'
            
        # Add keywords in a natural way
        keyword_phrase = ", ".join(k.title() for k in keywords_to_add)
        enriched_text += f" Experienced in {keyword_phrase}."
            
        return enriched_text
    
    def _enhance_bullet_point(self, bullet_point, keywords, max_keywords=2):
        """Enhance a bullet point with relevant keywords."""
        # Only process if there are keywords to add
        if not keywords:
            return bullet_point
            
        # Find keywords that aren't already in the bullet point
        keywords_to_add = []
        for keyword in keywords:
            if keyword.lower() not in bullet_point.lower() and len(keywords_to_add) < max_keywords:
                keywords_to_add.append(keyword)
                
        if not keywords_to_add:
            return bullet_point
            
        # Add keywords to the bullet point in a natural way
        enhanced_bullet = bullet_point.rstrip()
        
        # Check if the bullet point ends with proper punctuation
        if not enhanced_bullet.endswith('.') and not enhanced_bullet.endswith('!') and not enhanced_bullet.endswith('?'):
            enhanced_bullet += '.'
            
        # Add a phrase with the keyword
        keyword_phrase = " and ".join(k.lower() for k in keywords_to_add)
        
        # Choose a connecting phrase based on the bullet content
        if "developed" in enhanced_bullet.lower() or "created" in enhanced_bullet.lower():
            enhanced_bullet = enhanced_bullet[:-1] + f", utilizing {keyword_phrase}."
        elif "managed" in enhanced_bullet.lower() or "led" in enhanced_bullet.lower():
            enhanced_bullet = enhanced_bullet[:-1] + f", with focus on {keyword_phrase}."
        else:
            enhanced_bullet = enhanced_bullet[:-1] + f", incorporating {keyword_phrase}."
            
        return enhanced_bullet
    
    def _add_action_verbs(self, experience_text, missing_verbs):
        """Add action verbs to experience bullet points."""
        if not missing_verbs:
            return experience_text
            
        lines = experience_text.split('\n')
        verb_index = 0
        
        for i in range(len(lines)):
            line = lines[i].strip()
            # Only modify bullet points that don't start with action verbs
            if (line.startswith('•') or line.startswith('-')) and not any(line.lower().startswith(f"• {v}") or line.lower().startswith(f"- {v}") for v in missing_verbs):
                # Strip the bullet and leading whitespace
                content = line[1:].strip()
                
                # Add the action verb at the beginning
                verb = missing_verbs[verb_index % len(missing_verbs)].title()
                lines[i] = f"{line[0]} {verb} {content}"
                verb_index += 1
        
        return '\n'.join(lines)
    
    def _suggest_quantifiable_results(self, experience_text):
        """Suggest adding quantifiable results to experience bullet points."""
        lines = experience_text.split('\n')
        modified = False
        
        for i in range(len(lines)):
            line = lines[i].strip()
            # Look for bullet points without numbers
            if (line.startswith('•') or line.startswith('-')) and not re.search(r'\d+', line) and len(line) > 15:
                # Add a placeholder for quantifiable result
                lines[i] = f"{line} [Add specific metrics e.g., 'resulting in 20% improvement']"
                modified = True
                break  # Only modify one line as an example
        
        return '\n'.join(lines)
        
    def generate_docx(self, resume_text):
        """Generate a professionally formatted DOCX document from the optimized resume text."""
        doc = Document()
        
        # Set document styles
        styles = doc.styles
        style = styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = docx.shared.Pt(11)
        
        # Extract contact info from the resume for the header
        from utils.parser import extract_contact_info
        contact_info = extract_contact_info(resume_text)
        
        # Create professional header with contact information
        if contact_info['name']:
            # Add name as large, bold header
            name_paragraph = doc.add_paragraph()
            name_run = name_paragraph.add_run(contact_info['name'])
            name_run.bold = True
            name_run.font.size = docx.shared.Pt(16)
            name_paragraph.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
            
            # Add contact details in centered row
            contact_paragraph = doc.add_paragraph()
            contact_paragraph.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
            
            contact_details = []
            if contact_info['email']:
                contact_details.append(contact_info['email'])
            if contact_info['phone']:
                contact_details.append(contact_info['phone'])
            if contact_info['linkedin']:
                contact_details.append(contact_info['linkedin'])
                
            contact_text = " | ".join(contact_details)
            contact_paragraph.add_run(contact_text)
            
            # Add line separator
            doc.add_paragraph().add_run('_' * 50)
        else:
            # No contact info found, just add the title
            doc.add_heading('Optimized Resume', 0)
        
        # Parse the resume into sections
        sections = {}
        current_section = None
        
        for line in resume_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a section header
            if re.search(r'(?i)^(summary|profile|objective|experience|work|employment|education|skills|projects|certifications|languages).*$', line):
                current_section = line
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)
        
        # Add each section to the document
        for section_title, section_content in sections.items():
            # Add section title
            heading = doc.add_heading(section_title, level=1)
            heading.style.font.color.rgb = docx.shared.RGBColor(0, 0, 0)
            heading.style.font.bold = True
            
            # Process section content based on format
            for line in section_content:
                if line.startswith('•') or line.startswith('-'):
                    # Bullet point
                    p = doc.add_paragraph(style='List Bullet')
                    p.add_run(line[1:].strip())
                elif re.match(r'\d+\.', line):
                    # Numbered list
                    p = doc.add_paragraph(style='List Number')
                    p.add_run(re.sub(r'\d+\.\s*', '', line))
                else:
                    # Regular paragraph
                    p = doc.add_paragraph()
                    p.add_run(line)
            
            # Add spacing between sections
            doc.add_paragraph()
        
        # Set document properties
        core_properties = doc.core_properties
        core_properties.title = f"Optimized Resume - {contact_info['name'] or 'Candidate'}"
        core_properties.subject = "Resume"
        core_properties.author = contact_info['name'] or "Candidate"
        core_properties.language = "en-US"
        
        return doc 