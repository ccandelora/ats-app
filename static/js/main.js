/**
 * Main JavaScript for ATS Resume Checker
 */

// Wait for DOM content to load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    initTooltips();
    
    // Initialize file input enhancements
    enhanceFileInputs();
    
    // Initialize industry selection behavior
    initIndustrySelection();
    
    // Initialize contact info form behavior
    initContactInfoForm();
    
    // Initialize form submission and progress tracking
    initProgressTracking();
});

/**
 * Initialize progress tracking during form submission
 */
function initProgressTracking() {
    const analyzeForm = document.querySelector('form[action*="analyze_resume"]');
    const loadingOverlay = document.getElementById('loading-overlay');
    const progressBar = document.getElementById('analysis-progress');
    const loadingStageText = document.getElementById('loading-stage-text');
    
    if (!analyzeForm || !loadingOverlay || !progressBar || !loadingStageText) return;
    
    analyzeForm.addEventListener('submit', function(e) {
        // Show loading overlay
        loadingOverlay.classList.remove('d-none');
        
        // Start simulated progress updates
        simulateAnalysisProgress();
    });
    
    function simulateAnalysisProgress() {
        const stages = [
            { progress: 10, text: 'Parsing document...' },
            { progress: 30, text: 'Extracting content...' },
            { progress: 50, text: 'Analyzing keywords...' },
            { progress: 65, text: 'Calculating section match score...' },
            { progress: 75, text: 'Evaluating format compatibility...' },
            { progress: 85, text: 'Generating recommendations...' },
            { progress: 95, text: 'Finalizing analysis...' }
        ];
        
        let currentStage = 0;
        
        // Update progress bar every 800ms to simulate analysis stages
        const progressInterval = setInterval(function() {
            if (currentStage < stages.length) {
                const stage = stages[currentStage];
                updateProgress(stage.progress, stage.text);
                currentStage++;
            } else {
                // Complete the progress bar when all stages are done
                updateProgress(100, 'Analysis complete!');
                clearInterval(progressInterval);
            }
        }, 800);
        
        function updateProgress(percent, text) {
            progressBar.style.width = percent + '%';
            progressBar.setAttribute('aria-valuenow', percent);
            progressBar.textContent = percent + '%';
            loadingStageText.textContent = text;
            
            // If we've reached 100%, add a small delay before form submission completes
            // This ensures users see "Analysis complete!" message
            if (percent === 100) {
                // We don't need to do anything here since the form is already submitting
            }
        }
    }
}

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Enhance file input fields with better UI
 */
function enhanceFileInputs() {
    // Get all custom file inputs
    const customFileInputs = document.querySelectorAll('.custom-file-input');
    
    customFileInputs.forEach(input => {
        // Add event listener to update label text with filename
        input.addEventListener('change', function() {
            // Find the associated label
            const label = this.nextElementSibling;
            if (label) {
                const fileName = this.files[0]?.name || 'No file chosen';
                label.textContent = fileName;
            }
            
            // Check if file is valid
            validateFileInput(this);
        });
    });
}

/**
 * Validate file input (file type checking)
 * @param {HTMLInputElement} input - File input element
 */
function validateFileInput(input) {
    const allowedExtensions = ['pdf', 'docx', 'doc', 'txt', 'rtf'];
    const fileName = input.files[0]?.name || '';
    
    if (fileName) {
        const fileExtension = fileName.split('.').pop().toLowerCase();
        const isValid = allowedExtensions.includes(fileExtension);
        
        // Get form submit button
        const form = input.closest('form');
        const submitBtn = form?.querySelector('button[type="submit"]');
        
        if (submitBtn) {
            submitBtn.disabled = !isValid;
        }
        
        // Show validation message
        const validationMessage = input.parentElement.querySelector('.invalid-feedback');
        if (validationMessage) {
            validationMessage.style.display = isValid ? 'none' : 'block';
        }
    }
}

/**
 * Initialize industry selection behavior
 */
function initIndustrySelection() {
    const industrySelect = document.getElementById('industry');
    if (industrySelect) {
        // Add event listener for industry selection change
        industrySelect.addEventListener('change', function() {
            const selectedIndustry = this.value;
            
            // Get industry description element
            const industryDesc = document.getElementById('industry-description');
            if (industryDesc) {
                // Update description based on selection
                switch (selectedIndustry) {
                    case 'tech':
                        industryDesc.innerHTML = '<p><strong>Technology Industry:</strong> Emphasizes technical skills, programming languages, methodologies, certifications, and specific technical achievements.</p>';
                        break;
                    case 'finance':
                        industryDesc.innerHTML = '<p><strong>Finance Industry:</strong> Focuses on analytical skills, financial certifications, compliance knowledge, and quantifiable financial achievements.</p>';
                        break;
                    case 'healthcare':
                        industryDesc.innerHTML = '<p><strong>Healthcare Industry:</strong> Prioritizes medical terminology, certifications, patient care metrics, and compliance with healthcare regulations.</p>';
                        break;
                    case 'marketing':
                        industryDesc.innerHTML = '<p><strong>Marketing Industry:</strong> Highlights campaign metrics, social media expertise, content creation, and measurable marketing outcomes.</p>';
                        break;
                    case 'education':
                        industryDesc.innerHTML = '<p><strong>Education Industry:</strong> Emphasizes teaching methodologies, curriculum development, student outcomes, and educational certifications.</p>';
                        break;
                    case 'sales':
                        industryDesc.innerHTML = '<p><strong>Sales Industry:</strong> Focuses on revenue generation, client acquisition, negotiation skills, and quantifiable sales achievements.</p>';
                        break;
                    default:
                        industryDesc.innerHTML = '<p><strong>General Analysis:</strong> Balanced scoring across all categories without industry-specific emphasis.</p>';
                }
            }
        });
        
        // Trigger change event to set initial description
        const event = new Event('change');
        industrySelect.dispatchEvent(event);
    }
}

/**
 * Initialize contact info form behavior
 */
function initContactInfoForm() {
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        // Get all input fields in the form
        const inputs = contactForm.querySelectorAll('input');
        
        // Add input event listeners to validate in real-time
        inputs.forEach(input => {
            input.addEventListener('input', function() {
                validateContactInput(this);
            });
        });
    }
}

/**
 * Validate contact info input field
 * @param {HTMLInputElement} input - Input field to validate
 */
function validateContactInput(input) {
    const value = input.value.trim();
    const inputType = input.id;
    let isValid = true;
    let errorMessage = '';
    
    // Validate based on input type
    switch (inputType) {
        case 'email':
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            isValid = value === '' || emailRegex.test(value);
            errorMessage = 'Please enter a valid email address';
            break;
            
        case 'phone':
            const phoneRegex = /^[\d\+\-\(\) ]{7,20}$/;
            isValid = value === '' || phoneRegex.test(value);
            errorMessage = 'Please enter a valid phone number';
            break;
            
        case 'linkedin':
            const linkedinRegex = /^(https?:\/\/)?(www\.)?linkedin\.com\/in\/[a-zA-Z0-9_-]+\/?$/;
            isValid = value === '' || linkedinRegex.test(value);
            errorMessage = 'Please enter a valid LinkedIn profile URL';
            break;
    }
    
    // Update UI based on validation result
    if (!isValid) {
        input.classList.add('is-invalid');
        
        // Find or create error message element
        let feedback = input.nextElementSibling;
        if (!feedback || !feedback.classList.contains('invalid-feedback')) {
            feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            input.parentNode.insertBefore(feedback, input.nextSibling);
        }
        
        feedback.textContent = errorMessage;
    } else {
        input.classList.remove('is-invalid');
    }
}

/**
 * Update contact information (used in the results page)
 */
function updateContactInfo() {
    const form = document.getElementById('manualContactForm');
    if (form) {
        // Get all inputs
        const name = form.querySelector('#name').value;
        const email = form.querySelector('#email').value;
        const phone = form.querySelector('#phone').value;
        const linkedin = form.querySelector('#linkedin').value;
        
        // Create contact info object
        const contactInfo = {
            name: name,
            email: email,
            phone: phone,
            linkedin: linkedin
        };
        
        // Set hidden input value for the main form
        const hiddenInput = document.getElementById('contact_info_data');
        if (hiddenInput) {
            hiddenInput.value = JSON.stringify(contactInfo);
        }
        
        // Submit the main form
        const optimizeForm = document.querySelector('form[action*="optimize_resume"]');
        if (optimizeForm) {
            optimizeForm.submit();
        }
    }
} 