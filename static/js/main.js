// Main JavaScript file for ATS Resume Checker

document.addEventListener('DOMContentLoaded', function() {
    // File upload validation
    const resumeInput = document.getElementById('resume');
    if (resumeInput) {
        resumeInput.addEventListener('change', function() {
            const file = this.files[0];
            const fileSize = file.size / 1024 / 1024; // in MB
            const fileType = file.type;
            const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
            
            // Check file type
            if (!validTypes.includes(fileType)) {
                alert('Please select a PDF or DOCX file.');
                this.value = ''; // Clear the file input
                return;
            }
            
            // Check file size
            if (fileSize > 16) {
                alert('File size exceeds 16MB. Please select a smaller file.');
                this.value = ''; // Clear the file input
                return;
            }
        });
    }
    
    // Job description character count
    const jobDescInput = document.getElementById('job_description');
    if (jobDescInput) {
        jobDescInput.addEventListener('input', function() {
            const currentLength = this.value.length;
            const minRecommended = 100;
            
            if (currentLength > 0 && currentLength < minRecommended) {
                this.classList.add('is-invalid');
            } else {
                this.classList.remove('is-invalid');
            }
        });
    }
}); 

// Function to update contact information
function updateContactInfo() {
    // Get values from form
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const phone = document.getElementById('phone').value;
    const linkedin = document.getElementById('linkedin').value;
    
    // Update the displayed information
    const contactItems = document.querySelectorAll('.list-unstyled li');
    contactItems.forEach(item => {
        if (item.textContent.includes('Name:')) {
            item.innerHTML = `<strong>Name:</strong> ${name || "Not detected"}`;
        } else if (item.textContent.includes('Email:')) {
            item.innerHTML = `<strong>Email:</strong> ${email || "Not detected"}`;
        } else if (item.textContent.includes('Phone:')) {
            item.innerHTML = `<strong>Phone:</strong> ${phone || "Not detected"}`;
        } else if (item.textContent.includes('LinkedIn:')) {
            item.innerHTML = `<strong>LinkedIn:</strong> ${linkedin || "Not detected"}`;
        }
    });
    
    // Store the updated contact info in a hidden input for form submission
    let contactInfoInput = document.getElementById('contact_info_data');
    if (!contactInfoInput) {
        contactInfoInput = document.createElement('input');
        contactInfoInput.type = 'hidden';
        contactInfoInput.id = 'contact_info_data';
        contactInfoInput.name = 'contact_info_data';
        const optimizeForm = document.querySelector('form[action="/optimize"]');
        if (optimizeForm) {
            optimizeForm.appendChild(contactInfoInput);
        }
    }
    
    contactInfoInput.value = JSON.stringify({
        name: name,
        email: email,
        phone: phone,
        linkedin: linkedin
    });
    
    // Close the form
    const contactForm = document.getElementById('contactForm');
    if (contactForm && typeof bootstrap !== 'undefined') {
        const bsCollapse = new bootstrap.Collapse(contactForm);
        bsCollapse.hide();
    } else {
        // Fallback if Bootstrap JS is not available
        document.getElementById('contactForm').classList.remove('show');
    }
    
    // Show success message
    alert('Contact information updated successfully!');
} 