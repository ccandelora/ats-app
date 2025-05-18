/**
 * Resume Heatmap Visualization
 * This script creates a heatmap visualization of keyword matches in a resume.
 */

// Configuration
const heatmapConfig = {
    highMatchColor: 'rgba(40, 167, 69, 0.3)', // Green with transparency
    mediumMatchColor: 'rgba(255, 193, 7, 0.3)', // Yellow with transparency
    lowMatchColor: 'rgba(220, 53, 69, 0.2)', // Red with transparency
    noMatchColor: 'transparent',
    highMatchThreshold: 0.8,
    mediumMatchThreshold: 0.6
};

/**
 * Initialize the resume heatmap visualization
 * @param {string} resumeContainerId - ID of the container element for the resume
 * @param {string} matchesContainerId - ID of the container element for matched keywords
 * @param {Array} keywordMatches - Array of keyword match objects from the analysis
 */
function initResumeHeatmap(resumeContainerId, matchesContainerId, keywordMatches) {
    const resumeContainer = document.getElementById(resumeContainerId);
    const matchesContainer = document.getElementById(matchesContainerId);
    
    if (!resumeContainer || !matchesContainer || !keywordMatches) {
        console.error('Missing required elements for heatmap visualization');
        return;
    }

    // Get resume text and create a copy for highlighting
    const resumeText = resumeContainer.innerText;
    
    // Create a sorted list of keywords by length (longest first)
    // This prevents nested highlighting issues
    const sortedMatches = [...keywordMatches].sort((a, b) => {
        return b.keyword.length - a.keyword.length;
    });
    
    // Create the keyword match legend
    createMatchLegend(matchesContainer);
    
    // Apply highlighting to resume text
    applyKeywordHighlighting(resumeContainer, resumeText, sortedMatches);
    
    // Create the keyword match list
    createMatchList(matchesContainer, sortedMatches);
}

/**
 * Create a legend for the heatmap
 * @param {HTMLElement} container - Container element for the legend
 */
function createMatchLegend(container) {
    const legend = document.createElement('div');
    legend.className = 'heatmap-legend mb-3';
    
    legend.innerHTML = `
        <h5>Keyword Match Strength</h5>
        <div class="d-flex justify-content-between">
            <div class="legend-item">
                <span class="legend-color" style="background-color: ${heatmapConfig.highMatchColor.replace('0.3', '0.8')}"></span>
                <span class="legend-label">Strong match (>80%)</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: ${heatmapConfig.mediumMatchColor.replace('0.3', '0.8')}"></span>
                <span class="legend-label">Medium match (60-80%)</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: ${heatmapConfig.lowMatchColor.replace('0.2', '0.6')}"></span>
                <span class="legend-label">Weak match (<60%)</span>
            </div>
        </div>
    `;
    
    container.appendChild(legend);
}

/**
 * Apply keyword highlighting to resume text
 * @param {HTMLElement} container - Container element for the resume
 * @param {string} text - Original resume text
 * @param {Array} matches - Array of keyword match objects
 */
function applyKeywordHighlighting(container, text, matches) {
    // Create a DOM parser to safely create HTML
    const parser = new DOMParser();
    
    // Create a temporary div with the text content
    let tempDiv = document.createElement('div');
    tempDiv.innerText = text;
    let htmlContent = tempDiv.innerHTML;
    
    // Replace each keyword with a highlighted version
    for (const match of matches) {
        const keyword = match.keyword;
        const matchType = match.match_type || 'direct';
        const score = match.score || 1.0;
        
        // Determine highlight color based on score
        let highlightColor;
        if (score >= heatmapConfig.highMatchThreshold) {
            highlightColor = heatmapConfig.highMatchColor;
        } else if (score >= heatmapConfig.mediumMatchThreshold) {
            highlightColor = heatmapConfig.mediumMatchColor;
        } else {
            highlightColor = heatmapConfig.lowMatchColor;
        }
        
        // Create regex to find the keyword (word boundary sensitive)
        const regex = new RegExp(`\\b${escapeRegExp(keyword)}\\b`, 'gi');
        
        // Replace with highlighted version
        htmlContent = htmlContent.replace(regex, match => {
            return `<span class="keyword-highlight" 
                     style="background-color: ${highlightColor}; padding: 2px; border-radius: 2px;" 
                     title="Matched keyword: ${keyword} (${Math.round(score * 100)}% match, ${matchType})"
                     data-keyword="${keyword}">
                      ${match}
                    </span>`;
        });
    }
    
    // Insert the highlighted content
    container.innerHTML = `<div class="resume-highlighted-content">${htmlContent}</div>`;
    
    // Add event listener for keyword hover effects
    const highlights = container.querySelectorAll('.keyword-highlight');
    highlights.forEach(highlight => {
        highlight.addEventListener('mouseenter', function() {
            const keyword = this.dataset.keyword;
            highlightAllInstances(keyword);
        });
        
        highlight.addEventListener('mouseleave', function() {
            const keyword = this.dataset.keyword;
            resetHighlighting(keyword);
        });
    });
}

/**
 * Create a list of matched keywords
 * @param {HTMLElement} container - Container element for the list
 * @param {Array} matches - Array of keyword match objects
 */
function createMatchList(container, matches) {
    const matchList = document.createElement('div');
    matchList.className = 'matched-keywords-list mt-4';
    
    // Group matches by strength
    const strongMatches = matches.filter(m => (m.score || 1.0) >= heatmapConfig.highMatchThreshold);
    const mediumMatches = matches.filter(m => {
        const score = m.score || 1.0;
        return score >= heatmapConfig.mediumMatchThreshold && score < heatmapConfig.highMatchThreshold;
    });
    const weakMatches = matches.filter(m => (m.score || 1.0) < heatmapConfig.mediumMatchThreshold);
    
    matchList.innerHTML = `
        <h5>Matched Keywords (${matches.length})</h5>
        <div class="row">
            <div class="col-md-4">
                <h6>Strong Matches (${strongMatches.length})</h6>
                <div class="keyword-group strong-matches">
                    ${createKeywordTags(strongMatches, 'success')}
                </div>
            </div>
            <div class="col-md-4">
                <h6>Medium Matches (${mediumMatches.length})</h6>
                <div class="keyword-group medium-matches">
                    ${createKeywordTags(mediumMatches, 'warning')}
                </div>
            </div>
            <div class="col-md-4">
                <h6>Weak Matches (${weakMatches.length})</h6>
                <div class="keyword-group weak-matches">
                    ${createKeywordTags(weakMatches, 'danger')}
                </div>
            </div>
        </div>
    `;
    
    container.appendChild(matchList);
    
    // Add event listeners to keyword tags
    const keywordTags = matchList.querySelectorAll('.keyword-tag');
    keywordTags.forEach(tag => {
        tag.addEventListener('mouseenter', function() {
            const keyword = this.dataset.keyword;
            highlightAllInstances(keyword);
        });
        
        tag.addEventListener('mouseleave', function() {
            const keyword = this.dataset.keyword;
            resetHighlighting(keyword);
        });
        
        tag.addEventListener('click', function() {
            const keyword = this.dataset.keyword;
            scrollToFirstInstance(keyword);
        });
    });
}

/**
 * Create HTML for keyword tags
 * @param {Array} matches - Array of keyword match objects
 * @param {string} bootstrapClass - Bootstrap class for tag styling
 * @returns {string} HTML for keyword tags
 */
function createKeywordTags(matches, bootstrapClass) {
    return matches.map(match => {
        const score = match.score || 1.0;
        return `
            <span class="badge bg-${bootstrapClass} keyword-tag m-1 p-2" 
                  data-keyword="${match.keyword}"
                  title="${match.match_type || 'direct'} match (${Math.round(score * 100)}%)">
                ${match.keyword}
            </span>
        `;
    }).join('');
}

/**
 * Highlight all instances of a keyword
 * @param {string} keyword - Keyword to highlight
 */
function highlightAllInstances(keyword) {
    const highlights = document.querySelectorAll(`.keyword-highlight[data-keyword="${keyword}"]`);
    highlights.forEach(highlight => {
        highlight.style.boxShadow = '0 0 5px rgba(0,0,0,0.5)';
        highlight.style.fontWeight = 'bold';
    });
    
    const tags = document.querySelectorAll(`.keyword-tag[data-keyword="${keyword}"]`);
    tags.forEach(tag => {
        tag.style.boxShadow = '0 0 5px rgba(0,0,0,0.5)';
    });
}

/**
 * Reset highlighting for a keyword
 * @param {string} keyword - Keyword to reset
 */
function resetHighlighting(keyword) {
    const highlights = document.querySelectorAll(`.keyword-highlight[data-keyword="${keyword}"]`);
    highlights.forEach(highlight => {
        highlight.style.boxShadow = 'none';
        highlight.style.fontWeight = 'normal';
    });
    
    const tags = document.querySelectorAll(`.keyword-tag[data-keyword="${keyword}"]`);
    tags.forEach(tag => {
        tag.style.boxShadow = 'none';
    });
}

/**
 * Scroll to the first instance of a keyword
 * @param {string} keyword - Keyword to scroll to
 */
function scrollToFirstInstance(keyword) {
    const firstInstance = document.querySelector(`.keyword-highlight[data-keyword="${keyword}"]`);
    if (firstInstance) {
        firstInstance.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

/**
 * Escape special characters in a string for use in RegExp
 * @param {string} string - String to escape
 * @returns {string} Escaped string
 */
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
} 