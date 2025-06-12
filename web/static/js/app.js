/**
 * Student Performance Prediction App - JavaScript Utilities
 */

// Global app configuration
const AppConfig = {
    apiBase: '/api',
    refreshInterval: 30000, // 30 seconds
    chartColors: {
        primary: '#667eea',
        secondary: '#764ba2',
        success: '#28a745',
        warning: '#ffc107',
        danger: '#dc3545',
        info: '#17a2b8'
    }
};

// Utility functions
const Utils = {
    /**
     * Format performance metrics for display
     */
    formatMetric: function(value, type = 'percentage') {
        if (value === null || value === undefined) return 'N/A';
        
        switch (type) {
            case 'percentage':
                return (value * 100).toFixed(1) + '%';
            case 'decimal':
                return value.toFixed(3);
            case 'integer':
                return Math.round(value);
            default:
                return value.toString();
        }
    },

    /**
     * Get risk level color
     */
    getRiskColor: function(riskLevel) {
        const colorMap = {
            'low_risk': AppConfig.chartColors.success,
            'medium_risk': AppConfig.chartColors.warning,
            'high_risk': AppConfig.chartColors.danger
        };
        return colorMap[riskLevel] || AppConfig.chartColors.info;
    },

    /**
     * Debounce function for performance
     */
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Show loading state
     */
    showLoading: function(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.add('loading');
        }
    },

    /**
     * Hide loading state
     */
    hideLoading: function(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.classList.remove('loading');
        }
    },

    /**
     * Show notification
     */
    showNotification: function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
};

// Chart utilities
const ChartUtils = {
    /**
     * Create a feature importance chart
     */
    createFeatureImportanceChart: function(elementId, data, title = 'Feature Importance') {
        const features = data.slice(0, 10); // Top 10 features
        
        const trace = {
            x: features.map(f => f.importance),
            y: features.map(f => f.feature),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: features.map((_, i) => 
                    `rgba(102, 126, 234, ${1 - (i * 0.08)})`
                )
            }
        };
        
        const layout = {
            title: title,
            xaxis: { title: 'Importance Score' },
            yaxis: { 
                title: 'Features',
                autorange: 'reversed',
                tickfont: { size: 10 }
            },
            margin: { l: 120, r: 40, t: 60, b: 40 },
            height: 400,
            font: { family: '"Segoe UI", Roboto, sans-serif' }
        };
        
        Plotly.newPlot(elementId, [trace], layout, {
            responsive: true,
            displayModeBar: false
        });
    },

    /**
     * Create model comparison chart
     */
    createModelComparisonChart: function(elementId, performanceData) {
        const models = Object.keys(performanceData);
        const metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1'];
        const metricNames = ['Accuracy', 'Precision', 'Recall', 'F1-Score'];
        
        const traces = metrics.map((metric, index) => ({
            x: models,
            y: models.map(model => 
                performanceData[model][metric] ? 
                performanceData[model][metric] * 100 : 0
            ),
            type: 'bar',
            name: metricNames[index],
            marker: {
                color: [
                    AppConfig.chartColors.primary,
                    AppConfig.chartColors.success,
                    AppConfig.chartColors.warning,
                    AppConfig.chartColors.info
                ][index]
            }
        }));
        
        const layout = {
            title: 'Model Performance Comparison',
            xaxis: { title: 'Models' },
            yaxis: { title: 'Score (%)' },
            barmode: 'group',
            height: 400,
            font: { family: '"Segoe UI", Roboto, sans-serif' }
        };
        
        Plotly.newPlot(elementId, traces, layout, {
            responsive: true,
            displayModeBar: false
        });
    },

    /**
     * Create risk distribution pie chart
     */
    createRiskDistributionChart: function(elementId, riskData) {
        const trace = {
            values: Object.values(riskData),
            labels: Object.keys(riskData).map(risk => 
                risk.replace('_', ' ').toUpperCase()
            ),
            type: 'pie',
            marker: {
                colors: Object.keys(riskData).map(risk => 
                    Utils.getRiskColor(risk)
                )
            },
            textinfo: 'label+percent',
            textposition: 'outside'
        };
        
        const layout = {
            title: 'Risk Level Distribution',
            height: 400,
            font: { family: '"Segoe UI", Roboto, sans-serif' },
            showlegend: false
        };
        
        Plotly.newPlot(elementId, [trace], layout, {
            responsive: true,
            displayModeBar: false
        });
    }
};

// API utilities
const ApiUtils = {
    /**
     * Generic API call wrapper
     */
    call: async function(endpoint, options = {}) {
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(AppConfig.apiBase + endpoint, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return { success: true, data };
        } catch (error) {
            console.error('API call failed:', error);
            return { success: false, error: error.message };
        }
    },

    /**
     * Get model performance data
     */
    getModelPerformance: async function() {
        return this.call('/model-performance');
    },

    /**
     * Get feature importance data
     */
    getFeatureImportance: async function(modelName) {
        return this.call(`/feature-importance/${modelName}`);
    },

    /**
     * Make prediction
     */
    predict: async function(studentData) {
        return this.call('/predict', {
            method: 'POST',
            body: JSON.stringify(studentData)
        });
    }
};

// Form utilities
const FormUtils = {
    /**
     * Serialize form data to object
     */
    serializeForm: function(formElement) {
        const formData = new FormData(formElement);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        return data;
    },

    /**
     * Validate prediction form
     */
    validatePredictionForm: function(data) {
        const required = ['age', 'sex', 'school', 'address'];
        const missing = required.filter(field => !data[field]);
        
        if (missing.length > 0) {
            return {
                valid: false,
                message: `Missing required fields: ${missing.join(', ')}`
            };
        }
        
        // Age validation
        if (data.age < 15 || data.age > 22) {
            return {
                valid: false,
                message: 'Age must be between 15 and 22'
            };
        }
        
        return { valid: true };
    },

    /**
     * Reset form to default values
     */
    resetForm: function(formId) {
        const form = document.getElementById(formId);
        if (form) {
            form.reset();
            
            // Reset any custom states
            form.querySelectorAll('.is-invalid').forEach(el => {
                el.classList.remove('is-invalid');
            });
        }
    }
};

// Data processing utilities
const DataUtils = {
    /**
     * Process prediction results for display
     */
    processPredictionResults: function(predictions) {
        const processed = {
            consensus: null,
            individual: [],
            confidence: 0
        };
        
        const validPredictions = Object.entries(predictions)
            .filter(([_, pred]) => !pred.error);
        
        if (validPredictions.length === 0) {
            return processed;
        }
        
        // Calculate consensus
        const riskCounts = {};
        validPredictions.forEach(([_, pred]) => {
            const risk = pred.prediction;
            riskCounts[risk] = (riskCounts[risk] || 0) + 1;
        });
        
        processed.consensus = Object.keys(riskCounts)
            .reduce((a, b) => riskCounts[a] > riskCounts[b] ? a : b);
        
        // Calculate average confidence
        const confidences = validPredictions
            .map(([_, pred]) => pred.probability ? Math.max(...pred.probability) : 0)
            .filter(conf => conf > 0);
        
        processed.confidence = confidences.length > 0 ? 
            confidences.reduce((a, b) => a + b, 0) / confidences.length : 0;
        
        // Process individual results
        processed.individual = Object.entries(predictions).map(([model, pred]) => ({
            model: model.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
            prediction: pred.prediction || 'Error',
            confidence: pred.probability ? Math.max(...pred.probability) * 100 : 0,
            error: pred.error || null
        }));
        
        return processed;
    },

    /**
     * Aggregate feature importance across models
     */
    aggregateFeatureImportance: function(importanceData) {
        const aggregated = {};
        
        Object.values(importanceData).forEach(features => {
            features.forEach(feature => {
                if (aggregated[feature.feature]) {
                    aggregated[feature.feature] += feature.importance;
                } else {
                    aggregated[feature.feature] = feature.importance;
                }
            });
        });
        
        return Object.entries(aggregated)
            .map(([feature, importance]) => ({ feature, importance }))
            .sort((a, b) => b.importance - a.importance);
    }
};

// Page-specific functionality
const PageHandlers = {
    /**
     * Dashboard page initialization
     */
    initDashboard: function() {
        console.log('Initializing dashboard...');
        
        // Auto-refresh dashboard data
        const refreshDashboard = Utils.debounce(async () => {
            const result = await ApiUtils.getModelPerformance();
            if (result.success) {
                // Update dashboard metrics
                this.updateDashboardMetrics(result.data);
            }
        }, 1000);
        
        setInterval(refreshDashboard, AppConfig.refreshInterval);
    },

    /**
     * Prediction page initialization
     */
    initPrediction: function() {
        console.log('Initializing prediction page...');
        
        const form = document.getElementById('prediction-form');
        if (form) {
            form.addEventListener('submit', this.handlePredictionSubmit.bind(this));
        }
    },

    /**
     * Handle prediction form submission
     */
    handlePredictionSubmit: async function(event) {
        event.preventDefault();
        
        const form = event.target;
        const data = FormUtils.serializeForm(form);
        
        // Validate form
        const validation = FormUtils.validatePredictionForm(data);
        if (!validation.valid) {
            Utils.showNotification(validation.message, 'danger');
            return;
        }
        
        // Show loading state
        Utils.showLoading('prediction-results');
        
        try {
            const result = await ApiUtils.predict(data);
            
            if (result.success) {
                this.displayPredictionResults(result.data);
                Utils.showNotification('Prediction completed successfully!', 'success');
            } else {
                throw new Error(result.error);
            }
        } catch (error) {
            Utils.showNotification('Prediction failed: ' + error.message, 'danger');
        } finally {
            Utils.hideLoading('prediction-results');
        }
    },

    /**
     * Display prediction results
     */
    displayPredictionResults: function(results) {
        const processed = DataUtils.processPredictionResults(results.predictions);
        const container = document.getElementById('prediction-results');
        
        if (!container) return;
        
        // Create results HTML
        const html = `
            <div class="prediction-consensus mb-4">
                <h6><i class="fas fa-bullseye"></i> Consensus Prediction</h6>
                <div class="risk-indicator risk-${processed.consensus.toLowerCase().replace('_', '-')}">
                    ${processed.consensus.replace('_', ' ').toUpperCase()} RISK
                </div>
                <small class="text-muted">
                    Average Confidence: ${processed.confidence.toFixed(1)}%
                </small>
            </div>
            
            <div class="individual-predictions">
                <h6><i class="fas fa-list"></i> Individual Model Predictions</h6>
                ${processed.individual.map(pred => `
                    <div class="prediction-item card mb-2">
                        <div class="card-body py-2">
                            <div class="d-flex justify-content-between align-items-center">
                                <strong>${pred.model}</strong>
                                <div>
                                    ${pred.error ? 
                                        `<span class="badge bg-danger">Error</span>` :
                                        `<span class="badge bg-primary">${pred.prediction.replace('_', ' ')}</span>`
                                    }
                                    ${pred.confidence > 0 ? 
                                        `<small class="text-muted ms-2">${pred.confidence.toFixed(1)}%</small>` : 
                                        ''
                                    }
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        container.innerHTML = html;
        container.style.display = 'block';
    },

    /**
     * Update dashboard metrics
     */
    updateDashboardMetrics: function(performanceData) {
        // This would update real-time metrics on the dashboard
        console.log('Updating dashboard metrics:', performanceData);
    }
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Student Performance Prediction App initialized');
    
    // Initialize page-specific handlers based on current page
    const currentPage = document.body.getAttribute('data-page');
    
    switch (currentPage) {
        case 'dashboard':
            PageHandlers.initDashboard();
            break;
        case 'prediction':
            PageHandlers.initPrediction();
            break;
        default:
            console.log('No specific page handler for:', currentPage);
    }
});

// Export utilities for global use
window.AppUtils = {
    Utils,
    ChartUtils,
    ApiUtils,
    FormUtils,
    DataUtils,
    PageHandlers
}; 