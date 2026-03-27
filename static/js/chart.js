/**
 * Custom Chart.js configurations and initialization for the dashboard
 * This file handles all chart creation and rendering for the analytics dashboard
 */

// Wait for DOM to be fully loaded before initializing charts
document.addEventListener('DOMContentLoaded', function() {
    
    // Check if we're on the dashboard page by looking for chart canvases
    const riskCanvas = document.getElementById('riskChart');
    if (!riskCanvas) return; // Exit if not on dashboard page
    
    // Get data from the data attribute (passed from Flask)
    const dashboardDataElement = document.getElementById('dashboard-data');
    let chartData = null;
    
    if (dashboardDataElement) {
        try {
            chartData = JSON.parse(dashboardDataElement.textContent);
        } catch(e) {
            console.error('Error parsing dashboard data:', e);
        }
    }
    
    // If data not available via data attribute, it will be passed via JavaScript variable
    // from the template. The template injects data as a JavaScript object.
    
    // Create all charts
    initializeCharts(chartData);
});

/**
 * Main function to initialize all dashboard charts
 * @param {Object} data - The dashboard data from Flask
 */
function initializeCharts(data) {
    
    // 1. Risk Distribution Chart (Doughnut)
    const riskChartElement = document.getElementById('riskChart');
    if (riskChartElement && data && data.risk_distribution) {
        new Chart(riskChartElement, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'High Risk'],
                datasets: [{
                    data: [data.risk_distribution.low_risk, data.risk_distribution.high_risk],
                    backgroundColor: ['#28a745', '#dc3545'],
                    borderColor: ['#1e7e34', '#b02a37'],
                    borderWidth: 2,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                cutout: '65%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: {
                                size: 13,
                                weight: '500'
                            },
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} users (${percentage}%)`;
                            }
                        },
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: '#fff',
                        bodyColor: '#e2e8f0',
                        padding: 10,
                        cornerRadius: 8
                    }
                }
            }
        });
    }
    
    // 2. Screen Time Distribution Chart (Bar)
    const screenTimeElement = document.getElementById('screenTimeChart');
    if (screenTimeElement && data && data.screen_distribution) {
        new Chart(screenTimeElement, {
            type: 'bar',
            data: {
                labels: Object.keys(data.screen_distribution),
                datasets: [{
                    label: 'Number of Users',
                    data: Object.values(data.screen_distribution),
                    backgroundColor: '#667eea',
                    borderRadius: 8,
                    barPercentage: 0.7,
                    categoryPercentage: 0.8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Users: ${context.raw}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0,0,0,0.05)',
                            drawBorder: false
                        },
                        title: {
                            display: true,
                            text: 'Number of Users',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Screen Time Category',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    }
                }
            }
        });
    }
    
    // 3. Sleep vs Addiction Chart (Line)
    const sleepElement = document.getElementById('sleepChart');
    if (sleepElement && data && data.sleep_vs_addiction) {
        const sleepLabels = Object.keys(data.sleep_vs_addiction);
        const sleepValues = Object.values(data.sleep_vs_addiction);
        
        new Chart(sleepElement, {
            type: 'line',
            data: {
                labels: sleepLabels,
                datasets: [{
                    label: 'Addiction Rate',
                    data: sleepValues,
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 5,
                    pointHoverRadius: 8,
                    pointBackgroundColor: '#ffc107',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                return `Addiction Rate: ${(value * 100).toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        },
                        title: {
                            display: true,
                            text: 'Addiction Rate',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Sleep Hours',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    }
                }
            }
        });
    }
    
    // 4. Social Media Usage Comparison (Bar)
    const socialElement = document.getElementById('socialChart');
    if (socialElement && data && data.social_by_risk) {
        new Chart(socialElement, {
            type: 'bar',
            data: {
                labels: ['Low Risk', 'High Risk'],
                datasets: [{
                    label: 'Average Social Media Hours',
                    data: [data.social_by_risk['0'] || 0, data.social_by_risk['1'] || 0],
                    backgroundColor: ['#28a745', '#dc3545'],
                    borderRadius: 8,
                    barPercentage: 0.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw.toFixed(1)} hours/day`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        },
                        title: {
                            display: true,
                            text: 'Hours per Day',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
    
    // 5. Gaming Usage Comparison (Bar)
    const gamingElement = document.getElementById('gamingChart');
    if (gamingElement && data && data.gaming_by_risk) {
        new Chart(gamingElement, {
            type: 'bar',
            data: {
                labels: ['Low Risk', 'High Risk'],
                datasets: [{
                    label: 'Average Gaming Hours',
                    data: [data.gaming_by_risk['0'] || 0, data.gaming_by_risk['1'] || 0],
                    backgroundColor: ['#28a745', '#dc3545'],
                    borderRadius: 8,
                    barPercentage: 0.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw.toFixed(1)} hours/day`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0,0,0,0.05)'
                        },
                        title: {
                            display: true,
                            text: 'Hours per Day',
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
    
    // 6. Gender Distribution Chart (Pie)
    const genderElement = document.getElementById('genderChart');
    if (genderElement && data && data.gender_counts) {
        new Chart(genderElement, {
            type: 'pie',
            data: {
                labels: ['Female', 'Male'],
                datasets: [{
                    data: [data.gender_counts['0'] || 0, data.gender_counts['1'] || 0],
                    backgroundColor: ['#6f42c1', '#20c997'],
                    borderWidth: 0,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: {
                                size: 13,
                                weight: '500'
                            },
                            padding: 15
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} users (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }
}

/**
 * Helper function to update chart data dynamically
 * @param {Chart} chart - The chart instance to update
 * @param {Object} newData - New data to update the chart with
 */
function updateChartData(chart, newData) {
    if (!chart) return;
    
    chart.data.datasets.forEach((dataset, index) => {
        if (newData.datasets && newData.datasets[index]) {
            dataset.data = newData.datasets[index].data;
        }
    });
    
    if (newData.labels) {
        chart.data.labels = newData.labels;
    }
    
    chart.update();
}

/**
 * Create a new chart with custom configuration
 * @param {string} canvasId - ID of the canvas element
 * @param {string} type - Chart type ('bar', 'line', 'pie', 'doughnut')
 * @param {Object} data - Chart data
 * @param {Object} options - Custom chart options
 * @returns {Chart} - The created chart instance
 */
function createCustomChart(canvasId, type, data, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    font: {
                        size: 12
                    }
                }
            }
        }
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    return new Chart(canvas, {
        type: type,
        data: data,
        options: mergedOptions
    });
}

// Export functions for potential use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializeCharts,
        updateChartData,
        createCustomChart
    };
}