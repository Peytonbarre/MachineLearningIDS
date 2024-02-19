// Sample data for the charts
const data = {
    labels: ['January', 'February', 'March', 'April', 'May'],
    datasets: [{
        label: 'Sample Data',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        data: [65, 59, 80, 81, 56],
    }],
};

// Chart.js functions
const ctx = document.getElementById('myChart').getContext('2d');
let currentChart;

function showLineChart() {
    destroyCurrentChart();
    currentChart = new Chart(ctx, {
        type: 'line',
        data: data,
    });
}

function showBarChart() {
    destroyCurrentChart();
    currentChart = new Chart(ctx, {
        type: 'bar',
        data: data,
    });
}

function showPieChart() {
    destroyCurrentChart();
    currentChart = new Chart(ctx, {
        type: 'pie',
        data: data,
    });
}

function showDoughnutChart() {
    destroyCurrentChart();
    currentChart = new Chart(ctx, {
        type: 'doughnut',
        data: data,
    });
}

function destroyCurrentChart() {
    if (currentChart) {
        currentChart.destroy();
    }
}

// Initial chart display
showLineChart();