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

async function fetchXGBData() {
    try {
        const response = await fetch('http://127.0.0.1:5000//MTH_XGBoost');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching JSON data:', error);
    }
}
const heatmapData = {
    labels: ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7'],
    datasets: [{
        data: [
            [3614, 5, 1, 7, 1, 5, 12],
            [5, 388, 0, 0, 0, 0, 0],
            [0, 0, 19, 0, 0, 0, 0],
            [6, 0, 0, 598, 0, 0, 5],
            [2, 0, 0, 0, 5, 0, 0],
            [1, 0, 0, 0, 0, 250, 0],
            [4, 0, 0, 0, 0, 0, 432]
        ],
        label: 'Heatmap Data',
    }],
};

function showHeatmap() {
    destroyCurrentChart()
    const newFormatData = convertToNewFormat(heatmapData.datasets[0].data);
    console.log(newFormatData)
    const newDataset = [{
        label: 'My Matrix',
        data: newFormatData,
        borderWidth: 1,
        width: ({chart}) => (chart.chartArea || {}).width / 7 - 1,
        height: ({chart}) =>(chart.chartArea || {}).height / 7 - 1,
    }];
    currentChart = new Chart(ctx, {
        type: 'matrix',
        data: {
            labels: heatmapData.labels,
            datasets: newDataset,
        },
        options: {
            plugins: {
                legend: false,
                tooltip: {
                  callbacks: {
                    title() {
                      return '';
                    },
                    label(context) {
                      const v = context.dataset.data[context.dataIndex];
                      return ['x: ' + v.x, 'y: ' + v.y, 'v: ' + v.v];
                    }
                  }
                }
            },
            scales: {
                x: {
                    ticks: {
                        stepSize: 1
                      },
                      grid: {
                        display: false
                    }
                },
                y: {
                    offset: true,
                ticks: {
                    stepSize: 1
                },
                    grid: {
                    display: false
                }
                }
            }
        }
    });
}

function convertToNewFormat(data) {
    const newData = [];
    data.forEach((row, x) => {
        row.forEach((value, y) => {
            newData.push({ x: x + 1, y: y + 1, v: value });
        });
    });
    return newData;
}

async function fetchXGBChart() {
    try {
        const data = await fetchXGBData();
        console.log(data)
        heatmapData.datasets[0].data = data.cm;
        showHeatmap();
    } catch (error) {
        console.error('Error fetching JSON data:', error);
    }
}

// Initial chart display
showLineChart();