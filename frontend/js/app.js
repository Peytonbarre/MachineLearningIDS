
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
        document.getElementById('loadingIndicator').classList.remove('hidden');
        const response = await fetch('http://127.0.0.1:5000//MTH_XGBoost');
        const data = await response.json();
        document.getElementById('loadingIndicator').classList.add('hidden');
        return data;
    } catch (error) {
        console.error('Error fetching JSON data:', error);
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}

async function fetchETData() {
    try {
        document.getElementById('loadingIndicator').classList.remove('hidden');
        const response = await fetch('http://127.0.0.1:5000//MTH_ET');
        const data = await response.json();
        document.getElementById('loadingIndicator').classList.add('hidden');
        return data;
    } catch (error) {
        console.error('Error fetching JSON data:', error);
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}

async function fetchDTData() {
    try {
        document.getElementById('loadingIndicator').classList.remove('hidden');
        const response = await fetch('http://127.0.0.1:5000//MTH_DT');
        const data = await response.json();
        document.getElementById('loadingIndicator').classList.add('hidden');
        return data;
    } catch (error) {
        console.error('Error fetching JSON data:', error);
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}

async function fetchRFData() {
    try {
        document.getElementById('loadingIndicator').classList.remove('hidden');
        const response = await fetch('http://127.0.0.1:5000//MTH_RF');
        const data = await response.json();
        document.getElementById('loadingIndicator').classList.add('hidden');
        return data;
    } catch (error) {
        console.error('Error fetching JSON data:', error);
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}

async function fetchSTACKData() {
    try {
        document.getElementById('loadingIndicator').classList.remove('hidden');
        const response = await fetch('http://127.0.0.1:5000//MTH_STACK');
        const data = await response.json();
        document.getElementById('loadingIndicator').classList.add('hidden');
        return data;
    } catch (error) {
        console.error('Error fetching JSON data:', error);
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}

const heatmapData = {
    labels: ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7'],
    datasets: [{
        data: [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        label: 'Heatmap Data',
    }],
};

function showHeatmap() {
    destroyCurrentChart()
    const newFormatData = convertToNewFormat(heatmapData.datasets[0].data);
    console.log(newFormatData)
    const minValue = Math.min(...newFormatData.map(value => value.v));
    const maxValue = Math.max(...newFormatData.map(value => value.v));
    const colorScale = chroma.scale(['#f7fbff', '#4428BC']).domain([minValue, maxValue]);
    const backgroundColors = newFormatData.map(value => colorScale(value.v).hex());
    const newDataset = [{
        label: 'My Matrix',
        data: newFormatData,
        backgroundColor: backgroundColors,
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
            },
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
    document.getElementById('FeaturesPanel1').style.display = 'none';
    document.getElementById('FeaturesButton1').style.backgroundColor = '';
}

async function fetchETChart() {
    try {
        const data = await fetchETData();
        console.log(data)
        heatmapData.datasets[0].data = data.cm;
        showHeatmap();
    } catch (error) {
        console.error('Error fetching JSON data:', error);
    }
    document.getElementById('FeaturesPanel1').style.display = 'none';
    document.getElementById('FeaturesButton1').style.backgroundColor = '';
}

async function fetchDTChart() {
    try {
        const data = await fetchDTData();
        console.log(data)
        heatmapData.datasets[0].data = data.cm;
        showHeatmap();
    } catch (error) {
        console.error('Error fetching JSON data:', error);
    }
    document.getElementById('FeaturesPanel1').style.display = 'none';
    document.getElementById('FeaturesButton1').style.backgroundColor = '';
}

async function fetchRFChart() {
    try {
        const data = await fetchRFData();
        console.log(data)
        heatmapData.datasets[0].data = data.cm;
        showHeatmap();
    } catch (error) {
        console.error('Error fetching JSON data:', error);
    }
    document.getElementById('FeaturesPanel1').style.display = 'none';
    document.getElementById('FeaturesButton1').style.backgroundColor = '';
}

async function fetchSTACKChart() {
    try {
        const data = await fetchSTACKData();
        console.log(data)
        heatmapData.datasets[0].data = data.cm;
        showHeatmap();
    } catch (error) {
        console.error('Error fetching JSON data:', error);
    }
    document.getElementById('FeaturesPanel1').style.display = 'none';
    document.getElementById('FeaturesButton1').style.backgroundColor = '';
}


document.addEventListener('DOMContentLoaded', function() {
    var buttons = document.querySelectorAll('.midButton');
    var header  = document.getElementById('DashHeader');

    //This changes the header background-color when a ML Algorithm button is pressed
    buttons.forEach(function(button) {
        button.addEventListener('click', function() {
            var color = window.getComputedStyle(button).getPropertyValue('background-color');
            header.style.backgroundColor = color;
        });
    });
});

    //hides Features panel when exit button is clicked
    document.getElementById('closeButton1').addEventListener('click', function() {
        document.getElementById('FeaturesPanel1').style.display = 'none';
        document.getElementById('FeaturesButton1').style.backgroundColor = '';
    });

    //toggles display of feature panel via clicks to the 'Features...' button
    function toggleFeatPanel1() {
        var panel = document.getElementById('FeaturesPanel1');
        var button = document.getElementById('FeaturesButton1');

        if (panel.style.display === 'none' || panel.style.display === '') {
            panel.style.display = 'block';
            button.style.backgroundColor = '#444';
        } else {
            panel.style.display = 'none';
            button.style.backgroundColor = '';
        }
    }

     //hides Features panel when exit button is clicked
     document.getElementById('closeButton2').addEventListener('click', function() {
        document.getElementById('FeaturesPanel2').style.display = 'none';
        document.getElementById('FeaturesButton2').style.backgroundColor = '';
    });

    //toggles display of feature panel via clicks to the 'Features...' button
    function toggleFeatPanel2() {
        var panel = document.getElementById('FeaturesPanel2');
        var button = document.getElementById('FeaturesButton2');

        if (panel.style.display === 'none' || panel.style.display === '') {
            panel.style.display = 'block';
            button.style.backgroundColor = '#444';
        } else {
            panel.style.display = 'none';
            button.style.backgroundColor = '';
        }
    }

     //hides Features panel when exit button is clicked
     document.getElementById('closeButton3').addEventListener('click', function() {
        document.getElementById('FeaturesPanel3').style.display = 'none';
        document.getElementById('FeaturesButton3').style.backgroundColor = '';
    });

    //toggles display of feature panel via clicks to the 'Features...' button
    function toggleFeatPanel3() {
        var panel = document.getElementById('FeaturesPanel3');
        var button = document.getElementById('FeaturesButton3');

        if (panel.style.display === 'none' || panel.style.display === '') {
            panel.style.display = 'block';
            button.style.backgroundColor = '#444';
        } else {
            panel.style.display = 'none';
            button.style.backgroundColor = '';
        }
    }

// Initial chart display
showLineChart();