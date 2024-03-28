const ctx = document.getElementById('myChart').getContext('2d');
let currentChart;

const heatmapData = {
    labels: ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7'],
    datasets: [{
        data: [
            [20, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        label: 'Heatmap Data',
    }],
};

function destroyCurrentChart() {
    if (currentChart) {
        currentChart.destroy();
    }
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

function showHeatmap() {
    destroyCurrentChart()
    const container = document.getElementById('content');
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    ctx.canvas.width = containerWidth;
    ctx.canvas.height = containerHeight;
    const newFormatData = convertToNewFormat(heatmapData.datasets[0].data);
    console.log(newFormatData)
    const minValue = Math.min(...newFormatData.map(value => value.v));
    const maxValue = Math.max(...newFormatData.map(value => value.v));
    const colorScale = chroma.scale(['#f7fbff', '#04AA6D']).domain([minValue, maxValue]);
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

function MTHSelected(){
    var MTHButton = document.getElementById('MTH');
    var LCCDEButton = document.getElementById('LCCDE');
    var TreeButton = document.getElementById('Tree-Based');
    MTHButton.style.background = '#04AA6D';
    LCCDEButton.style.background = 'darkgray';
    TreeButton.style.background = 'darkgray';
}

function LCCDESelected(){
    var MTHButton = document.getElementById('MTH');
    var LCCDEButton = document.getElementById('LCCDE');
    var TreeButton = document.getElementById('Tree-Based');
    MTHButton.style.background = 'darkgray';
    LCCDEButton.style.background = '#04AA6D';
    TreeButton.style.background = 'darkgray';
}

function TreeSelected(){
    var MTHButton = document.getElementById('MTH');
    var LCCDEButton = document.getElementById('LCCDE');
    var TreeButton = document.getElementById('Tree-Based');
    MTHButton.style.background = 'darkgray';
    LCCDEButton.style.background = 'darkgray';
    TreeButton.style.background = '#04AA6D';
}

function addRight() {
    var graphContainer = document.createElement('div');
    graphContainer.classList.add('graphContainer');
    var canvas = document.createElement('canvas');
    var uniqueId = 'myChart_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
    canvas.id = uniqueId;
    graphContainer.appendChild(canvas);
    var rightAdd = document.querySelector('.rightAdd');
    var graphContent = document.querySelector('.upperContent');
    graphContent.insertBefore(graphContainer, rightAdd);
    var ctx = canvas.getContext('2d');
    var myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            datasets: [{
                label: '# of Votes',
                data: [12, 19, 3, 5, 2, 3],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function addBelow() {
    var graph = document.createElement('div');  
    graph.classList.add('graphContainer');
    graph.textContent = 'Graph below';
    var leftAdd = document.querySelector('.leftAdd');
    var graphContent = document.querySelector('.graphcontent');
    graphContent.insertBefore(graph, leftAdd);
}

document.querySelector('input')
    .addEventListener('input', evt => {
  trainText = document.getElementById('trainPercent');
  testText = document.getElementById('testPercent');
  trainText.textContent = evt.target.value + '% Train';
  testText.textContent = (100 - evt.target.value) + '% Test';
});

var container = document.getElementById("container");
var content = document.querySelector('.content');
var sidebar = document.getElementById('sidebarRestrict');
var scale = 1;
var isDragging = false;
var startX, startY, initialLeft, initialTop;

document.addEventListener('mousedown', function(e) {
    if (e.clientX > 280 && e.clientY > 75) {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        initialLeft = container.offsetLeft;
        initialTop = container.offsetTop;
        e.preventDefault();
        console.log(startX);
        console.log(startY);
        console.log(initialLeft);
        console.log(initialTop);
    }
});

document.addEventListener('mousemove', function(e) {
    if (e.clientX > 280 && e.clientY > 75) {
        if (isDragging) {
            var deltaX = e.clientX - startX - 280;
            var deltaY = e.clientY - startY - 75;
            content.style.left = (initialLeft + deltaX) + 'px';
            content.style.top = (initialTop + deltaY) + 'px';
        }
    }
});

document.addEventListener('mouseup', function() {
    isDragging = false;
});

document.addEventListener('wheel', function(e) {
    if (e.clientX > 280 && e.clientY > 75) {
        e.preventDefault();
        var delta = Math.max(-1, Math.min(1, (e.wheelDelta || -e.detail)));
        var zoomStep = 0.1;
        if (delta > 0) {
            scale += zoomStep;
        } else {
            scale -= zoomStep;
        }
        content.style.transform = 'scale(' + scale + ')';
    }
});

showHeatmap();