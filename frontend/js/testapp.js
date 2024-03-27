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

showHeatmap();