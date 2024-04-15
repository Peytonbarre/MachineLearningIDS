const ctx = document.getElementById("myChart").getContext("2d");
let currentChart;

const heatmapData = {
    labels: [
        "Category 1",
        "Category 2",
        "Category 3",
        "Category 4",
        "Category 5",
        "Category 6",
        "Category 7",
    ],
    datasets: [
        {
            data: [
                [20, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            label: "Heatmap Data",
        },
    ],
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
    destroyCurrentChart();
    const container = document.getElementById("content");
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    ctx.canvas.width = containerWidth;
    ctx.canvas.height = containerHeight;
    const newFormatData = convertToNewFormat(heatmapData.datasets[0].data);
    const minValue = Math.min(...newFormatData.map((value) => value.v));
    const maxValue = Math.max(...newFormatData.map((value) => value.v));
    const colorScale = chroma
        .scale(["#f7fbff", "#04AA6D"])
        .domain([minValue, maxValue]);
    const backgroundColors = newFormatData.map((value) =>
        colorScale(value.v).hex()
    );
    const newDataset = [
        {
            label: "Matrix Dataset",
            data: newFormatData,
            backgroundColor: backgroundColors,
            borderWidth: 1,
            width: ({ chart }) => (chart.chartArea || {}).width / 7 - 1,
            height: ({ chart }) => (chart.chartArea || {}).height / 7 - 1,
        },
    ];
    currentChart = new Chart(ctx, {
        type: "matrix",
        data: {
            labels: heatmapData.labels,
            datasets: newDataset,
        },
        options: {
            plugins: {
                legend: {
                    title: {
                      display: true,
                      text: 'Confusion Matrix',
                    }
                },
                tooltip: {
                    callbacks: {
                        title() {
                            return "";
                        },
                        label(context) {
                            const v = context.dataset.data[context.dataIndex];
                            return ["x: " + v.x, "y: " + v.y, "v: " + v.v];
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: {
                        stepSize: 1,
                    },
                    grid: {
                        display: false,
                    },
                },
                y: {
                    offset: true,
                    ticks: {
                        stepSize: 1,
                    },
                    grid: {
                        display: false,
                    },
                },
            },
        },
    });
}

function MTHSelected() {
    var MTHButton = document.getElementById("MTH");
    var LCCDEButton = document.getElementById("LCCDE");
    var TreeButton = document.getElementById("Tree-Based");
    MTHButton.style.background = "#04AA6D";
    LCCDEButton.style.background = "darkgray";
    TreeButton.style.background = "darkgray";
}

function LCCDESelected() {
    var MTHButton = document.getElementById("MTH");
    var LCCDEButton = document.getElementById("LCCDE");
    var TreeButton = document.getElementById("Tree-Based");
    MTHButton.style.background = "darkgray";
    LCCDEButton.style.background = "#04AA6D";
    TreeButton.style.background = "darkgray";
}

function TreeSelected() {
    var MTHButton = document.getElementById("MTH");
    var LCCDEButton = document.getElementById("LCCDE");
    var TreeButton = document.getElementById("Tree-Based");
    MTHButton.style.background = "darkgray";
    LCCDEButton.style.background = "darkgray";
    TreeButton.style.background = "#04AA6D";
}

function displaySidebar() {
    var idleState = document.getElementsByClassName("idleState");
    var addGraph = document.getElementsByClassName("addingGraph");
    for (let i = 0; i < idleState.length; i++) {
        idleState[i].style.display = "none";
    }
    for (let i = 0; i < addGraph.length; i++) {
        addGraph[i].style.display = "flex";
    }
}

function cancelGraph() {
    var idleState = document.getElementsByClassName("idleState");
    var addGraph = document.getElementsByClassName("addingGraph");
    for (let i = 0; i < idleState.length; i++) {
        idleState[i].style.display = "flex";
    }
    for (let i = 0; i < addGraph.length; i++) {
        addGraph[i].style.display = "none";
    }
}

function updateBoxes() {
    var rows = document.querySelectorAll(".rows");
    var rowContainer = document.querySelector(".upperContent");
    var rowIterator = 1;
    rows.forEach((cols) => {
        var colIterator = 1;
        var altColIterator = 1;
        cols.childNodes.forEach((canvas) => {
            if (canvas.nodeType !== Node.TEXT_NODE) {
                //If next col doesn't exist OR next col's ...
                if (typeof cols.childNodes[colIterator + 3] === "undefined") {
                    canvas.childNodes[1].childNodes[1].childNodes[3].style.display =
                        "flex";
                } else {
                    canvas.childNodes[1].childNodes[1].childNodes[3].style.display =
                        "none";
                    canvas.style.padding = "";
                }

                if (
                    typeof rowContainer.childNodes[rowIterator + 2] == "undefined" ||
                    typeof rowContainer.childNodes[rowIterator + 2].childNodes[
                    altColIterator
                    ] == "undefined"
                ) {
                    canvas.childNodes[1].childNodes[3].style.display = "flex";
                } else {
                    //console.log(rowContainer.childNodes[rowIterator+2])
                    canvas.childNodes[1].childNodes[3].style.display = "none";
                }
                colIterator += 3;
                altColIterator += 2;
            }
        });
        rowIterator += 2;
    });
}

function addRight(id) {
    formData = document.getElementById("dataForm");
    formData.querySelector('input[name="coord"]').value = id;
    formData.querySelector('input[name="direction"]').value = "right";

    displaySidebar();
}

function addBelow(id) {
    formData = document.getElementById("dataForm");
    formData.querySelector('input[name="coord"]').value = id;
    formData.querySelector('input[name="direction"]').value = "left";

    displaySidebar();
}

document.querySelector("input").addEventListener("input", (evt) => {
    trainText = document.getElementById("trainPercent");
    testText = document.getElementById("testPercent");
    trainText.textContent = evt.target.value + "% Train";
    testText.textContent = 100 - evt.target.value + "% Test";
});

var container = document.getElementById("container");
var content = document.querySelector(".content");
var sidebar = document.getElementById("sidebarRestrict");
var scale = 1;
var isDragging = false;
var startX, startY, initialLeft, initialTop;

document.addEventListener("mousedown", function (e) {
    if (e.clientX > 280 && e.clientY > 75) {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        initialLeft = container.offsetLeft;
        initialTop = container.offsetTop;
        e.preventDefault();
    }
});

document.addEventListener("mousemove", function (e) {
    if (e.clientX > 280 && e.clientY > 75) {
        if (isDragging) {
            var deltaX = e.clientX - startX - 280;
            var deltaY = e.clientY - startY - 75;
            content.style.left = initialLeft + deltaX + "px";
            content.style.top = initialTop + deltaY + "px";
        }
    }
});

document.addEventListener("mouseup", function () {
    isDragging = false;
});

document.addEventListener("wheel", function (e) {
    if (e.clientX > 280 && e.clientY > 75) {
        e.preventDefault();
        var delta = Math.max(-1, Math.min(1, e.wheelDelta || -e.detail));
        var zoomStep = 0.1;
        if (delta > 0) {
            scale += zoomStep;
        } else {
            scale -= zoomStep;
        }
        content.style.transform = "scale(" + scale + ")";
    }
});

function matrixSelected() {
    var confusionMatrixButton = document.getElementById("confusionMatrix");
    var MatrixButton = document.getElementById("matrix");
    var LineButton = document.getElementById("line");
    var BarButton = document.getElementById("bar");
    var PieButton = document.getElementById("pie");
    var CalloutButton = document.getElementById("callout");
    MatrixButton.style.background = "#04AA6D";
    LineButton.style.background = "darkgray";
    BarButton.style.background = "darkgray";
    PieButton.style.background = "darkgray";
    CalloutButton.style.background = "darkgray";
    confusionMatrixButton.style.background = "darkgray";
    var matrixParametersItems =
        document.getElementsByClassName("MatrixParameters");
    var lineParametersItems = document.getElementsByClassName("LineParameters");
    var barParametersItems = document.getElementsByClassName("BarParameters");
    var pieParametersItems = document.getElementsByClassName("PieParameters");
    var calloutParametersItems =
        document.getElementsByClassName("CalloutParameters");
    for (let i = 0; i < matrixParametersItems.length; i++) {
        let element = matrixParametersItems[i];
        element.style.display = "flex";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < lineParametersItems.length; i++) {
        let element = lineParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < barParametersItems.length; i++) {
        let element = barParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < pieParametersItems.length; i++) {
        let element = pieParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < calloutParametersItems.length; i++) {
        let element = calloutParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
}

function lineSelected() {
    var LineButton = document.getElementById("matrix");
    var MatrixButton = document.getElementById("line");
    var BarButton = document.getElementById("bar");
    var PieButton = document.getElementById("pie");
    var CalloutButton = document.getElementById("callout");
    var avgOfEventButton = document.getElementById("avgOfEvent");
    avgOfEventButton.style.background = "darkgray";
    MatrixButton.style.background = "#04AA6D";
    LineButton.style.background = "darkgray";
    BarButton.style.background = "darkgray";
    PieButton.style.background = "darkgray";
    CalloutButton.style.background = "darkgray";
    var matrixParametersItems =
        document.getElementsByClassName("MatrixParameters");
    var lineParametersItems = document.getElementsByClassName("LineParameters");
    var barParametersItems = document.getElementsByClassName("BarParameters");
    var pieParametersItems = document.getElementsByClassName("PieParameters");
    var calloutParametersItems =
        document.getElementsByClassName("CalloutParameters");
    for (let i = 0; i < matrixParametersItems.length; i++) {
        let element = matrixParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < lineParametersItems.length; i++) {
        let element = lineParametersItems[i];
        element.style.display = "flex";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < barParametersItems.length; i++) {
        let element = barParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < pieParametersItems.length; i++) {
        let element = pieParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < calloutParametersItems.length; i++) {
        let element = calloutParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
}

function barSelected() {
    var LineButton = document.getElementById("matrix");
    var BarButton = document.getElementById("line");
    var MatrixButton = document.getElementById("bar");
    var PieButton = document.getElementById("pie");
    var CalloutButton = document.getElementById("callout");
    var precisionByEvent = document.getElementById("precisionByEvent");
    var recallByEvent = document.getElementById("recallByEvent");
    var f1ByEvent = document.getElementById("f1ByEvent");
    var supportByEvent = document.getElementById("supportByEvent");
    precisionByEvent.style.background = "darkgray";
    recallByEvent.style.background = "darkgray";
    f1ByEvent.style.background = "darkgray";
    supportByEvent.style.background = "darkgray";
    MatrixButton.style.background = "#04AA6D";
    LineButton.style.background = "darkgray";
    BarButton.style.background = "darkgray";
    PieButton.style.background = "darkgray";
    CalloutButton.style.background = "darkgray";
    var matrixParametersItems =
        document.getElementsByClassName("MatrixParameters");
    var lineParametersItems = document.getElementsByClassName("LineParameters");
    var barParametersItems = document.getElementsByClassName("BarParameters");
    var pieParametersItems = document.getElementsByClassName("PieParameters");
    var calloutParametersItems =
        document.getElementsByClassName("CalloutParameters");
    for (let i = 0; i < matrixParametersItems.length; i++) {
        let element = matrixParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < lineParametersItems.length; i++) {
        let element = lineParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < barParametersItems.length; i++) {
        let element = barParametersItems[i];
        element.style.display = "flex";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < pieParametersItems.length; i++) {
        let element = pieParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < calloutParametersItems.length; i++) {
        let element = calloutParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
}

function pieSelected() {
    var PieButton = document.getElementById("matrix");
    var LineButton = document.getElementById("line");
    var BarButton = document.getElementById("bar");
    var MatrixButton = document.getElementById("pie");
    var CalloutButton = document.getElementById("callout");
    var classifierComposition = document.getElementById("classifierComposition");
    classifierComposition.style.background = "darkgray";
    MatrixButton.style.background = "#04AA6D";
    LineButton.style.background = "darkgray";
    BarButton.style.background = "darkgray";
    PieButton.style.background = "darkgray";
    CalloutButton.style.background = "darkgray";
    var matrixParametersItems =
        document.getElementsByClassName("MatrixParameters");
    var lineParametersItems = document.getElementsByClassName("LineParameters");
    var barParametersItems = document.getElementsByClassName("BarParameters");
    var pieParametersItems = document.getElementsByClassName("PieParameters");
    var calloutParametersItems =
        document.getElementsByClassName("CalloutParameters");
    for (let i = 0; i < matrixParametersItems.length; i++) {
        let element = matrixParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < lineParametersItems.length; i++) {
        let element = lineParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < barParametersItems.length; i++) {
        let element = barParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < pieParametersItems.length; i++) {
        let element = pieParametersItems[i];
        element.style.display = "flex";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < calloutParametersItems.length; i++) {
        let element = calloutParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
}

function calloutSelected() {
    var LineButton = document.getElementById("matrix");
    var CalloutButton = document.getElementById("line");
    var BarButton = document.getElementById("bar");
    var PieButton = document.getElementById("pie");
    var MatrixButton = document.getElementById("callout");
    var avgAccuracy = document.getElementById("avgAccuracy");
    var avgPrecision = document.getElementById("avgPrecision");
    var avgRecall = document.getElementById("avgRecall");
    var avgf1Score = document.getElementById("avgf1Score");
    avgAccuracy.style.background = "darkgray";
    avgPrecision.style.background = "darkgray";
    avgRecall.style.background = "darkgray";
    avgf1Score.style.background = "darkgray";
    MatrixButton.style.background = "#04AA6D";
    LineButton.style.background = "darkgray";
    BarButton.style.background = "darkgray";
    PieButton.style.background = "darkgray";
    CalloutButton.style.background = "darkgray";
    var matrixParametersItems =
        document.getElementsByClassName("MatrixParameters");
    var lineParametersItems = document.getElementsByClassName("LineParameters");
    var barParametersItems = document.getElementsByClassName("BarParameters");
    var pieParametersItems = document.getElementsByClassName("PieParameters");
    var calloutParametersItems =
        document.getElementsByClassName("CalloutParameters");
    for (let i = 0; i < matrixParametersItems.length; i++) {
        let element = matrixParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < lineParametersItems.length; i++) {
        let element = lineParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < barParametersItems.length; i++) {
        let element = barParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < pieParametersItems.length; i++) {
        let element = pieParametersItems[i];
        element.style.display = "none";
        element.style.flexDirection = "column";
    }
    for (let i = 0; i < calloutParametersItems.length; i++) {
        let element = calloutParametersItems[i];
        element.style.display = "flex";
        element.style.flexDirection = "column";
    }
}

function confusionMatrixSelected() {
    var confusionMatrixButton = document.getElementById("confusionMatrix");
    confusionMatrixButton.style.background = "#04AA6D";
}

function avgOfEventSelected() {
    var avgOfEventButton = document.getElementById("avgOfEvent");
    avgOfEventButton.style.background = "#04AA6D";
}

function precisionByEventSelected() {
    var precisionByEvent = document.getElementById("precisionByEvent");
    var recallByEvent = document.getElementById("recallByEvent");
    var f1ByEvent = document.getElementById("f1ByEvent");
    var supportByEvent = document.getElementById("supportByEvent");
    precisionByEvent.style.background = "#04AA6D";
    recallByEvent.style.background = "darkgray";
    f1ByEvent.style.background = "darkgray";
    supportByEvent.style.background = "darkgray";
}

function recallByEventSelected() {
    var precisionByEvent = document.getElementById("precisionByEvent");
    var recallByEvent = document.getElementById("recallByEvent");
    var f1ByEvent = document.getElementById("f1ByEvent");
    var supportByEvent = document.getElementById("supportByEvent");
    precisionByEvent.style.background = "darkgray";
    recallByEvent.style.background = "#04AA6D";
    f1ByEvent.style.background = "darkgray";
    supportByEvent.style.background = "darkgray";
}

function f1ByEventSelected() {
    var precisionByEvent = document.getElementById("precisionByEvent");
    var recallByEvent = document.getElementById("recallByEvent");
    var f1ByEvent = document.getElementById("f1ByEvent");
    var supportByEvent = document.getElementById("supportByEvent");
    precisionByEvent.style.background = "darkgray";
    recallByEvent.style.background = "darkgray";
    f1ByEvent.style.background = "#04AA6D";
    supportByEvent.style.background = "darkgray";
}

function supportByEventSelected() {
    var precisionByEvent = document.getElementById("precisionByEvent");
    var recallByEvent = document.getElementById("recallByEvent");
    var f1ByEvent = document.getElementById("f1ByEvent");
    var supportByEvent = document.getElementById("supportByEvent");
    precisionByEvent.style.background = "darkgray";
    recallByEvent.style.background = "darkgray";
    f1ByEvent.style.background = "darkgray";
    supportByEvent.style.background = "#04AA6D";
}

function classifierCompositionSelected() {
    var classifierComposition = document.getElementById("classifierComposition");
    classifierComposition.style.background = "#04AA6D";
}

function avgAccuracySelected() {
    var avgAccuracy = document.getElementById("avgAccuracy");
    var avgPrecision = document.getElementById("avgPrecision");
    var avgRecall = document.getElementById("avgRecall");
    var avgf1Score = document.getElementById("avgf1Score");
    avgAccuracy.style.background = "#04AA6D";
    avgPrecision.style.background = "darkgray";
    avgRecall.style.background = "darkgray";
    avgf1Score.style.background = "darkgray";
}

function avgPrecisionSelected() {
    var avgAccuracy = document.getElementById("avgAccuracy");
    var avgPrecision = document.getElementById("avgPrecision");
    var avgRecall = document.getElementById("avgRecall");
    var avgf1Score = document.getElementById("avgf1Score");
    avgAccuracy.style.background = "darkgray";
    avgPrecision.style.background = "#04AA6D";
    avgRecall.style.background = "darkgray";
    avgf1Score.style.background = "darkgray";
}

function avgRecall() {
    var avgAccuracy = document.getElementById("avgAccuracy");
    var avgPrecision = document.getElementById("avgPrecision");
    var avgRecall = document.getElementById("avgRecall");
    var avgf1Score = document.getElementById("avgf1Score");
    avgAccuracy.style.background = "darkgray";
    avgPrecision.style.background = "darkgray";
    avgRecall.style.background = "#04AA6D";
    avgf1Score.style.background = "darkgray";
}

function avgf1ScoreSelected() {
    var avgAccuracy = document.getElementById("avgAccuracy");
    var avgPrecision = document.getElementById("avgPrecision");
    var avgRecall = document.getElementById("avgRecall");
    var avgf1Score = document.getElementById("avgf1Score");
    avgAccuracy.style.background = "darkgray";
    avgPrecision.style.background = "darkgray";
    avgRecall.style.background = "darkgray";
    avgf1Score.style.background = "#04AA6D";
}

document
    .getElementById("expandParameters")
    .addEventListener("click", function () {
        toggleSection("hideSection");
        var button = document.getElementById("expandParameters");
        button.children[0].textContent = button.children[0].textContent.includes(
            "less"
        )
            ? "expand_more"
            : "expand_less";
    });

document
    .getElementById("classifierButton")
    .addEventListener("click", function () {
        toggleSection("classifierList");
        var button = document.getElementById("classifierButton");
        button.children[0].textContent = button.children[0].textContent.includes(
            "less"
        )
            ? "expand_more"
            : "expand_less";
    });

document
    .getElementById("samplingButton")
    .addEventListener("click", function () {
        toggleSection("samplingList");
        var button = document.getElementById("samplingButton");
        button.children[0].textContent = button.children[0].textContent.includes(
            "less"
        )
            ? "expand_more"
            : "expand_less";
    });

document.getElementById("graphsButton").addEventListener("click", function () {
    toggleSection("graphList");
    var button = document.getElementById("graphsButton");
    button.children[0].textContent = button.children[0].textContent.includes(
        "less"
    )
        ? "expand_more"
        : "expand_less";
});

document
    .getElementById("expandGraphType")
    .addEventListener("click", function () {
        toggleSection("graphTypes");
        var button = document.getElementById("expandGraphType");
        button.children[0].textContent = button.children[0].textContent.includes(
            "less"
        )
            ? "expand_more"
            : "expand_less";
    });

document
    .getElementById("graphParametersButton")
    .addEventListener("click", function () {
        toggleSection("parametersList");
        var button = document.getElementById("graphParametersButton");
        button.children[0].textContent = button.children[0].textContent.includes(
            "less"
        )
            ? "expand_more"
            : "expand_less";
    });

//DOES NOT WORK.  I'm guessing this is due to the History div not having children?
document.getElementById('historyButton').addEventListener('click', function() {
    toggleSection('historyTable');
    var button = document.getElementById('historyButton')
    button.children[0].textContent = button.children[0].textContent.includes('less') ? 'expand_more' : 'expand_less';
});

function toggleSection(sectionClass) {
    var section = document.querySelector("." + sectionClass);
    section.classList.toggle("hidden");
}

document
    .getElementById("generateDataButton")
    .addEventListener("click", function (event) {
        event.preventDefault(); // Prevent the default form submission
        var classifier;
        var classifierList = document.getElementById("classifierList");
        for (let i = 0; i < classifierList.children.length; i++) {
            if (classifierList.children[i].style.background === "rgb(4, 170, 109)") {
                classifier = classifierList.children[i].textContent;
            }
        }
        if (classifier === undefined) {
            console.error("Missing classsifier");
        }
        var slideValue = document.getElementById("myRange").value;
        var SMOTEValue1 = document.getElementById("SMOTE1").value;
        var SMOTEValue2 = document.getElementById("SMOTE2").value;
        var graphType;
        var graphList = document.getElementById("graphTypes");
        for (let i = 0; i < graphList.children.length; i++) {
            if (graphList.children[i].style.background === "rgb(4, 170, 109)") {
                graphType = graphList.children[i].textContent;
            }
        }
        if (graphType === undefined) {
            console.error("Missing Graph Type");
        }
        var parameter;
        var parametersList = document.getElementById("parametersList");
        for (let i = 0; i < parametersList.children.length; i++) {
            for (let j = 0; j < parametersList.children[i].children.length; j++) {
                if (
                    parametersList.children[i].children[j].style.background ===
                    "rgb(4, 170, 109)"
                ) {
                    parameter = parametersList.children[i].children[j].textContent;
                }
            }
        }
        if (parameter === undefined) {
            console.error("Missing Parameter");
        }
        var formData = {
            classifier: classifier,
            slideValue: slideValue,
            SMOTEValue: SMOTEValue1 + ":" + SMOTEValue2,
            graphType: graphType,
            parameter: parameter,
        };
        cancelGraph();
        fetch("http://localhost:5000/processParameters", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(formData),
        })
            .then((response) => response.json())
            .then((data) => {
                dataForm = document.getElementById("dataForm");
                direction = dataForm.querySelector('input[name="direction"]').value;
                coord = dataForm.querySelector('input[name="coord"]').value;
                if (direction === "right") {
                    addRightHelper(graphType, parameter, data["data"], coord, classifier);
                } else if (direction === "left") {
                    addLeftHelper(graphType, parameter, data["data"], coord, classifier);
                } else {
                    throw error("Direction not recognized!");
                }
            })
            .catch((error) => {
                console.error(error);
            });
    });

//FLow == add left/right -> generateButton listener -> addLeft/right helper (HERE)
function addRightHelper(graphType, parameter, data, coord, classifier) {
    console.log(parameter);
    console.log(data);
    newCol = String(coord).split(" ")[1];
    newRow = String(coord).split(" ")[0];
    var row = document.getElementsByClassName("rows");

    var copiedContainer = document.querySelector("#myContent");
    var containerClone = copiedContainer.cloneNode(true);
    var uniqueId =
        "myChart_" + Date.now() + "_" + Math.floor(Math.random() * 1000);
    containerClone.id = uniqueId;
    containerClone.childNodes[1].childNodes[3].childNodes[1].id = String(
        newRow + " " + (parseInt(newCol) + 3)
    );
    containerClone.childNodes[3].childNodes[1].id = String(
        newRow + " " + (parseInt(newCol) + 3)
    );

    var newCol = document.createElement("div");
    newCol.classList.add("cols");
    newCol.appendChild(document.createTextNode(""));
    newCol.appendChild(containerClone);
    newCol.appendChild(document.createTextNode(""));

    row[(newRow - 1) / 2].appendChild(document.createTextNode(""));
    row[(newRow - 1) / 2].appendChild(newCol);
    row[(newRow - 1) / 2].appendChild(document.createTextNode(""));

    updateBoxes();

    var uniqueCanvasId =
        "myChart_" + Date.now() + "_" + Math.floor(Math.random() * 10000);
    canvas = containerClone.childNodes[1].childNodes[1].childNodes[1];
    canvas.id = uniqueCanvasId;
    var ctx = canvas.getContext("2d");
    if (graphType === "Matrix") {
        const newHeatMapData = {
            labels: [
                "Category 1",
                "Category 2",
                "Category 3",
                "Category 4",
                "Category 5",
                "Category 6",
                "Category 7",
            ],
            datasets: [
                {
                    data: JSON.parse(data),
                    label: "Heatmap Data",
                },
            ],
        };
        const container = document.getElementById("content");
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        ctx.canvas.width = containerWidth;
        ctx.canvas.height = containerHeight;
        console.log(newHeatMapData.datasets[0].data);
        const newFormatData = convertToNewFormat(newHeatMapData.datasets[0].data);
        const minValue = Math.min(...newFormatData.map((value) => value.v));
        const maxValue = Math.max(...newFormatData.map((value) => value.v));
        const colorScale = chroma
            .scale(["#f7fbff", "#04AA6D"])
            .domain([minValue, maxValue]);
        const backgroundColors = newFormatData.map((value) =>
            colorScale(value.v).hex()
        );
        const newDataset = [
            {
                label: "Confusion Matrix",
                data: newFormatData,
                backgroundColor: backgroundColors,
                borderWidth: 1,
                width: ({ chart }) => (chart.chartArea || {}).width / 7 - 1,
                height: ({ chart }) => (chart.chartArea || {}).height / 7 - 1,
            },
        ];
        currentChart = new Chart(ctx, {
            type: "matrix",
            data: {
                labels: heatmapData.labels,
                datasets: newDataset,
            },
            options: {
                plugins: {
                    legend: {
                        title: {
                          display: false,
                          text: '',
                        },
                        labels: {
                            display: false
                        }
                    },
                    tooltip: {
                        callbacks: {
                            title() {
                                return "";
                            },
                            label(context) {
                                const v = context.dataset.data[context.dataIndex];
                                return ["x: " + v.x, "y: " + v.y, "v: " + v.v];
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: {
                            stepSize: 1,
                        },
                        grid: {
                            display: false,
                        },
                    },
                    y: {
                        offset: true,
                        ticks: {
                            stepSize: 1,
                        },
                        grid: {
                            display: false,
                        },
                    },
                },
            },
        });
    } else if (graphType === "Bar") {
        if (classifier === "MTH" || classifier === "Tree-Based") {
            labelSet = [
                "Decision Trees",
                "Random Forest",
                "Extra Trees",
                "XGBoost",
                "Stacking",
            ];
        } else if (classifier === "LCCDE") {
            labelSet = ["LightGBM", "XGBoost", "CatBoost", "Stacking"];
        }
        console.log(data);
        var myChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labelSet,
                datasets: [
                    {
                        label: parameter,
                        data: JSON.parse(data),
                        backgroundColor: [
                            "rgba(255, 99, 132, 0.2)",
                            "rgba(54, 162, 235, 0.2)",
                            "rgba(255, 206, 86, 0.2)",
                            "rgba(75, 192, 192, 0.2)",
                            "rgba(153, 102, 255, 0.2)",
                        ],
                        borderColor: [
                            "rgba(255, 99, 132, 1)",
                            "rgba(54, 162, 235, 1)",
                            "rgba(255, 206, 86, 1)",
                            "rgba(75, 192, 192, 1)",
                            "rgba(153, 102, 255, 1)",
                        ],
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: false,
                    },
                },
            },
        });
    } else if (graphType === "Line") {
        labelSet = [];
        var lineDataset = [];
        if (classifier === "MTH" || classifier === "Tree-Based") {
            lineDataset = [
                {
                    label: "Accuracy",
                    data: [
                        JSON.parse(data)[0][0],
                        JSON.parse(data)[1][0],
                        JSON.parse(data)[2][0],
                        JSON.parse(data)[3][0],
                        JSON.parse(data)[4][0],
                    ],
                    backgroundColor: ["rgba(255, 99, 132, 0.2)"],
                    borderColor: ["rgba(255, 99, 132, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "Precision",
                    data: [
                        JSON.parse(data)[0][1],
                        JSON.parse(data)[1][1],
                        JSON.parse(data)[2][1],
                        JSON.parse(data)[3][1],
                        JSON.parse(data)[4][1],
                    ],
                    backgroundColor: ["rgba(54, 162, 235, 0.2)"],
                    borderColor: ["rgba(54, 162, 235, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "Recall",
                    data: [
                        JSON.parse(data)[0][2],
                        JSON.parse(data)[1][2],
                        JSON.parse(data)[2][2],
                        JSON.parse(data)[3][2],
                        JSON.parse(data)[4][2],
                    ],
                    backgroundColor: ["rgba(255, 206, 86, 0.2)"],
                    borderColor: ["rgba(255, 206, 86, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "F-1 Score",
                    data: [
                        JSON.parse(data)[0][3],
                        JSON.parse(data)[1][3],
                        JSON.parse(data)[2][3],
                        JSON.parse(data)[3][3],
                        JSON.parse(data)[4][3],
                    ],
                    backgroundColor: ["rgba(75, 192, 192, 0.2)"],
                    borderColor: ["rgba(75, 192, 192, 1)"],
                    borderWidth: 1,
                },
            ];
            labelSet = [
                "Decision Trees",
                "Random Forest",
                "Extra Trees",
                "XGBoost",
                "Stacking",
            ];
        } else if (classifier === "LCCDE") {
            lineDataset = [
                {
                    label: "Accuracy",
                    data: [
                        JSON.parse(data)[0][0],
                        JSON.parse(data)[1][0],
                        JSON.parse(data)[2][0],
                        JSON.parse(data)[3][0],
                    ],
                    backgroundColor: ["rgba(255, 99, 132, 0.2)"],
                    borderColor: ["rgba(255, 99, 132, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "Precision",
                    data: [
                        JSON.parse(data)[0][1],
                        JSON.parse(data)[1][1],
                        JSON.parse(data)[2][1],
                        JSON.parse(data)[3][1],
                    ],
                    backgroundColor: ["rgba(54, 162, 235, 0.2)"],
                    borderColor: ["rgba(54, 162, 235, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "Recall",
                    data: [
                        JSON.parse(data)[0][2],
                        JSON.parse(data)[1][2],
                        JSON.parse(data)[2][2],
                        JSON.parse(data)[3][2],
                    ],
                    backgroundColor: ["rgba(255, 206, 86, 0.2)"],
                    borderColor: ["rgba(255, 206, 86, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "F-1 Score",
                    data: [
                        JSON.parse(data)[0][3],
                        JSON.parse(data)[1][3],
                        JSON.parse(data)[2][3],
                        JSON.parse(data)[3][3],
                    ],
                    backgroundColor: ["rgba(75, 192, 192, 0.2)"],
                    borderColor: ["rgba(75, 192, 192, 1)"],
                    borderWidth: 1,
                },
            ];
            labelSet = [
                "LightGBM", 
                "XGBoost", 
                "CatBoost", 
                "Stacking"
            ];
        }

        var myChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labelSet,
                datasets: lineDataset,
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: false,
                    },
                },
                plugins: {
                    legend: {
                      title: {
                        display: true,
                        text: 'Average of Event Metrics by Classifier',
                      }
                    }
                }
            },
        });
    }else if(graphType === 'Callout'){
        console.log(canvas);
        canvas.remove()
        var textContainer = document.createElement('div')
        textContainer.classList.add('newParameterContainer')
        var parameterData = document.createElement('div')
        parameterData.classList.add('parameterData')
        parameterData.innerHTML = parameter
        var newContainer = containerClone.childNodes[1].childNodes[1]
        var textData = document.createElement('div')
        textData.classList.add('callout')
        textData.innerHTML = data
        textContainer.appendChild(textData)
        textContainer.appendChild(parameterData)
        newContainer.appendChild(textContainer)
    }
}

function addLeftHelper(graphType, parameter, data, coord, classifier) {
    console.log(parameter);
    console.log(data);
    newCol = String(coord).split(" ")[1];
    newRow = String(coord).split(" ")[0];

    //Copy the container and give it an ID
    var copiedContainer = document.querySelector("#myContent");
    var containerClone = copiedContainer.cloneNode(true);
    var uniqueId =
        "myChart_" + Date.now() + "_" + Math.floor(Math.random() * 1000);
    containerClone.id = uniqueId;
    containerClone.childNodes[1].childNodes[3].childNodes[1].id = String(
        parseInt(newRow) + 2 + " " + newCol
    );
    containerClone.childNodes[3].childNodes[1].id = String(
        parseInt(newRow) + 2 + " " + newCol
    );

    //Create new col
    var newCol = document.createElement("div");
    newCol.classList.add("cols");
    newCol.appendChild(document.createTextNode(""));
    newCol.appendChild(containerClone);
    newCol.appendChild(document.createTextNode(""));

    //Create new row (TODO row detection)
    var newRow = document.createElement("div");
    newRow.classList.add("rows");
    newRow.appendChild(newCol);
    newRow.appendChild(document.createTextNode(""));

    var upperContent = document.getElementsByClassName("upperContent")[0];
    upperContent.appendChild(newRow);
    upperContent.appendChild(document.createTextNode(""));

    updateBoxes();

    var uniqueCanvasId =
        "myChart_" + Date.now() + "_" + Math.floor(Math.random() * 10000);
    canvas = containerClone.childNodes[1].childNodes[1].childNodes[1];
    canvas.id = uniqueCanvasId;
    var ctx = canvas.getContext("2d");
    if (graphType === "Matrix") {
        const newHeatMapData = {
            labels: [
                "Category 1",
                "Category 2",
                "Category 3",
                "Category 4",
                "Category 5",
                "Category 6",
                "Category 7",
            ],
            datasets: [
                {
                    data: JSON.parse(data),
                    label: "Heatmap Data",
                },
            ],
        };
        const container = document.getElementById("content");
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        ctx.canvas.width = containerWidth;
        ctx.canvas.height = containerHeight;
        console.log(newHeatMapData.datasets[0].data);
        const newFormatData = convertToNewFormat(newHeatMapData.datasets[0].data);
        const minValue = Math.min(...newFormatData.map((value) => value.v));
        const maxValue = Math.max(...newFormatData.map((value) => value.v));
        const colorScale = chroma
            .scale(["#f7fbff", "#04AA6D"])
            .domain([minValue, maxValue]);
        const backgroundColors = newFormatData.map((value) =>
            colorScale(value.v).hex()
        );
        const newDataset = [
            {
                label: "Confusion Matrix",
                data: newFormatData,
                backgroundColor: backgroundColors,
                borderWidth: 1,
                width: ({ chart }) => (chart.chartArea || {}).width / 7 - 1,
                height: ({ chart }) => (chart.chartArea || {}).height / 7 - 1,
            },
        ];
        currentChart = new Chart(ctx, {
            type: "matrix",
            data: {
                labels: heatmapData.labels,
                datasets: newDataset,
            },
            options: {
                plugins: {
                    legend: {
                        title: {
                          display: false,
                          text: '',
                        },
                        labels: {
                            display: false
                        }
                    },
                    tooltip: {
                        callbacks: {
                            title() {
                                return "";
                            },
                            label(context) {
                                const v = context.dataset.data[context.dataIndex];
                                return ["x: " + v.x, "y: " + v.y, "v: " + v.v];
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: {
                            stepSize: 1,
                        },
                        grid: {
                            display: false,
                        },
                    },
                    y: {
                        offset: true,
                        ticks: {
                            stepSize: 1,
                        },
                        grid: {
                            display: false,
                        },
                    },
                },
            },
        });
    } else if (graphType === "Bar") {
        if (classifier === "MTH" || classifier === "Tree-Based") {
            labelSet = [
                "Decision Trees",
                "Random Forest",
                "Extra Trees",
                "XGBoost",
                "Stacking",
            ];
        } else if (classifier === "LCCDE") {
            labelSet = ["LightGBM", "XGBoost", "CatBoost", "Stacking"];
        }
        console.log(data);
        var myChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labelSet,
                datasets: [
                    {
                        label: parameter,
                        data: JSON.parse(data),
                        backgroundColor: [
                            "rgba(255, 99, 132, 0.2)",
                            "rgba(54, 162, 235, 0.2)",
                            "rgba(255, 206, 86, 0.2)",
                            "rgba(75, 192, 192, 0.2)",
                            "rgba(153, 102, 255, 0.2)",
                        ],
                        borderColor: [
                            "rgba(255, 99, 132, 1)",
                            "rgba(54, 162, 235, 1)",
                            "rgba(255, 206, 86, 1)",
                            "rgba(75, 192, 192, 1)",
                            "rgba(153, 102, 255, 1)",
                        ],
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: false,
                    },
                },
            },
        });
    } else if (graphType === "Line") {
        labelSet = [];
        var lineDataset = [];
        if (classifier === "MTH" || classifier === "Tree-Based") {
            lineDataset = [
                {
                    label: "Accuracy",
                    data: [
                        JSON.parse(data)[0][0],
                        JSON.parse(data)[1][0],
                        JSON.parse(data)[2][0],
                        JSON.parse(data)[3][0],
                        JSON.parse(data)[4][0],
                    ],
                    backgroundColor: ["rgba(255, 99, 132, 0.2)"],
                    borderColor: ["rgba(255, 99, 132, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "Precision",
                    data: [
                        JSON.parse(data)[0][1],
                        JSON.parse(data)[1][1],
                        JSON.parse(data)[2][1],
                        JSON.parse(data)[3][1],
                        JSON.parse(data)[4][1],
                    ],
                    backgroundColor: ["rgba(54, 162, 235, 0.2)"],
                    borderColor: ["rgba(54, 162, 235, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "Recall",
                    data: [
                        JSON.parse(data)[0][2],
                        JSON.parse(data)[1][2],
                        JSON.parse(data)[2][2],
                        JSON.parse(data)[3][2],
                        JSON.parse(data)[4][2],
                    ],
                    backgroundColor: ["rgba(255, 206, 86, 0.2)"],
                    borderColor: ["rgba(255, 206, 86, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "F-1 Score",
                    data: [
                        JSON.parse(data)[0][3],
                        JSON.parse(data)[1][3],
                        JSON.parse(data)[2][3],
                        JSON.parse(data)[3][3],
                        JSON.parse(data)[4][3],
                    ],
                    backgroundColor: ["rgba(75, 192, 192, 0.2)"],
                    borderColor: ["rgba(75, 192, 192, 1)"],
                    borderWidth: 1,
                },
            ];
            labelSet = [
                "Decision Trees",
                "Random Forest",
                "Extra Trees",
                "XGBoost",
                "Stacking",
            ];
        } else if (classifier === "LCCDE") {
            lineDataset = [
                {
                    label: "Accuracy",
                    data: [
                        JSON.parse(data)[0][0],
                        JSON.parse(data)[1][0],
                        JSON.parse(data)[2][0],
                        JSON.parse(data)[3][0],
                    ],
                    backgroundColor: ["rgba(255, 99, 132, 0.2)"],
                    borderColor: ["rgba(255, 99, 132, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "Precision",
                    data: [
                        JSON.parse(data)[0][1],
                        JSON.parse(data)[1][1],
                        JSON.parse(data)[2][1],
                        JSON.parse(data)[3][1],
                    ],
                    backgroundColor: ["rgba(54, 162, 235, 0.2)"],
                    borderColor: ["rgba(54, 162, 235, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "Recall",
                    data: [
                        JSON.parse(data)[0][2],
                        JSON.parse(data)[1][2],
                        JSON.parse(data)[2][2],
                        JSON.parse(data)[3][2],
                    ],
                    backgroundColor: ["rgba(255, 206, 86, 0.2)"],
                    borderColor: ["rgba(255, 206, 86, 1)"],
                    borderWidth: 1,
                },
                {
                    label: "F-1 Score",
                    data: [
                        JSON.parse(data)[0][3],
                        JSON.parse(data)[1][3],
                        JSON.parse(data)[2][3],
                        JSON.parse(data)[3][3],
                    ],
                    backgroundColor: ["rgba(75, 192, 192, 0.2)"],
                    borderColor: ["rgba(75, 192, 192, 1)"],
                    borderWidth: 1,
                },
            ];
            labelSet = [
                "LightGBM", 
                "XGBoost", 
                "CatBoost", 
                "Stacking"
            ];
        }

        var myChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labelSet,
                datasets: lineDataset,
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: false,
                    },
                },
                plugins: {
                    legend: {
                      title: {
                        display: true,
                        text: 'Average of Event Metrics by Classifier',
                      }
                    }
                }
            },
        });
    }else if(graphType === 'Callout'){
        console.log(canvas);
        canvas.remove()
        var textContainer = document.createElement('div')
        textContainer.classList.add('newParameterContainer')
        var parameterData = document.createElement('div')
        parameterData.classList.add('parameterData')
        parameterData.innerHTML = parameter
        var newContainer = containerClone.childNodes[1].childNodes[1]
        var textData = document.createElement('div')
        textData.classList.add('callout')
        textData.innerHTML = data
        textContainer.appendChild(textData)
        textContainer.appendChild(parameterData)
        newContainer.appendChild(textContainer)
    }
}

showHeatmap();
updateBoxes();


function copyString(elementId) {
    var text = document.getElementById(elementId).innerText;
    var elem = document.createElement("textarea");
    document.body.appendChild(elem);
    elem.value = text;
    elem.select();
    document.execCommand("copy");
    document.body.removeChild(elem);

    alert("Copied: " + text);
}
