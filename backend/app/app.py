from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/sendData')
def sendData():
    labels = [ 
            'January', 
            'February',  
            'March',  
            'April',  
            'May',  
            'June',  
            'July'
            ] 
    chartLabel = "my data"
    chartdata = [0, 10, 5, 2, 20, 30, 45] 
    data ={ 
                    "labels":labels, 
                    "chartLabel":chartLabel, 
                    "chartdata":chartdata, 
            } 
    return data

if __name__ == '__main__':
    app.run(debug=True)