from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('regression_model.pickle', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/regression_project_report',methods=['GET'])
def eda():
    return render_template('regression_project_report.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        ptemp = float(request.form['ptemp'])
        hdf = int(str(request.form['hdf']).split(':')[0].strip())

        data = pd.DataFrame([ptemp, hdf]).T
        data.columns = ['Process temperature [K]', 'HDF']
        prediction = model.predict(data)[0]

        prediction_text = f'Predicted air temperature is {prediction:.2f} K.'
        return render_template('index.html', prediction_text=prediction_text)
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
        