from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scalar=pickle.load(open('models/scaler.pkl','rb'))

@application.route("/")
def index():
    return render_template('index.html')

@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        # 1. Grab data from the HTML form fields (matching 'name' attributes in HTML)
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        # 2. Scale the input data using the loaded scaler
        # (Ensure this order exactly matches the order of features during your model training)
        new_data_scaled = standard_scalar.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])

        # 3. Predict the FWI using the Ridge model
        prediction = ridge_model.predict(new_data_scaled)

        # 4. Format prediction (round it to 2 decimal places)
        results = round(prediction[0], 2)

        # 5. Render the page again, passing the results back to the template
        return render_template('form.html', results=results)
    else:
        return render_template('form.html')




if __name__=="__main__":
    application.run(host="0.0.0.0",debug=True)