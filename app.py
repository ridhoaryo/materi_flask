from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/breast_cancer', methods=["POST", "GET"])
def cancer_predict():
    if request.method == "POST":
        input = request.form
        feature = [
            float(input['worstconcave']),
            float(input['worstperim']),
            float(input['meanconcave']),
            float(input['worstradius']),
            float(input['meanperim']),
            float(input['worstarea']),
            float(input['meanradius']),
            float(input['meanarea']),
            float(input['meanconcavity']),
            float(input['worstconcavity'])
        ]

        scaled_feature = scaler.transform([feature])
        feature_pca = pca10.transform(scaled_feature)

        predpca = gr10.predict(feature_pca)[0]
        pred_probapca = gr10.predict_proba(feature_pca)
        endresult = f"{round(np.max(pred_probapca)*100,2)}% {'BENIGN' if predpca == 1 else 'NOT BENIGN'}"


        # pred = gbc.predict([feature])[0]
        # pred_proba = gbc.predict_proba([feature])
        # pred_and_proba = f"{round(np.max(pred_proba)*100,2)}% {'BENIGN' if pred == 1 else 'NOT BENIGN'}"

        return render_template('result.html',
        data=input, prediction=endresult, worstconcave=input['worstconcave'],
        worstperim=input['worstperim'], meanconcave=input['meanconcave'],
        worstradius=input['worstradius'], meanperim=input['meanperim'],
        worstarea=input['worstarea'], meanradius=input['meanradius'],
        meanarea=input['meanarea'], meanconcavity=input['meanconcavity'],
        worstconcavity=input['worstconcavity'])

if __name__ == '__main__':
    gbc = joblib.load('gbc_breast_cancer')
    gr10 = joblib.load("gr10") # load model ML GBC pca 2
    scaler = joblib.load("scalerbreast10") # load si scaler
    pca10 = joblib.load("pcabreast10") # load si pca

    app.run(debug=True, port=4000)