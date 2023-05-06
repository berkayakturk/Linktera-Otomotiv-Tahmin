#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Modelin yüklenmesi
model = joblib.load('linear_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json

    test_data = pd.DataFrame(input_data, index=[0])
    test_data.set_index('Date', inplace=True)

    predictions = predict_future(test_data)
    result = {'prediction': predictions.tolist()}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# In[ ]:




