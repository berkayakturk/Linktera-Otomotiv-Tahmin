#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_excel("linktera_veri_bilimi_veri_seti.xlsx")
df = pd.DataFrame(df)
df['Date'] = pd.to_datetime(df.Date, format = '%d/%m/%Y')
df.dropna(inplace=True)
# Tarih sütununu pandas datetime nesnesine dönüştürme
df['Date'] = pd.to_datetime(df['Date'])

# Kukla değişkenini oluşturma
df['Donem'] = df['Date'].dt.month.apply(lambda x: 1 if 1 <= x <= 4 else (2 if 5 <= x <= 8 else 3))
df['Date'] = pd.to_datetime(df['Date'])
X = df[['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok', 'Donem']]
y = df['Otomotiv Satis']

reg_full = LinearRegression()
reg_full.fit(X, y)

# Modeli kullanarak gelecekteki test verileri için tahminlerde bulunma
def predict_future(test_data):
    # Test verilerini logaritmik dönüşüme uygulama
    test_data_log = np.log1p(test_data)

    # Modeli kullanarak tahminlerde bulunma
    predictions_log = reg_full.predict(test_data_log)

    # Tahminleri ters dönüşüme uygulama
    predictions = np.expm1(predictions_log)

    return predictions
import joblib

joblib.dump(reg_full, 'linear_regression_model.pkl')


# In[ ]:




