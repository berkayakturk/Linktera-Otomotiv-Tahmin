#!/usr/bin/env python
# coding: utf-8

# # Linktera Otomotiv Satış Tahmini

# In[1]:


# Veri manipülasyonu
import pandas as pd
import requests
import numpy as np


# Veri görselleştirme, İstatistik
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.compat import lzip
from statsmodels.graphics.gofplots import ProbPlot

# Regresyon
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ## Veri İnceleme

# In[2]:


df = pd.read_excel("linktera_veri_bilimi_veri_seti.xlsx")
df.head()


# In[3]:


# Değişken tipleri
df = pd.DataFrame(df)
df['Date'] = pd.to_datetime(df.Date, format = '%d/%m/%Y')
df.info()


# In[4]:


df.head(10)


# Temel istatistikleri aşağıda inceleyebiliriz.

# In[5]:


df.describe()


# In[6]:


# Boş değer kontrolü
df.isnull().sum()


# In[7]:


# Eksik değer kontrolü
df.isna().any()


# In[8]:


# Veri setinde tahmin edilmesi istenen tarihler mevcut olduğundan son 13 gözlem veriden çıkartılır.
df.dropna(inplace=True)


# In[9]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 18))

# Otomotiv Satis
axes[0, 0].plot(df['Date'], df['Otomotiv Satis'])
axes[0, 0].set_title("Otomotiv Satış (2010-2022)")
axes[0, 0].set_xlabel("Tarih")
axes[0, 0].set_ylabel("Otomotiv Satış")

# OTV Orani
axes[0, 1].plot(df['Date'], df['OTV Orani'])
axes[0, 1].set_title("OTV Oranı (2010-2022)")
axes[0, 1].set_xlabel("Tarih")
axes[0, 1].set_ylabel("OTV Oranı")

# Faiz
axes[1, 0].plot(df['Date'], df['Faiz'])
axes[1, 0].set_title("Faiz (2010-2022)")
axes[1, 0].set_xlabel("Tarih")
axes[1, 0].set_ylabel("Faiz")

# EUR/TL
axes[1, 1].plot(df['Date'], df['EUR/TL'])
axes[1, 1].set_title("EUR/TL (2010-2022)")
axes[1, 1].set_xlabel("Tarih")
axes[1, 1].set_ylabel("EUR/TL")

# Kredi Stok
axes[2, 0].plot(df['Date'], df['Kredi Stok'])
axes[2, 0].set_title("Kredi Stok (2010-2022)")
axes[2, 0].set_xlabel("Tarih")
axes[2, 0].set_ylabel("Kredi Stok")

# Son grafik boş
axes[2, 1].axis('off')

plt.tight_layout()
plt.show()


# Değişkenlerin zamana bağlı değişimini yukarıda inceleyebiliriz. 2018 yılından sonra gerçekleşen artışlar otomotiv satışlarındaki sezonsal trendi olduça etkilemiş görünüyor.

# In[10]:




fig = go.Figure()

# Otomotiv Satis
fig.add_trace(go.Scatter(x=df['Date'], y=df['Otomotiv Satis'], name="Otomotiv Satis", yaxis="y1"))

# EUR/TL
fig.add_trace(go.Scatter(x=df['Date'], y=df['EUR/TL'], name="EUR/TL", yaxis="y2"))

fig.update_layout(
    title="Aylık Verilerin Çift Y Eksenli Çizgi Grafiği (2010-2022)",
    xaxis=dict(title="Tarih"),
    yaxis=dict(title="Otomotiv Satış", side="left"),
    yaxis2=dict(title="EUR/TL", side="right", overlaying="y", anchor="x"),
    legend=dict(x=1.1, y=1, orientation="v"),
)

fig.show()


# EUR/TL'nin 2019 yılındaki artışı ile birlikte otomotiv satışlarında beklenen yükseliş yakalanamamış ve yıl sonunda gerçekleşen yüksek otomotiv satışı bir daha gerçekleşememiştir.  

# In[11]:


# Lokasyona göre gruplama ve son gözlemleri alma
cor = df.drop(columns=['Date'])

# Korelasyon matrisi hesaplama
corr = cor.corr()

# Korelasyon ısı haritası oluşturma
plt.figure(figsize=(10, 8))

sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=.5)

plt.title('Değişkenler Arasındaki Korelasyon Isı Haritası')

plt.show()


# Otomotiv satışını etkileyen en güçlü değişken yüzdelik faiz değerleridir. Faizin artışı otomotiv satışlarını ciddi bir şekilde azaltmaktadır. Diğer değişkenler incelendiğinde ise en güçlü korelasyonu kredi stok ile EUR/TL arasında görmekteyiz. EUR/TL'nin artışı kredi stoklarını ciddi bir şekilde arttırmaktadır.

# In[12]:


df['Date'] = pd.to_datetime(df['Date'])

# Yıllık toplam ve ortalama değerleri hesaplayın
df['Year'] = df['Date'].dt.year
yearly_totals = df.groupby('Year')['Otomotiv Satis'].sum().reset_index()
yearly_averages = df.groupby('Year')['Otomotiv Satis'].mean().reset_index()

# Histogram grafiğini plotly ile çizdirme
fig = go.Figure()

fig.add_trace(go.Bar(x=yearly_totals['Year'], y=yearly_totals['Otomotiv Satis'],
                     name='Yıllık Toplam Otomotiv Satis', marker_color='blue'))

fig.add_trace(go.Scatter(x=yearly_averages['Year'], y=yearly_averages['Otomotiv Satis'],
                         mode='lines+markers', name='Yıllık Ortalama Otomotiv Satis', marker_color='red'))

fig.update_layout(title='Yıllık Toplam ve Ortalama Otomotiv Satisları (2010-2022)',
                  xaxis_title='Yıl', yaxis_title='Otomotiv Satis')

fig.show()


# Otomotiv satış değerlerini yıllık bazda toplarsak 2018 ve 2019 yılında ciddi bir düşüşün gerçekleştiğini görebiliriz. 2022 yılında otomotiv satış değerlerinin hepsine sahip olmadığımız için ortalama değerler üzerinden yorum yapabiliriz. Mayıs ayına kadar gerçekleşen otomotiv satışları, diğer yıllara göre (2020-2021) benzer şekilde ilerlemektedir.

# In[13]:


# Yukarıdaki grafiği çizebilmek için yıl değişkeni eklendi. Veri setinden çıkartılması gerekmektedir.

df = df.drop('Year', axis=1)


# ## Regresyon Analizi

# Regresyon analizini gerçekleştirebilmek için bazı varsayımların sağlanması gerekmektedir. Ancak varsayımlardan önce ham verinin otomotiv satışlarına etkisi aşağıdaki gibi incelenebilir.

# In[14]:


df['Date'] = pd.to_datetime(df['Date'])

X = df[['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok']]
y = df['Otomotiv Satis']

# Çok değişkenli regresyon modeli oluşturma
model = sm.OLS(y, X)

# Modeli eğitme
results = model.fit()

# Etki değerlerini ve p-değerlerini gösterme
print("Etki Değerleri (Katsayılar):\n", results.params)
print("\nP-değerleri:\n", results.pvalues)

# Modelin özeti
print("\nModel Özeti:\n", results.summary())


# Özetle yukarıdaki sonuçlar, klasik çok değişkenli regresyon modeli (OTV Orani, Faiz, EUR/TL ve Kredi Stok) ile Otomotiv Satış bağımlı değişkenini tahmin etmeye çalıştık ve bağımsız değişkenlerdeki değişimin %88.8'ini (R-squared) açıkladığını gördük. Bu, bağımsız değişkenlerin Otomotiv Satış üzerinde önemli bir etkisi olduğunu gösterir. Bağımsız değişkenler otomotiv satışının %88.8'ini açıklayabilmektedir.
# 
# Ancak, EUR/TL ve Kredi Stok değişkenlerinin %95 güven düzeyinde istatistiksel olarak anlamlı olmadığı ortaya çıktı. Bu durum, bu değişkenlerin modelde yer almasının Otomotiv Satış üzerindeki tahmin performansını düşürebileceğini gösterir. Bu değişkenlerin modelden çıkarılması ve daha düşük AIC ve BIC değerlerine sahip başka bir modelin denenmesi düşünülebilir.
# 
# Durbin-Watson değeri artıkların pozitif yönlü otokorelasyona sahip olduğunu gösterir. Bu durum modelin performansını ve tahmin gücünü düşürebilir. Bu sorunu çözmek için zaman serisi analizinde farklı regresyon tekniklerinin uygulanabileceğini gösterir.
# 
# Sonuç olarak, şu anki model anlamlı değişkenlerle bir miktar başarı sağlamış olsa da, modelin performansını artırmak ve daha doğru tahminler elde etmek için değişken seçimi, otokorelasyon ve durağanlık sorunlarına daha fazla dikkat etmek önemlidir. Bu amaçla, daha uygun regresyon modelleri ve zaman serisi analiz teknikleri kullanılabilir.

# In[15]:


result = adfuller(df['Otomotiv Satis'])
print('ADF istatistiği: ', result[0])
print('p-value: ', result[1])
print('Kritik değerler: ', result[4])


# Augmented Dickey-Fuller (ADF) testi, zaman serisinin durağan olup olmadığını test etmek için kullanılır. Durağanlık, zaman serisinin ortalama ve varyansının zamanla sabit olduğunu ve otokorelasyon yapısının zamanla değişmediğini gösterir. Augmented Dickey-Fuller testi incelendiğinde ADF test değeri %95 güven düzeyinden büyük olduğu için Otomotiv Satışı durağan değildir.

# In[16]:


# Calculate VIF for each variable
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif)


# Çıktıda, tüm bağımsız değişkenlerin VIF değerleri 5'ten çok daha yüksektir. Bu, bağımsız değişkenler arasında ciddi bir çoklu bağlantı sorunu olduğunu gösterir. Özellikle, "EUR/TL" ve "Kredi Stok" değişkenleri arasındaki çoklu bağlantı sorunu daha yüksektir. Bu durum, regresyon modelinin doğruluğunu ve güvenilirliğini etkileyebilir. Çözüm olarak modelin anlamlılığını bozan değişkenler çıkarılabilir ya da PCA ve Ridge regresyon yöntemlerine başvurulabilir.

# In[17]:


df['Date'] = pd.to_datetime(df['Date'])

# Bağımlı değişkenin birinci farkı için yeni bir sütun
df['Diff_Otomotiv_Satis'] = df['Otomotiv Satis'].diff()

# Bağımlı değişkenin ve bağımsız değişkenlerin birinci farkıyla yeni bir veri çerçevesi
df_diff = pd.concat([df['Diff_Otomotiv_Satis'], X], axis=1)

df_diff = df_diff.dropna()

# GridSpec nesnesi oluşturma
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 2])

fig = plt.figure(figsize=(12, 8))

# Otomotiv Satis'in ilk farkı
ax1 = plt.subplot(gs[0, :])
ax1.plot(df_diff['Diff_Otomotiv_Satis'])
ax1.set_title('Otomotiv Satışının Birinci Farkı')
ax1.set_xlabel('Date')
ax1.set_ylabel('Difference')

# Otomotiv Satis'in birinci farkının otokorelasyonunu
ax2 = plt.subplot(gs[1, 0])
sm.graphics.tsa.plot_acf(df_diff['Diff_Otomotiv_Satis'], lags=50, ax=ax2)
ax2.set_title('Otomotiv Satışının Birinci Farkının Otokorelasyonu')

# Otomotiv Satis'in birinci farkının kısmi otokorelasyonu
ax3 = plt.subplot(gs[1, 1])
sm.graphics.tsa.plot_pacf(df_diff['Diff_Otomotiv_Satis'], lags=50, ax=ax3)
ax3.set_title('Otomotiv Satışının Birinci Farkının Kısmi Otokorelasyonu')

# Düzeni ayarla
plt.tight_layout()
plt.show()


# Grafiksel gösterimde pozitif yönlü otokorelasyonun olduğu ve yaklaşık en az 10 farklı gecikmede bu problemin olduğu söylenebilir. Özellikle ilk grafik incelendiğinde Otomotiv Satışının son zamanlarında durağanlığın ciddi şekilde bozulduğu görülebilmektedir. Bu problemi çözebilmek için oldukça fazla çözüm yolu mevcuttur. Zaman serisinin farkını almak, zaman serisindeki trendi kaldırmak, ARIMA modellerine başvurmak çözüm yollarından bazıları olabilir. Ancak projenin devamı, çok değişkenli regresyon analizi ile devam edilmesi istendiği için, yukarıdaki tüm problemlerin çözümü için üç farklı yol izlenebilir.
# 
# 1: İstatistiksel olarak regresyon analizi uygulamasında anlamlılığı bozan değişkenler çıkartılarak analizler gerçekleştirilebilir.
# 
# 2: Tahmin modeli için Ridge, Lasso ve Elastic Net regresyon yöntemleri karşılaştırılabilir.
# 
# 3: Zaman serisi için bir kukla değişken atanabilir.
# 
# Son olarak, tahmin modelinde daha iyi sonuçlar alabilmek için logaritmik dönüşüm uygulanabilir. İlk olarak mevsimselliği kontrol edelim.

# In[18]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[19]:


# Frekans parametresini belirtin, örneğin aylık veri için 12
frekans = 12

decomposition = sm.tsa.seasonal_decompose(df['Otomotiv Satis'], model='additive', period=frekans)
fig = decomposition.plot()
plt.show()


# Seasonal grafiği incelendiğinde, ekim, kasım ve aralık aylarında otomotiv satışlarında belirgin bir mevsimsellik olduğu görülmektedir. Aralık aylarında artan satışlar, ocak ayı itibariyle düşüşe geçmektedir. Bu durum, çift Y eksenli grafik incelendiğinde enflasyon ve EUR/TL değerlerinin artışı ile bozulmuş olsa da, eylül, ekim, kasım ve aralık aylarının diğer aylara göre otomotiv satışlarının oldukça yüksek olduğu görülmektedir. Bu durum incelendiğinde, ocak, şubat, mart ve nisan aylarına 1, diğer aylara 0 değerini atayarak kukla değişkeni oluşturulabilir. Bu şekilde zamana bağlı değişimin etkisi otomotiv satışlarına göre incelenebilir. Bu yaklaşım, modelin mevsimsel etkileri dikkate alarak daha doğru tahminler yapmasına yardımcı olabilir.

# In[20]:


# Veri setini tekrar çekelim.
df = pd.read_excel("linktera_veri_bilimi_veri_seti.xlsx")
df = pd.DataFrame(df)
df['Date'] = pd.to_datetime(df.Date, format = '%d/%m/%Y')
df.dropna(inplace=True)


# Yukarıda bahsi geçen kukla değişkeni atayalım.

# In[21]:


# Tarih sütununu pandas datetime nesnesine dönüştürme
df['Date'] = pd.to_datetime(df['Date'])

# Kukla değişkenini oluşturma
df['Donem'] = df['Date'].dt.month.apply(lambda x: 1 if 9 <= x <= 12  else 0)

df.head()


# Regresyon modelini tekrar kuralım. Ancak EUR/TL ve Kredi stok değişkenlerini çıkartılacak ve kukla değişkeni eklenecek.

# In[22]:


df['Date'] = pd.to_datetime(df['Date'])

X = df[['OTV Orani', 'Faiz', 'Donem']]
y = df['Otomotiv Satis']

# Çok değişkenli regresyon modeli oluşturma
model = sm.OLS(y, X)

# Modeli eğitme
results = model.fit()

# Etki değerlerini ve p-değerlerini gösterme
print("Etki Değerleri (Katsayılar):\n", results.params)
print("\nP-değerleri:\n", results.pvalues)

# Modelin özeti
print("\nModel Özeti:\n", results.summary())


# In[23]:


result = adfuller(df['Otomotiv Satis'])
print('ADF istatistiği: ', result[0])
print('p-value: ', result[1])
print('Kritik değerler: ', result[4])


# In[24]:


# Calculate VIF for each variable
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif)


# In[25]:


bg_test = acorr_breusch_godfrey(results, nlags=1)
print(f'Breusch-Godfrey Test İstatistiği: {bg_test[0]}')
print(f'Breusch-Godfrey Test p-değeri: {bg_test[1]}')

bp_test = het_breuschpagan(results.resid, results.model.exog)
labels = ['LM İstatistiği', 'LM p-değeri', 'F İstatistiği', 'F p-değeri']
results_table = lzip(labels, bp_test)
print(results_table)


# Regresyon modelinin analizi sonucunda, AIC, BIC ve Jarque-Bera (JB) değerlerinde önemli bir azalma tespit edilmiştir. Bu azalma, otokorelasyon sorununun hafifletildiğini ve modelin açıklama gücünün arttığını göstermektedir. Ayrıca, ADF istatistiği %95 güven düzeyinde istatistiksel olarak anlamlı bulunmuştur. Kukla değişken kullanılarak otomotiv satışlarındaki durağanlık sağlanmıştır. Varyans Şişme Faktörü (VIF) değerine göre, kukla değişkenin çoklu bağlantı sorunu oluşturmadığı görülmektedir (1.45). Hata terimleri arasında otokorelasyon problemini ise Durbin-Watson testinin verdiği sonuca göre hafifletilmiş olsada Breusch-Godfrey testi problemin devam ettiğini söylemektedir. Breusch-Pagan testinde yer alan her iki p-değeri de (LM p-değeri ve F p-değeri) %5 (0.05) anlamlılık düzeyinden çok daha düşüktür. Bu, modelde heteroskedastisite (değişen varyans) olduğunu gösterir. Heteroskedastisite varsa, OLS tahminlerinin verimliliği düşer ve standart hataların yanlı olduğu söylenir. Bu iki sorunun çözümü için Ağırlıklı En Küçük Kareler (AEKK) yöntemi veya ARİMA modellerine başvurulabilir. Çalışma gereği bu işlemler yapılmamaktadır. Ancak bu çalışmanın devamı için logaritmik fark almak veya Lasso gibi cezalandırma yöntemleri yer almaktadır.
# 
# Mevsimsel etkiler ve zamanla değişen etkileri dikkate alan bu regresyon modeli, otomotiv satışlarını tahmin etmede daha başarılıdır. Model, otokorelasyon, durağanlık ve çoklu bağlantı problemlerini tam olarak çözüm sunmasada, tahmin performansını artırmıştır. Bu sonuçlar doğrultusunda, bağımsız değişkenlerin etkileri incelenebilir.
# 
# OTV Oranı'nın katsayısı 2080.9854 olup, otomotiv satışları üzerinde pozitif bir etkiye sahiptir. Bu, OTV Oranı'ndaki bir birimlik artışın otomotiv satışlarını 2080.9854 birim artıracağı anlamına gelir. Bu durum OTV oranıyla pazarlanan arabalardan kazanç sağlanabileceğini göstermektedir. Faiz'in katsayısı ise -2107.8622 olup, otomotiv satışları üzerinde negatif bir etkiye sahiptir. Bu durum, faiz oranlarındaki bir birimlik artışın otomotiv satışlarını 2107.8622 birim azaltacağı anlamına gelir. Son olarak, dönem değişkeninin katsayısı 2.489e+04 olup, otomotiv satışları üzerinde pozitif bir etkiye sahiptir. Bu, dönem değişkenindeki bir birimlik artışın, yani eylül ayı ile aralık ayına kadar olan sürecin, otomotiv satışlarını yüzde 2.489e+04 birim artırdığı anlamına gelir.
# 

# ## Kukla değişkeni ile Regresyon Analizi Tahmin Modeli

# Regresyon etki değerleri incelendikten sonra gelecek 1 yılın tahmini gerçekleştirilebilir. Ancak yukarıda gerçekleştirilen varsayımların sağlanmadığı ve bu durumun başarılı sonuçlar üretmeyeceği söylenebilir. Çözüm olarak genelleştirilmiş en küçük kareler (UGEKK), Cochrane-Orcutt (CO) dönüşümü, Prais-Winsten (PW) dönüşümü gibi teknikler uygulanarak daha başarılı sonuçlar elde edilebilir. Çalışma gereği bu tekniklerden sadece logaritmik fark alma uygulanmıştır ve Lasso, Ridge, Elastic Net gibi cezalandırma yöntemleri karşılaştırılmıştır.

# In[26]:



# Tarih sütununu pandas datetime nesnesine dönüştürme
df['Date'] = pd.to_datetime(df['Date'])

X = df[['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok', 'Donem']]
y = df['Otomotiv Satis']

# Verileri eğitim ve test setlerine bölebilirsiniz:
test_index = int(len(df) * 0.8)

# Veri setini eğitim ve test verisi olarak ayır
X_train = X.iloc[:test_index]
X_test = X.iloc[test_index:]
y_train = y.iloc[:test_index]
y_test = y.iloc[test_index:]

# Çok değişkenli regresyon modeli oluşturma
reg = LinearRegression()

# Modeli eğitme
reg.fit(X_train, y_train)

# Eğitim verileri üzerinde tahmin
y_train_pred = reg.predict(X_train)

# Test verileri üzerinde tahmin
y_test_pred = reg.predict(X_test)

# RMSE değerlerini hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Eğitim verisi RMSE: {train_rmse}")
print(f"Test verisi RMSE: {test_rmse}")


# In[27]:


plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test, label="Gerçek Değerler", marker='o')
plt.plot(y_test.index, y_test_pred, label="Tahminler", marker='x')
plt.title("Otomotiv Satis: Gerçek Değerler ve Tahminler (Test Verisi)")
plt.xlabel("Index")
plt.ylabel("Otomotiv Satis")
plt.legend()
plt.show()


# Eğitim ve test verileri için hesaplanan RMSE (Kök Ortalama Kare Hata) değerlerini karşılaştırarak, modelinizin performansını yorumlayabilirsiniz. Bu sonuçlara göre, eğitim verisi üzerinde modelin hata oranı düşükken (19975.78), test verisi üzerinde hata oranı oldukça yüksek (75408.41). Bu durum, modelin eğitim verisine aşırı uyum sağladığını ve yeni verilere genelleme yapamadığını gösterir. Bu durum, genellikle overfitting (aşırı uyum) olarak adlandırılır. Overfitting sorunun çözümü için oldukça fazla yöntem bulunmaktadır. Lasso (L1) veya Ridge (L2) düzenlileştirmesi gibi tekniklerle modelin aşırı uyumunu önleyebilir. Logaritmik dönüşümler daha başarılı tahmin modeli önermemizi sağlayabilir. Farklı tahmin yöntemleri uygulanabilir (SVM, KNN, ANN, CART, RF, XGBoost). Veri çerçevesinin test train ayrımı düzeltilebilir veya veri çerçevesinin gözlem sayıları arttırılabilir. Ancak test verisinin tahmin modeli incelendiğinde 140. indexten sonra tahminler negatif değer almaktadır. Bu sorunun çözümü Lasso, Ridge ve Elastic Net Regresyonlarıyla ya da logaritmik dönüşüm yoluyla çözülebilir. Son olarak cross validation uygulamak küçük ölçekli ve zaman serisi verilerinde başarılı sonuç vermesede (test-train ayrımı rastgele yapılmamaktadır, test verisi son %20'lik gözlemi temsil eder) karşılaştırmak için iyi bir yol olabilir.

# ## Regresyon Analizi ile Lasso, Ridge ve Elastic Net Regresyonlarının Farklı Parametre Değerleri ile Karşılaştırılması ve 10 Katlı Cross-Validation Yöntemi ile Tahmini  

# Farklı parametrelerde 3 yöntemi ve klasik regresyon yöntemini karşılaştıralım ve en doğru sonucu veren yöntemi seçelim. Gerekli varsayımlar sağlanmadığı için overfitting sorunu yaşadığımızı unutmayalım.

# In[28]:


# Veri setini ayırma
X = df[['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok', 'Donem']]
y = df['Otomotiv Satis']

# Verileri eğitim ve test setlerine bölebilirsiniz:
test_index = int(len(df) * 0.8)

# Veri setini eğitim ve test verisi olarak ayır
X_train = X.iloc[:test_index]
X_test = X.iloc[test_index:]
y_train = y.iloc[:test_index]
y_test = y.iloc[test_index:]


# In[29]:


# 10 katlı cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# Parametre aralıkları
param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
param_grid_ridge = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
param_grid_elasticnet = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                         'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

# Model tanımlamaları
linear_reg = LinearRegression()
lasso_reg = Lasso()
ridge_reg = Ridge()
elastic_net = ElasticNet()

# GridSearchCV nesneleri
grid_search_lasso = GridSearchCV(lasso_reg, param_grid_lasso, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=kf)
grid_search_ridge = GridSearchCV(ridge_reg, param_grid_ridge, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=kf)
grid_search_elasticnet = GridSearchCV(elastic_net, param_grid_elasticnet, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=kf)

# Grid search fit
linear_reg.fit(X_train, y_train)
grid_search_lasso.fit(X_train, y_train)
grid_search_ridge.fit(X_train, y_train)
grid_search_elasticnet.fit(X_train, y_train)


# In[30]:


# En iyi parametreler ve en iyi RMSE
best_linear = linear_reg
best_linear_rmse = np.sqrt(mean_squared_error(y_train, best_linear.predict(X_train)))
best_lasso = grid_search_lasso.best_estimator_
best_lasso_rmse = np.sqrt(np.abs(grid_search_lasso.best_score_))
best_ridge = grid_search_ridge.best_estimator_
best_ridge_rmse = np.sqrt(np.abs(grid_search_ridge.best_score_))
best_elasticnet = grid_search_elasticnet.best_estimator_
best_elasticnet_rmse = np.sqrt(np.abs(grid_search_elasticnet.best_score_))

# En iyi modeli seçme
best_model = min([(best_linear, best_linear_rmse),
                  (best_lasso, best_lasso_rmse),
                  (best_ridge, best_ridge_rmse),
                  (best_elasticnet, best_elasticnet_rmse)],
                 key=lambda x: x[1])[0]

# Test verisi üzerinde tahmin
y_pred_test = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# En iyi modelin adını ve RMSE değerini yazdırma
model_name = {best_linear: 'Linear Regression',
              best_lasso: 'Lasso Regression',
              best_ridge: 'Ridge Regression',
              best_elasticnet: 'ElasticNet Regression'}

print(f"En iyi model: {model_name[best_model]}")
print(f"En iyi modelin RMSE değeri: {min(best_linear_rmse, best_lasso_rmse, best_ridge_rmse, best_elasticnet_rmse)}")


# Belirttiğimiz koşullar altında, 4 farklı yöntem arasında en iyi eğitim performansını Lineer Regresyon yöntemi sağlamıştır. Ancak negatif tahmin problemi aşağıdaki gibi devam etmektedir.

# In[31]:


plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test, label="Gerçek Değerler", marker='o')
plt.plot(y_test.index, y_pred_test, label="Tahminler", marker='x')
plt.title("Otomotiv Satis: Gerçek Değerler ve Tahminler (Test Verisi)")
plt.xlabel("Index")
plt.ylabel("Otomotiv Satis")
plt.legend()
plt.show()


# In[32]:


mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)

print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('Mean Absolute Error (MAE):', mae)


# Overfitting sorunu devam ettiğine göre, başka önlemler alarak modelimizi iyileştirmeye çalışabiliriz. Çünkü Lasso, Ridge ve Elastic Net gibi yöntemler veri büyüklüğünün yetersiz oluşundan tam olarak yardımcı olamadı. Logaritmik dönüşüm, verinin dağılımını daha düzgün ve normalleşmiş hale getirerek modelin performansını artırmaya yardımcı olabilecek bir yöntem olduğundan bu yolu deneyebiliriz.

# ## Overfitting Sorunu için Logaritmik Dönüşüm Yolu

# In[33]:


df['Date'] = pd.to_datetime(df['Date'])

# Logaritmik dönüşüm uygulayalım
df[['Otomotiv Satis', 'OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok', 'Donem']] = np.log1p(df[['Otomotiv Satis', 'OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok', 'Donem']])

X = df[['OTV Orani', 'Faiz', 'EUR/TL', 'Kredi Stok','Donem']]
y = df['Otomotiv Satis']

# Verileri eğitim ve test setlerine bölebilirsiniz:
test_index = int(len(df) * 0.8)

# Veri setini eğitim ve test verisi olarak ayır
X_train = X.iloc[:test_index]
X_test = X.iloc[test_index:]
y_train = y.iloc[:test_index]
y_test = y.iloc[test_index:]

# Çok değişkenli regresyon modeli oluşturma
reg = LinearRegression()

# Modeli eğitme
reg.fit(X_train, y_train)

# Eğitim verileri üzerinde tahmin
y_train_pred = reg.predict(X_train)

# Test verileri üzerinde tahmin
y_test_pred = reg.predict(X_test)

# RMSE değerlerini hesaplama
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Eğitim verisi RMSE (log dönüşümlü): {train_rmse}")
print(f"Test verisi RMSE (log dönüşümlü): {test_rmse}")


# In[34]:


# Ters dönüşüm uygulayalım
y_train_pred_inv = np.expm1(y_train_pred)
y_test_pred_inv = np.expm1(y_test_pred)
y_train_inv = np.expm1(y_train)
y_test_inv = np.expm1(y_test)

# RMSE değerlerini hesaplama (ters dönüşümlü)
train_rmse_inv = np.sqrt(mean_squared_error(y_train_inv, y_train_pred_inv))
test_rmse_inv = np.sqrt(mean_squared_error(y_test_inv, y_test_pred_inv))

print(f"Eğitim verisi RMSE (ters dönüşümlü): {train_rmse_inv}")
print(f"Test verisi RMSE (ters dönüşümlü): {test_rmse_inv}")


# In[35]:


plt.figure(figsize=(15, 6))
plt.plot(y_test_inv.index, y_test_inv, label="Gerçek Değerler", marker='o')
plt.plot(y_test_inv.index, y_test_pred_inv, label="Tahminler", marker='x')
plt.title("Otomotiv Satis: Gerçek Değerler ve Tahminler (Test Verisi)")
plt.xlabel("Index")
plt.ylabel("Otomotiv Satis")
plt.legend()
plt.show()


# In[36]:


mse = mean_squared_error(y_test_inv, y_test_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_test_pred_inv)

print('Mean Squared Error (MSE):', mse)
print('Root Mean Squared Error (RMSE):', rmse)
print('Mean Absolute Error (MAE):', mae)


# Logaritmik dönüşüm, yaşanan negatif tahmin problemini çözmeye ve daha başarılı test tahmin değerleri elde etmeye yardımcı oldu. Bu dönüşüm, verilerin dağılımını düzelterek ve doğrusal olmayan ilişkileri düzleştirerek regresyon modelinin performansını iyileştirir. Overfitting sorununun ortadan kalkması, modelin hem eğitim hem de test verisi üzerinde daha iyi genelleme yapabilmesini sağlar.
# 
# Test verisinde yer alan son ayların düşük seviyede tahmin edilmesi, şu etkenlere bağlı olabilir:
# 
# Türkiye'de yaşanan ekonomik daralma nedeniyle, daha önce görülmemiş düzeyde faiz ve EUR/TL artışı yaşanmaktadır. Bu durum, eğitim verisine dayalı olarak öğrenilen regresyon modelinin, test verisinde düşük seviyede otomotiv satışlarını öngörmesine yol açar. Normal koşullarda, OTV oranı, kredi stoklarındaki artış ve diğer etkenler otomotiv satışlarının düşmesine neden olurken, yaşanan ekonomik daralma ve döviz kuru artışı gibi faktörler bu değişkenlerin öngörülen etkisini geçersiz kılar ve otomotiv satışlarının daha düşük seviyelere gerilemesine neden olur.
# 
# Bu durumda, modelinizi güncellemek veya ekonomik kriz dönemlerinde geçerli olabilecek farklı değişkenler ve ilişkileri dikkate alacak şekilde yeniden yapılandırmak önemlidir. Ayrıca, ekonomik belirsizliğin yüksek olduğu dönemlerde daha sık model güncellemeleri yaparak değişen koşullara daha hızlı uyum sağlamak önemli bir strateji olacaktır. Bu şekilde, modelimiz ekonomik daralmayla ilişkili olarak ortaya çıkan yeni faktörleri ve ilişkileri daha doğru bir şekilde tahmin edebilir.
# 
# Test verisinde bağımsız değişkenlerin artışına rağmen otomobil satışlarında ciddi bir azalma gözlemlenmemesi, regresyon modelinin dikkate almadığı bazı faktörlerin etkisine bağlı olabilir. Bu faktörler hükümet politikaları, kredi koşulları, yakıt fiyatları ve ikinci el piyasasındaki düzensiz otomotiv satış politikaları gibi değişkenler olabilir. Eğer bu etkenler modele eklenirse daha başarılı tahminler üretilebilir.
# 
# Bu çıkarımlardan sonra oluşturulan model tüm veriye öğretilebilir ve istenen koşullara göre (06.2022-06.2023) otomotiv satışını tahmin edilebilir.

# In[37]:


# Tüm veriyi kullanarak modeli eğitme
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


# İstenen tarihler arasında bizim belirlediğimiz bağımsız değişkenlerin değerlerini oluşturalım ve otomotiv satışını tahmin edelim. Dönem değişkeni kukla değişkeni olduğu için kukla değişkeni atarken belirlediğimiz koşullar altında değer belirlememiz gerektiğini unutmayalım.

# In[38]:


# Test verisi oluşturma
future_dates = pd.to_datetime([
    '2022-06-01', '2022-07-01', '2022-08-01', '2022-09-01',
    '2022-10-01', '2022-11-01', '2022-12-01', '2023-01-01',
    '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01',
    '2023-06-01'])

future_test_data = pd.DataFrame({
    'Date': future_dates,
    'OTV Orani': [65, 65, 65, 65, 65, 65, 65, 60, 60, 60, 60, 60, 60],
    'Faiz': [23.2, 23.5, 22.7, 22.9, 22.1, 22.5, 21.2, 21.5, 20.7, 20.9, 20.1, 19.5, 19.2],
    'EUR/TL': [17.4, 18.3, 18.29, 18.12, 18.37, 19.34, 20.01, 20.42, 19.96, 20.76, 21.43, 21.50, 21.90],
    'Kredi Stok': [441244, 451940, 461307, 473575, 487708, 480000, 490000, 490000, 500000, 500000, 500000, 510000, 510000],
    'Donem': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
})

future_test_data.set_index('Date', inplace=True)

# Tahminleri alın
future_predictions = predict_future(future_test_data)

print("Gelecekteki Otomotiv Satis verisi için tahminler:")
print(future_predictions)


# In[39]:


predictions = future_predictions

future_dates = future_dates

predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Otomotiv Satis Tahmin': predictions
})

predictions_df.set_index('Date', inplace=True)


# In[40]:


plt.figure(figsize=(12, 6))
plt.plot(predictions_df.index, predictions_df['Otomotiv Satis Tahmin'], label='Otomotiv Satis Tahmin', marker='o')
plt.xlabel('Tarih')
plt.ylabel('Otomotiv Satış Tahminleri')
plt.title('Gelecek Otomotiv Satış Tahminleri')
plt.legend()
plt.grid()
plt.show()

