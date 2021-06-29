
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima_model import ARIMA

airlines =pd.read_csv('C:/Users/nidhchoudhary/Desktop/Assignment/Task/Forecast/Airlines+Data.csv')


tsa_plots.plot_acf(airlines.passengers,lags=12)
tsa_plots.plot_pacf(airlines.passengers,lags=12)


model1=ARIMA(airlines.passengers,order=(1,1,6)).fit(disp=0)
model2=ARIMA(airlines.passengers,order=(1,1,5)).fit(disp=0)
model1.aic
model2.aic

p=1
q=0
d=1
pdq=[]
aic=[]
for q in range(7):
    try:
        model=ARIMA(airlines.passengers,order=(p,d,q)).fit(disp=0)

        x=model.aic

        x1= p,d,q
               
        aic.append(x)
        pdq.append(x1)
    except:
        pass
            
keys = pdq
values = aic
d = dict(zip(keys, values))
print (d)

