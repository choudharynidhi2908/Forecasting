import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
#from sm.tsa.statespace import sa

airlines =pd.read_csv('C:/Users/nidhchoudhary/Desktop/Assignment/Task/Forecast/Airlines+Data.csv')
 
# Converting the normal index of Airlines to time stamp 
airlines.index = pd.to_datetime(airlines.Month,format="%b-%y")

airlines.Passengers.plot() # time series plot 
# Creating a Date column to store the actual Date format for the given Month column
airlines["Date"] = pd.to_datetime(airlines.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

airlines["month"] = airlines.Date.dt.strftime("%b") # month extraction
#airlines["Day"] = airlines.Date.dt.strftime("%d") # Day extraction
#airlines["wkday"] = airlines.Date.dt.strftime("%A") # weekday extraction
airlines["year"] = airlines.Date.dt.strftime("%Y") # year extraction

# Some EDA on Time series data 
# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=airlines,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Passengers",data=airlines)
sns.boxplot(x="year",y="Passengers",data=airlines)
# sns.factorplot("month","Passengers",data=airlines,kind="box")

Passengerssns.lineplot(x="year",y="Passengers",hue="month",data=airlines)


# moving average for the time series to understand better about the trend character in airlines
airlines.Passengers.plot(label="org")
for i in range(2,24,6):
    airlines["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(airlines.Passengers,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(airlines.Passengers,model="multiplicative")
decompose_ts_mul.plot()

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(airlines.Passengers,lags=10)
tsa_plots.plot_pacf(airlines.Passengers)

# Passengers.index.freq = "MS" 
# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = airlines.head(133)
Test = airlines.tail(12)
# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers) 

# Holt method 
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers)



# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers) # 4.500954



# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers) # 4.109309


# Lets us use auto_arima from p
from pyramid.arima import auto_arima
auto_arima_model = auto_arima(Train["Passengers"],start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=12,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=False)
                
            
auto_arima_model.summary() # SARIMAX(1, 1, 1)x(0, 1, 1, 12)
# AIC ==> 1348.728
# BIC ==> 1362.665

# For getting Fitted values for train data set we use 
# predict_in_sample() function 
auto_arima_model.predict_in_sample( )

# For getting predictions for future we use predict() function 
pred_test = pd.Series(auto_arima_model.predict(n_periods=12))
# Adding the index values of Test Data set to predictions of Auto Arima
pred_test.index = Test.index
MAPE(pred_test,Test.Passengers)  # 8.75353


# Using Sarimax from statsmodels 
# As we do not have automatic function in indetifying the 
# best p,d,q combination 
# iterate over multiple combinations and return the best the combination
# For sarimax we require p,d,q and P,D,Q 
combinations_l = list(product(range(1,7),range(2),range(1,7)))
combinations_u = list(product(range(1,7),range(2),range(1,7)))
m =12 

results_sarima = []
best_aic = float("inf")

for i in combinations_l:
    for j in combinations_u:
        try:
            model_sarima = sm.tsa.statespace.SARIMAX(Train["Passengers"],
                                                     order = i,seasonal_order = j+(m,)).fit(disp=-1)
        except:
            continue
        aic = model_sarima.aic
        if aic < best_aic:
            best_model = model_sarima
            best_aic = aic
            best_l = i
            best_u = j
        results_sarima.append([i,j,model_sarima.aic])


result_sarima_table = pd.DataFrame(results_sarima)
result_sarima_table.columns = ["paramaters_l","parameters_j","aic"]
result_sarima_table = result_sarima_table.sort_values(by="aic",ascending=True).reset_index(drop=True)

best_fit_model = sm.tsa.statespace.SARIMAX(Train["Passengers"],
                                                     order = (1,1,1),seasonal_order = (1,1,1,12)).fit(disp=-1)
best_fit_model.summary()
best_fit_model.aic 
srma_pred = best_fit_model.predict(start = Test.index[0],end = Test.index[-1])
airlines["srma_pred"] = srma_pred

# Visualization of Forecasted values for Test data set using different methods 
plt.plot(Train.index, Train["Passengers"], label='Train',color="black")
plt.plot(Test.index, Test["Passengers"], label='Test',color="blue")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="red")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponential_1",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponential_2",color="yellow")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="Auto_Arima",color="grey")
plt.plot(pred_hwe_mul_add.index,srma_pred,label="Auto_Sarima",color="purple")
plt.legend(loc='best')

# Models and their MAPE values
model_mapes = pd.DataFrame(columns=["model_name","mape"])
model_mapes["model_name"] = ["]
# Visualizing the ACF and PACF plots for errors 