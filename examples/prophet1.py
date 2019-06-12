import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# df = pd.read_csv('adolescent#001.csv')
df = pd.read_csv('adult#001.csv')
#df.plot(kind='line')
#plt.show()
m = Prophet(changepoint_prior_scale=1).fit(df)
future = m.make_future_dataframe(periods=60, freq='min')
fcst = m.predict(future)
fig = m.plot(fcst)

#ax = plt.gca()

#df.plot(kind='line', x='ds', y='y', ax=ax)
#fcst.plot(kind='line', x='ds', y='yhat', color='red', ax=ax)
plt.show()

