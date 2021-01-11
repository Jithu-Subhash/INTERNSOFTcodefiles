import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.read_csv('AAPL.csv' , usecols=[0,1,2,3,4])

POHL_avg = s[['Price','Open','High','Low']].mean(axis=1)

obs = np.arange(1,len(s)+1,1)
 
plt.plot(obs,POHL_avg,'r',label='my first plot')
plt.show()
