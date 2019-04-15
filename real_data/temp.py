import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

data = pd.read_csv('ANMO_Zchannel.csv')
data = data.values
data = data[:,0]

plt.plot(np.diff(np.diff(data)))
plt.show()