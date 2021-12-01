import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 2 * np.pi, 100)
for i in range(5):
	Ya = np.sin((i+1)/5*X)
	plt.plot(X, Ya, label='angular frec:'+str((i+1)/5))
plt.xlabel('State')
plt.title('Binomial probabilities learned from V1 DS.')
plt.ylim([0.0,1.0])
plt.xlim([0.0,6.0])
plt.legend()
plt.show()