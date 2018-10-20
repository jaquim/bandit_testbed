import numpy as np
import pandas as pd
import matplotlib as mpl
import numpy.random as npr
import matplotlib.pyplot as plt
from dataclasses import dataclass

mpl.use('TkAgg')
plt.style.use('ggplot')

npr.seed(seed=42)

bandit = 10
epsilon = 0.2
timesteps = (10 ** 4)

Qmu, Qsigma = 5, 2
Qtrue = npr.normal(Qmu, Qsigma, 10)


Qtrue = [1, 2, 3, 3, 2, 2, 2, 1]
rewards = lambda x: np.array([npr.normal(mu, 10) for mu in Qtrue])[x]

Q = np.zeros((timesteps, bandit))
N = np.zeros((timesteps, bandit))

ucb = lambda Q, c, t, n: np.argmax([q + (c *(np.sqrt(np.log(t)/np.sum(n[0:t,i], axis=0)))) for i, q in enumerate(Q)])
egreedy = lambda q, *args: npr.randint(0, bandit) if npr.rand() < epsilon else npr.choice(np.flatnonzero(q == q.max())) # random tie breaking when multiple equal max

algorithm: str = 'ucb'
alg_map = {'egreedy': egreedy, 'ucb': ucb}
func = lambda *args: alg_map.get(algorithm)(*args)

for t in range(1, timesteps):
	
	At = func(Q[t-1,], 2, t, N)
	
	R = rewards(At)

	N[t-1, At] += 1
	Q[t,] = Q[t-1, ]
	Q[t, At] = Q[t-1, At] + ((R - Q[t-1, At])/ np.sum(N[0:t, At], axis=0))

df = pd.DataFrame(Q)
df_rolling_mean = df.expanding().mean()


for i in range(bandit):
	plt.subplot(5, 2, i+1)
	plt.plot(list(range(0, timesteps)), df_rolling_mean[i])

	plt.ylim(0, Qmu*1.8)
	plt.xlabel('Iteration')
	if i+1 == 5:
		plt.ylabel('Expanding average reward')
	plt.title('K=%s'%(i+1))


plt.title('%s-Bandit Algorithm' % bandit)
plt.show()





