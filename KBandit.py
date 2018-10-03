import numpy as np
import pandas as pd
import matplotlib as mpl
import numpy.random as npr
import matplotlib.pyplot as plt
from dataclasses import dataclass

npr.seed(seed=42)
plt.style.use('ggplot')

@dataclass
class KBandit:

	Qmu: float = 5
	algorithm: str = None
	bandit: int = 10
	Qsigma: float = 2
	Rsigma: float = 10
	exp: int = 4
	stationary: bool = False
	epsilon: float = None
	approximation: pd.DataFrame() = None
	expanded_mean: pd.DataFrame() = None

	def initialize(self):
		self.timesteps = (10 ** self.exp)
		self.Q = np.zeros((self.timesteps, self.bandit))
		self.N = np.zeros((self.timesteps, self.bandit))
		
		self.Qtrue = npr.normal(self.Qmu, self.Qsigma, self.bandit)
		self.rewards = lambda x: np.array([npr.normal(mu, self.Rsigma) for mu in self.Qtrue])[x]
		
		# random tie breaking when multiple equal max
		self.egreedy = lambda q, *args: npr.randint(0, self.bandit) if npr.rand() < self.epsilon else npr.choice(np.flatnonzero(q == q.max()))
		self.ucb = lambda Q, c, t, n: np.argmax([q + (c *(np.sqrt(np.log(t)/np.sum(n[0:t,i], axis=0)))) for i, q in enumerate(Q)])

		self.alg_map = {'egreedy': self.egreedy, 'ucb': self.ucb}
		self.func = lambda *args: self.alg_map.get(self.algorithm)(*args)

	def approximate(self):
		for t in range(1, self.timesteps):
			At = self.func(self.Q[t-1,], 2, t, self.N)
			
			R = self.rewards(At)

			self.N[t-1, At] += 1
			self.Q[t,] = self.Q[t-1, ]
			self.Q[t, At] = self.Q[t-1, At] + ((R - self.Q[t-1, At])/ np.sum(self.N[0:t, At], axis=0))

		self.approximation = pd.DataFrame(self.Q)
		self.expanded_mean = self.approximation.expanding().mean()

	def plot(self):
		plt.clf()
		for i in range(self.bandit):
			plt.subplot(5, 2, i+1)
			plt.plot(list(range(0, self.timesteps)), self.expanded_mean [i])

			plt.ylim(0, self.Qmu*1.8)
			plt.xlabel('Iteration')
			if i+1 == 5:
				plt.ylabel('Average reward')
			plt.title('K=%s'%(i+1))

		plt.title('%s-Bandit Algorithm' % self.bandit)
		plt.show()


UCB = KBandit(algorithm='ucb', bandit=10, exp=3)
UCB.initialize()
UCB.approximate()
UCB.plot()



