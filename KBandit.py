import numpy as np
import pandas as pd
import matplotlib as mpl
import numpy.random as npr
import matplotlib.pyplot as plt
from dataclasses import dataclass

npr.seed(seed=42)
plt.style.use('ggplot')
plt.rcParams['figure.subplot.left'] = 0.09
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.90
plt.rcParams['figure.subplot.top'] = 0.91
plt.rcParams['figure.subplot.wspace'] = 0.25
plt.rcParams['figure.subplot.hspace'] = 0.34

# plt.suptitle(r'$\epsilon$-greedy: for $\epsilon \in \{0.1,0.2,0.4,0.8\}$ and $5 \leq k \leq 20$')
plt.suptitle(r'UCB: for $c \in \{2,4,8,16\}$ and $5 \leq k \leq 20$')


@dataclass
class KBandit:

	Qmu: float = 5
	algorithm: str = None
	bandit: int = 10
	Qsigma: float = 2
	Rsigma: float = 0.5
	timesteps: int = 20000
	scale: float = 2
	optimism: float = None
	stationary: bool = False
	epsilon: float = None
	approximation: pd.DataFrame() = None
	expanded_mean: pd.DataFrame() = None

	def initialize(self):
		self.Q = np.ones((self.timesteps, self.bandit)) * self.optimism if bool(self.optimism) else np.zeros((self.timesteps, self.bandit))
		self.N = np.zeros((self.timesteps, self.bandit))
		self.r_over_time = np.zeros((self.timesteps, self.bandit))
		
		self.Qtrue = npr.normal(self.Qmu, self.Qsigma, self.bandit)
		self.rewards = lambda x: np.array([npr.normal(mu, self.Rsigma) for mu in self.Qtrue])[x]
		
		# random tie breaking when multiple equal max
		self.egreedy = lambda q, *args: npr.randint(0, self.bandit) if npr.rand() < self.epsilon else npr.choice(np.flatnonzero(q == q.max()))
		self.ucb = lambda Q, c, t, n: np.argmax([q + (self.scale *(np.sqrt(np.log(t)/np.sum(n[0:t,i], axis=0)))) for i, q in enumerate(Q)])

		self.alg_map = {'egreedy': self.egreedy, 'ucb': self.ucb}
		self.func = lambda *args: self.alg_map.get(self.algorithm)(*args)

	def approximate(self):
		for t in range(1, self.timesteps):
			At = self.func(self.Q[t-1,], 2, t, self.N)
			
			R = self.rewards(At)

			self.N[t, ] = self.N[t-1]
			self.N[t, At] += 1

			self.Q[t,] = self.Q[t-1, ]
			self.Q[t, At] = self.Q[t-1, At] + ((R - self.Q[t-1, At])/self.N[t, At])

			self.r_over_time[t, At] = R

		self.r_over_time = pd.DataFrame(self.r_over_time)
		self.approximation = pd.DataFrame(self.Q)
		self.expanded_mean = self.approximation.expanding().mean()


	def plot_mean(self, k):
		plt.subplot(4, 4, (self.bandit-5)+1)

		if 'ucb' in self.algorithm:
			plt.plot(list(range(0, self.timesteps)), self.r_over_time.sum(axis=1).expanding().mean(), label="scale=%s"%(self.scale))
		elif self.optimism is not None:
			plt.plot(list(range(0, self.timesteps)), self.r_over_time.sum(axis=1).expanding().mean(), label=r"$\epsilon=%s, optimism=%s$"%(self.epsilon, self.optimism))
		else:
			plt.plot(list(range(0, self.timesteps)), self.r_over_time.sum(axis=1).expanding().mean(), label=r"$\epsilon=%s$"%(self.epsilon))
		
		plt.title('K=%s'%self.bandit)
		plt.legend()

	def violin(self):
		for i in range(self.bandit):
			plt.violinplot(sorted(self.approximation[i]))
		plt.show()



for k in range(5, 21):
	UCB = KBandit(algorithm='ucb', bandit=k, scale=2)
	UCB.initialize()
	UCB.approximate()
	UCB.plot_mean()

	UCB = KBandit(algorithm='ucb', bandit=k, scale=4)
	UCB.initialize()
	UCB.approximate()
	UCB.plot_mean()

	UCB = KBandit(algorithm='ucb', bandit=k, scale=8)
	UCB.initialize()
	UCB.approximate()
	UCB.plot_mean()

	UCB = KBandit(algorithm='ucb', bandit=k, scale=16)
	UCB.initialize()
	UCB.approximate()
	UCB.plot_mean()


plt.show()

