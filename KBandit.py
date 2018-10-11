import numpy as np
import pandas as pd
import matplotlib as mpl
import numpy.random as npr
import matplotlib.pyplot as plt
from dataclasses import dataclass

import seaborn as sns
sns.set()

npr.seed(seed=42)
plt.style.use('ggplot')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

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


	def plot_mean(self, last=False):
		plt.subplot(4, 4, (self.bandit-5)+1)
		plt.ylim(0, 12)
		if 'ucb' in self.algorithm:
			plt.plot(list(range(0, self.timesteps)), self.r_over_time.sum(axis=1).expanding().mean(), label="UCB")
		elif self.optimism is not None:
			plt.plot(list(range(0, self.timesteps)), self.r_over_time.sum(axis=1).expanding().mean(), label=r"Optimistic $\epsilon$-greedy")
		else:
			plt.plot(list(range(0, self.timesteps)), self.r_over_time.sum(axis=1).expanding().mean(), label=r"$\epsilon$-greedy")
		
		plt.title(r'$K$=%s'%self.bandit)
		plt.legend()

	def return_mean(self):
		return self.r_over_time.sum(axis=1).expanding().mean().iloc[-1]

def plot_param_dependence():
	param_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
	ucb_values, egreedy_values, optEGreedy_values  = list(), list(), list()
	timesteps = 1000

	for param in param_values:
		UCB = KBandit(algorithm='ucb', bandit=10, scale=param, timesteps=timesteps)
		UCB.initialize()
		UCB.approximate()
		ucb_values.append(UCB.return_mean())

		EGreedy = KBandit(algorithm='egreedy', bandit=10, epsilon=param, timesteps=timesteps)
		EGreedy.initialize()
		EGreedy.approximate()
		egreedy_values.append(EGreedy.return_mean())

		optEGreedy = KBandit(algorithm='egreedy', bandit=10, epsilon=0.1, optimism=param, timesteps=timesteps)
		optEGreedy.initialize()
		optEGreedy.approximate()
		optEGreedy_values.append(optEGreedy.return_mean())

	plt.clf()

	plt.rcParams["figure.figsize"] = [15, 10]
	plt.plot(np.arange(10), ucb_values, label=r'UCB (c)')
	plt.plot(np.arange(10), optEGreedy_values, label=r'Optimistic $\epsilon$-greedy ($\epsilon$=0.1) ($Q_{0}$)')
	plt.plot(np.arange(10), egreedy_values, label=r'$\epsilon$-greedy ($\epsilon$)')
	plt.legend()

	plt.xticks(np.arange(10), [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$', r'$\frac{1}{16}$', r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', '1', '2', '4'])
	plt.xlabel(r'$\epsilon$, $c$, $Q_{0}$')
	plt.suptitle(r'Average reward over first' + '\n' + '1000 iterations ' + r'($k$=10)' , fontweight='bold')
	plt.ylabel(r'Average reward')
	plt.ylim(0, 12)

	plt.show()

def plot_expanding_mean(param=1/4, timesteps=10000):
	
	plt.clf()

	plt.rcParams['figure.subplot.left'] = 0.09
	plt.rcParams['figure.subplot.bottom'] = 0.08
	plt.rcParams['figure.subplot.right'] = 0.90
	plt.rcParams['figure.subplot.top'] = 0.91
	plt.rcParams['figure.subplot.wspace'] = 0.25
	plt.rcParams['figure.subplot.hspace'] = 0.34
	plt.rcParams["figure.figsize"] = [20, 10]

	for k in range(5, 21):
		
		print(k)

		UCB = KBandit(algorithm='ucb', bandit=k, scale=param, timesteps=timesteps)
		UCB.initialize()
		UCB.approximate()
		UCB.plot_mean()

		optEGreedy = KBandit(algorithm='egreedy', bandit=k, epsilon=param, optimism=param, timesteps=timesteps)
		optEGreedy.initialize()
		optEGreedy.approximate()
		optEGreedy.plot_mean()

		EGreedy = KBandit(algorithm='egreedy', bandit=k, epsilon=param, timesteps=timesteps)
		EGreedy.initialize()
		EGreedy.approximate()
		EGreedy.plot_mean()


	plt.suptitle(r'Average reward for varying $K$'+ '\n' + r'($\epsilon$, $c$, $Q_{0}$)=$\frac{1}{4}$', fontweight='bold')

	plt.text(-timesteps*1.55, -2, 'Iterations',
     rotation=0,
     size=16,
     horizontalalignment='center',
     verticalalignment='top',
     multialignment='center')

	plt.text(-timesteps*4.5, 35, 'Average reward',
     rotation=90,
     size=16,
     horizontalalignment='center',
     verticalalignment='top',
     multialignment='center')
		
	plt.show()

def plot_rewards_distribution():

	plt.clf()

	plt.suptitle(r'Distribution of rewards for varying k' + '\n'+ r'Sample size = $10^{5}$', fontweight='bold')
	plt.rcParams['figure.subplot.left'] = 0.09
	plt.rcParams['figure.subplot.bottom'] = 0.08
	plt.rcParams['figure.subplot.right'] = 0.90
	plt.rcParams['figure.subplot.top'] = 0.91
	plt.rcParams['figure.subplot.wspace'] = 0.25
	plt.rcParams['figure.subplot.hspace'] = 0.34
	plt.rcParams["figure.figsize"] = [35, 10]

	for k in range(5, 21):
		print(k)
		plt.subplot(4, 4, (k-5)+1)
		UCB = KBandit(algorithm='ucb', bandit=k, scale=1/4, timesteps=10000)
		UCB.initialize()

		draws = list()
		for draw in range(100000):
			draw = list()
			for arm in range(0,k):
				draw.append(UCB.rewards(arm))
			draws.append(draw)

		plt.xticks(visible=False)
		sns.violinplot(data=pd.DataFrame(draws), inner="points")
		plt.title(r'$K$=%s'%k)


	plt.text(-27, -2, 'Bandits',
     rotation=0,
     size=16,
     horizontalalignment='center',
     verticalalignment='top',
     multialignment='center',
     fontweight='bold')

	plt.text(-80, 35, 'Reward distribution',
     rotation=90,
     size=16,
     horizontalalignment='center',
     verticalalignment='top',
     multialignment='center',
     fontweight='bold')
	
	plt.show()
	
# plot_expanding_mean()
plot_rewards_distribution()
# plot_param_dependence()




