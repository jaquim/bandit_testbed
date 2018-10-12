'''
    File name: KBandit.py
    Author: Jaquim Cadogan
    Date created: 5-8-18
    Python Version: 3.7 (!)
'''

import numpy as np
import pandas as pd
import seaborn as sns
import numpy.random as npr
import matplotlib.pyplot as plt
from dataclasses import dataclass

sns.set()
npr.seed(seed=42)
plt.style.use('ggplot')

# painstakinly filling figure parameters (obtained by playing with sliders)
# for matplotlib in by hand (no excuse not to look fly)
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
	# in Python 3.7 we can type annotate variable for increased readility and get hints 
	# on what you are dealing with, note that nothing is enforced on byte level however
	Qmu: float = 5
	scale: float = 2
	bandit: int = 10
	Qsigma: float = 2
	Rsigma: float = 0.5
	algorithm: str = None
	epsilon: float = None
	optimism: float = None
	timesteps: int = 20000
	stationary: bool = False
	approximation: pd.DataFrame() = None
	expanded_mean: pd.DataFrame() = None

	def initialize(self):
		# if we are dealing with the optimitic egreedy algorithm we initialize the Q-values with some 
		# custom to be passed value, otherwise init with zero
		self.Q = np.ones((self.timesteps, self.bandit)) * self.optimism if bool(self.optimism) else np.zeros((self.timesteps, self.bandit))
		self.N = np.zeros((self.timesteps, self.bandit))
		self.r_over_time = np.zeros((self.timesteps, self.bandit))
		# let's define arms with true means that are also normally distributed
		self.Qtrue = npr.normal(self.Qmu, self.Qsigma, self.bandit)
		# retrieve the reward for a given arm, with fixed true mean for the respective arm, fixed variance for all arms
		self.rewards = lambda x: np.array([npr.normal(mu, self.Rsigma) for mu in self.Qtrue])[x]
		# egreedy random tie breaking when multiple equal max (disregard the *args for egreedy)
		self.egreedy = lambda q, *args: npr.randint(0, self.bandit) if npr.rand() < self.epsilon else npr.choice(np.flatnonzero(q == q.max()))
		# compute and argmax among the upper confidence bound for every of the bandit
		self.ucb = lambda Q, c, t, n: np.argmax([q + (self.scale *(np.sqrt(np.log(t)/np.sum(n[0:t,i], axis=0)))) for i, q in enumerate(Q)])
		# neat trick to avoid case statement and easily generalizible, as this dict is easily extendable and modular
		self.alg_map = {'egreedy': self.egreedy, 'ucb': self.ucb}
		# we can pass functions as variables, and we can even pass
		# variables to these functions as variables (very cool)
		self.func = lambda *args: self.alg_map.get(self.algorithm)(*args)

	def approximate(self):
		"""
		The implementation of the actual algorithm and the updates of the Q-values over time.
		I tried to stay as close as possible to the pseudo-algorithms as described in Sutton and Barto, Ch. 2
		Ultimately we are interested in how the average reward grows for the respective algorithms. 
		I use a neat trick from pandas to do this.
		"""
		for t in range(1, self.timesteps):
			At = self.func(self.Q[t-1,], 2, t, self.N) # according to specified algorithm (self.algorithm): choose arm to lever
	
			R = self.rewards(At) # retrieve normally distributed reward for chosen action

			self.N[t, ] = self.N[t-1] # get action counts up until time t
			self.N[t, At] += 1 # update action count for action played

			self.Q[t,] = self.Q[t-1, ] # get q values up until time t
			self.Q[t, At] = self.Q[t-1, At] + ((R - self.Q[t-1, At])/self.N[t, At]) # update Q for actiom played

			self.r_over_time[t, At] = R # this is what we are ultimately interested in

		# note: sum first to warp all rewards into 1 column, then compute its expanded mean
		self.r_over_time = pd.DataFrame(self.r_over_time).sum(axis=1).expanding().mean() 
		self.approximation = pd.DataFrame(self.Q)
		self.expanded_mean = self.approximation.expanding().mean()


	def plot_mean(self, last=False):
		"""Fill in a 4x4 subplot figure, 1 for every k (4<k<21) """
		plt.subplot(4, 4, (self.bandit-5)+1)
		plt.ylim(0, 12)
		if 'ucb' in self.algorithm:
			plt.plot(list(range(0, self.timesteps)), self.r_over_time, label="UCB")
		elif self.optimism is not None:
			plt.plot(list(range(0, self.timesteps)), self.r_over_time, label=r"Optimistic $\epsilon$-greedy")
		else:
			plt.plot(list(range(0, self.timesteps)), self.r_over_time, label=r"$\epsilon$-greedy")
		
		plt.title(r'$K$=%s'%self.bandit)
		plt.legend()

	def return_mean(self) -> float:
		"""Retrieve the expanded average mean value at termination of the iteration cycle"""
		return self.r_over_time.sum(axis=1).expanding().mean().iloc[-1]

def plot_param_dependence():
	"""
	Recreate the plot for the three algorithms and their dependence on
	the value for their controlling parameter as seen in Sutton and Barto, Ch.2

	"""

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

	# formatting stuff
	plt.rcParams["figure.figsize"] = [15, 10]
	plt.plot(np.arange(len(param_values)), ucb_values, label=r'UCB (c)')
	plt.plot(np.arange(len(param_values)), optEGreedy_values, label=r'Optimistic $\epsilon$-greedy ($\epsilon$=0.1) ($Q_{0}$)')
	plt.plot(np.arange(len(param_values)), egreedy_values, label=r'$\epsilon$-greedy ($\epsilon$)')
	plt.xticks(np.arange(len(param_values)), [r'$\frac{1}{128}$', r'$\frac{1}{64}$', r'$\frac{1}{32}$', r'$\frac{1}{16}$',\
			r'$\frac{1}{8}$', r'$\frac{1}{4}$', r'$\frac{1}{2}$', '1', '2', '4'])
	plt.xlabel(r'$\epsilon$, $c$, $Q_{0}$')
	plt.suptitle(r'Average reward over first' + '\n' + '1000 iterations ' + r'($k$=10)' , fontweight='bold')
	plt.ylabel(r'Average reward')
	plt.legend()
	plt.ylim(0, 12)

	plt.show()

def plot_expanding_mean(param=1/4, timesteps=10000):
	"""
		Plot the growth of the average reward for a specified parameter and timesteps
	"""
	
	plt.clf()
	# formatting stuff
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

	# formatting stuff
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
	"""
		In order to visualize what the distributions of the arms look like for varying K.
		Note that the closer the means for certain arms are, the lower the KL-divergence is.
		A nice way to explain for example an equal performance of UCB and optimistic egreedy,
		or even an outperformance of optimistic egreedy.
	
	"""
	plt.clf()

	# format stuff
	plt.suptitle(r'Distribution of rewards for varying k' + '\n'+ r'Sample size = $10^{5}$', fontweight='bold')
	plt.rcParams['figure.subplot.left'] = 0.09
	plt.rcParams['figure.subplot.bottom'] = 0.08
	plt.rcParams['figure.subplot.right'] = 0.90
	plt.rcParams['figure.subplot.top'] = 0.91
	plt.rcParams['figure.subplot.wspace'] = 0.25
	plt.rcParams['figure.subplot.hspace'] = 0.34
	plt.rcParams["figure.figsize"] = [35, 10]

	# sample for every arm, for every k 
	# to estimate its distribution
	for k in range(5, 21):
		print(k) # to indicate how far we are (takes a while, leaving it here delibaretly)
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

	# format stuff
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




