---
layout: page
title: "PAI 3: Bayesian Optimization(Bandits, etc.), Markov Decision Process(Policy Gradient, etc.)"
permalink: /pai/bayes-opt
group: pai
---

# Bayesian Optimization

## Exploration-Exploitation Dilemma

- Exploration: Exploring the input space while learning the function. Choosing data points that are most "informative". 

- Exploitation: Exploiting the learned functions by focusing on the most promising among explored areas. Choosing data points with better expected evaluations. 

## Bandits

**Online(Sequential) Learning/Multi-Armed Bandits/Bayesian Optimization**

We have $k$ actions(arms) at each time step, and wish to maximize a reward functiona at Horizon $T$. The reward function is $\Sigma_t f^*(x_t)$. $f^*$ is unknown. 

Since we do not know the reward distribution(and this a one-time game), we face the exploration-exploitation dilemma at each action. 

Bayesian Optimization is a variant to the this MAB problem, where we have *infinite arms* and *correlated rewards*.  

- Infinite Arms: $x$ is in a continuous space

- Correlated Rewards: Instead of observing the exact function value at each time step, we obtain a noisy observation: $y_t = f^*(x_t)+ \epsilon_t$. With a common assumption of similiar points yielding similiar outcomes, we fit a GP to the black-box reward function(GP prior). Recall GP are infitely dimensional(in this case, we have infinitely possible rewards), and the (kernelized) covariance matrix will describe the relationship between rewards. 

*"Optimism in the face of uncertainty" Principle*: Explore where we can hope for the best outcomes. 

For online learning, our previous inputs has no impact on the reward function for future timesteps(stationary environment). We'll start here and go to dynamic environment(RL). 

**Regret**: is the additive loss with respect to the static optimum. 

$$R_T = \Sigma_t (max(f^*(x))-f^*(x_t)) = Tmax(f^*(x)) -\Sigma_t f^*(x_t)$$

*Note that at each time step, the optimal action is the same. We simply don't know the  optimum during the optimization process.*

**Sublinear Regret** is the metric for a good solution

We wish to achieve sublinear regret, this means that as the horizon goes to infinite, our regret at each time step goes to zero: $lim_{T\to \infty} \frac{R_T}{T} = 0$ 

Sublinear regret means that we will eventually find the optimum(exploration) and we are able to exploit this optimum(exploitation). Therefore, we can say that we achieve a balance in the exploration-exploitation dilemma. 

Note that this is only hard because the reward function is a black box, and we never know when we'll or have found the optimum(or what the optimum would be).  

## Acquisition Functions

Given the problem, we wish to pick the next point greedily. Before observing and updating GP at each time step, we pick the data point with **acquisition function** $F$: $argmax_x F(x;\mu_{t-1}, k{t-1})$. 

Recall the active learning problem, uncertainty sampling(picking points with largest uncertainty/variance) can be seen as such an acquisition function with mere exploration. We probably don't want that. 

On the contrary, we probably do not want an acquisition function with mere exploitation(simply argmax the functionw e know). 

Also, note that the GP posterior is probabilistic, and we *hope* that the ground truth is bounded by some confidence interval. There is, also, an action value that produces a reward distribution with the best lower bound(**maximum lower bound**). (consult Figure 9.3)

There are, therefore, regions of actions of interest that can possibly yield rewards above that maximum lower bound (consult yellow regions in Figure 9.3).

Among the actions inside the region of interest, we can eventually choose an action with severl acquisition functions (with different interpretation of *"Optimism in the face of uncertainty"*). 

*Also, we may also notice that at the early gameplay. The GP is essentially dominated by prior and thus not bound the actual function quite well.*

*We may notice while introducing acquisition functions that some are more theoretical than others and we can derive explicit regret bounds.*

## Upper Confidence Bounds(UCB)

UCB chooses the action with the largest upper bound for the reward. 

$$x_{t+1} = argmax_x \mu_t(x) + \beta_{t+1}\sigma_t(x)$$

As the "z-score" $\beta \to \infty$, you can see that the standard deviation term dominate the acquisition function(uncertainty sampling). We also see that we can tune this z-score term over time to explore more in the early gameplay period and exploit in the late gameplay period.

**Well-calibrated confidence intervals**

We choose the $\beta$ sequence so that it leads to *well-calibrated confidence intervals*. 

(eq 9.7) With $P>1-\delta$, the ground truth function are bounded on all time step: 

$$\forall t \geq 1, \forall x \in X: f^*(x) \in C_t(x) = \mu_{t-1}(x) \pm \beta_t(\delta) \sigma_{t-1}(x)$$

With this criteria, we can derive a beta sequence under two different settings. 

**Bayesian Setting** means that we have a prior $f^* \sim GP(\mu_0, k_0)$ (*Theorem 9.4, Problem 9.2, Srinivas et al. 2010 A.2, Lemma 5.1, 5.5*)

The bound is derived from the tail probability of $N(0,1)$: 

$$Pr(r>c) = \frac{1}{2\pi}\int^\infty_c e^{-\frac{u^2}{2}}du$$

Note $r = \frac{f^*(x)-\mu_{t-1}(x)}{\sigma_{t-1}}>c$ is an event about $f^*$ interval's *upper bound* and thus $c > 0$(positive z-score for upper bound of GP). We define $z = u - c \to u = z+c$ to map the integral range to $(0, \infty)$. 

$$\begin{aligned}
& Pr(r>c) = \frac{1}{2\pi}\int^\infty_0 e^{-\frac{(z+c)^2}{2}} dz\\
& = \frac{1}{2\pi} \int^\infty_0 e^{-\frac{z^2}{2}}e^{-zc}e^{-\frac{c^2}{2}} dz \\
& = \frac{1}{2\pi} e^{-\frac{c^2}{2}} \int^\infty_0 e^{-\frac{z^2}{2}}e^{-zc} dz \\
& \leq \frac{1}{2\pi} e^{-\frac{c^2}{2}} \int^\infty_0 e^{-\frac{z^2}{2}} (1) dz\\
& = e^{-\frac{c^2}{2}}Pr(r>0) \leq e^{-\frac{c^2}{2}}
\end{aligned}$$

$$z \geq 0; c > 0 \to e^{-zc} < 1$$

For convenience(in math), we are interested in the event $r > \sqrt{\beta_t}$. 

$$\begin{aligned}
& Pr(\vert r \vert > \sqrt{\beta_t}) \leq e^{-\frac{\beta_t}{2}}\\
& \to Pr(\vert r \vert < \sqrt{\beta_t}) \geq 1-  e^{-\frac{\beta_t}{2}}
\end{aligned}$$

We define $\beta_t = - 2 \log(\delta/ \pi_t)$, and acquire $\Pi Pr(\vert \frac{f^*(x)-\mu_{t-1}(x)}{\sigma_{t-1}} \vert < \beta_t) \geq 1 - \frac{\delta}{\pi_t}$

$\pi_t$ is defined such that its inverse sums to one across all steps, and we use union bound to assume a lower bound for success: 

$Pr(\forall t, \vert \frac{f^*(x)-\mu_{t-1}(x)}{\sigma_{t-1}} \vert < \beta_t) \geq 1-\Sigma_t \frac{\delta}{\pi_t} = 1 - \delta$

The bound is not $\Pi_t (1-\frac{\delta}{\pi_t})$ because we *do not* assume independence. Union bounds means that we assume disjoint event and always overestimates the probability for an event(in our case, failure to bound). 

*This would, with further proofs, give us sublinear regret in theorem 9.3.*

**Freqeuntist Setting** means that we assume no prior about $f^*$. Rahter, we assume $f^* \in H_k(X)$ or $\vert \vert f^*\vert \vert_k < \infty$, recall the RKHS and function's norm in RKHS. (This can be shown to be contradictory to the bayesian assumption, according to the book. )

*I would not pursue proofs for the frequentist setting unless I have more time, but a rough skim through Chowdhury and Gopalan 2017 shows that the norm of the error sequence $\epsilon_{1:t}$ is bounded.*

## Improvement: PI, EI

The improvement is defined as($\hat f_t$ is the current optimum): 

$$I_t(x) = max\{0,(f(x) -\hat f_t)\}$$

**Probability of Improvement(PI)**

PI picks the points that maximize the probability to improve upon current optimum. 

$$\begin{aligned}
&x_{t+1} = argmax_x P(I_t(x)>0\vert x_{1:t},y_{1:t}) \\
& = argmax_x P(f(x)>\hat f_t\vert x_{1:t},y_{1:t}) \\
& = argmax_x \Phi(\frac{\mu(x) - \hat f_t}{\sigma_t(x)})
\end{aligned}$$

**Expected Improvement(EI)**

An alternative route is to maximize EI, so we'll have chances to explore regions with less PI however promising and not fully explored(lower mean, higher variance): 

$$x_{t+1} = argmax_x E[I_t(x) \vert x_{1:t},y_{1:t}]$$

The book mentioned that the EI acquisition function is flat thus suffer from vanishing gradient. It is equivalent to maximize the logarithm. 

$$x_{t+1} = argmax_x \log E[I_t(x) \vert x_{1:t},y_{1:t}]$$

## Thompson Sampling

**Probability Matching** *samples* points according to the probability that it is optimal: 

$$\pi(x\vert x_{1:t}, y_{1:t}) = P_{f\vert x_{1:t}, y_{1:t}}(f(x) = max_{x'} f(x'))$$
$$x_{t+1} \sim \pi(.\vert x_{1:t}, y_{1:t})$$

It is hard to compute $\pi$ in analytical form.

**Thompson Sampling** uses MC sampling

We can use Monte Carlo sampling(the function posterior is GP, thus an "easier" distribution): 

1. We sample a $\tilde{f}_{t+1}$ from function posterior. 

2. We find the optimum for that sampled function $x_{t+1}$. 

$$x_{t+1} = argmax_x \tilde{f}_{t+1}(x)$$

*By sampling from a GP, we essentially are sampling a vector of discrete function values. With these sampled means and the kernel function, (as you may recall from 4.1) means and kernel matrix is defined for new points.*

*By MC sampling, we assumes that the single best guess from one random sample is the only point of interest. However, MC are garuanteed to converge for infinite steps.*

## Information-Directed Sampling(IDS)

**Information Ratio** is ratio of instantenous regret(penalty) and functions about information gain(benefit). 

$$\Psi_t(x) = \frac{(\Delta x)^2}{I_t(x)}; $$
$$\Delta x = max_{x'} f^*(x') - f^*(x)$$

Here $I_t(x)$ is not the previously introduced improvement and is left to be defined. Here is a definition of marginal gain in example 9.10: 

$$I_t(x) = I(f_x; y_x \vert x_{1:t}, y_{1:t})$$

And we wish to find points to minimize this information ratio. 

$$x_{t+1} = argmin_x(\Psi_t(x))$$

The instantaneous regret is on the ground truth function(not observable). 

**Kirschner and Krause(2018)** proposes using current model to surrogate. 

*The book went through derivation of regret bounds for IDS and the K&K variant, which we may skip for now.*

## LITE, Menet and Hübotter(2025) (*9.3.5*)

LITE is a method developed by one of the arthor. Or you can see it as another acquisition function. 

This section focus on the problem: *How can we estimate the probability of maximality?* In other words, the previous introduced hard-to-compute $\pi$:

$$\pi(x\vert x_{1:t}, y_{1:t}) = P_{f\vert x_{1:t}, y_{1:t}}(f(x) = max_{x'} f(x'))$$

**LITE**

Introducing a threshold(higher than the current optimum) $\kappa^*$, we can approximate $\pi$:

$$\begin{aligned}
& \pi(x\vert x_{1:t}, y_{1:t}) = P_{f\vert x_{1:t}, y_{1:t}}(f(x) \geq max_{x'} f(x')) \\
& \approx P_{f\vert x_{1:t}, y_{1:t}}(f(x) \geq \kappa^*)\\
& = \Phi(\frac{\mu_t(x) -  \kappa^*}{\sigma_t(x)})
\end{aligned}$$

The threshold is chosen such that approximation of $\pi$ integrates to 1. 

Note, $\Phi$ is a CDF instead of a PDF. Also, we are integrating over the search space of $x$, and we have different $\mu_t(x)$. 

*I'm not very concerned about how to compute x in practice.*

LITE shows that this method maximize the **Variational Objective** with exploitation and exploration terms:

$$W(\pi) = \Sigma_x \pi(x)[\textcolor{red}{\mu_t(x)}+ \textcolor{blue}{\sqrt{2S'(\pi(x))}} \sigma_t(x)]$$

Quasi-surprise is an estimate of surprise:$S'(u) = \frac{1}{2} [\frac{\phi(\Phi^{-1}(u))}{u}]^2$

A new concept(Entropy Regularization, the weighted standard deviation) is interoduced so that the method is incentivized to choose a set of points that are more spread out and diverse, maximizing the chance of finding the true optimum regardless of its exact location.

- The exploration in "optimism in the face of uncertainty"(previously seen methods) encourages finding points with high probability(variance) above the current optimum mean. 

- The exploration in Entropy Regularization is about increasing the probability of finding optimum $\pi(x)$. (less about current optimum mean)

# Markov Decision Process

