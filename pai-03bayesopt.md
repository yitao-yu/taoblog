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

**Freqeuntist Setting** means that we assume no prior about $f^*$. Rahter, we assume $f^* \in H_k(X) \leftrightarrow \vert \vert f^*\vert \vert_k < \infty$, recall the RKHS and function's norm in RKHS. (This can be shown to be contradictory to the bayesian assumption, according to the book. )

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

Markov Decision Process is about probabislitic planning, featuring agents playing (making sequential actions) in a *known* stochastic environment. 

- states: $X = \{x_1 ... x_n\}$

- actions: $A = \{a_1 ... a_m\}$

- transition probabiltiy: $p(x'\vert x, a)$

- reward: $r: X\times A \to R$

$$R_t = r(X_t, A_t)$$

**Policy** assigns a probability to each action under a specific state. $\pi(a \vert x) = P(A_t = a \vert X_t = x)$

*I just wish to note here that this abstraction also extends to multi-player games. The agent would see the the actions of other players changing the state/environment as a response to their action. Be the policy of other players deterministic or probabilistic.*

*Also, to avoid confusion: in game theory, there is this notion of mixed-strategy Nash Equilibrium(basically probabilistic policy for some players). However, our book states that for fully observable environments, the agent always has an optimal deterministic policy. This is, however, not contradictory. In game theory, games are played for infinite rounds and every players are allowed to change their policy between rounds. However, here in MDP, (1) a game is played one time (2) a "fully observable environment" means that before each game, the agent is aware of the transition probability and the starting state. An expected reward can be calculated for every actions and thus some option must be optimal.*

```
Consider this enforcer-offender game: offender can break the rule(+10 reward, -100 if caught) or obey(-5 reward), enforcer can patrol(-50 reward, +100 if catching anyone) or stay in office(0 reward).

A game theory view is interested in finding the mixed-startegy equilibrium and allows the enforcer and offender to change there strategy before each iterations until no more increase in utility. As you can see, there is no pure/detereministic strategy equilibrium in this game, because if one side deploy a pure strategy the other side can always alter their pure strategy in a profitable way. For example, it's always profitable for enforcer to switch to "patrol" from "rest" when enforcer chooses "offend" consistently, and vice versa. 

A MDP agent playing the offender would know the strategy of the officer(say 10% chance of patroling) and calculate rewards, determining that "breaking the rule" would be a fixed optimal policy.
```

$E r_b = -100*0.1+0.9*10=-1 > E r_f = -5$

A fixed policy implies that the process can be described by a Markov chain(no tree-like structure). 

**Discounted Playoff** is a score written from infinite reward. 

$$G_t = \Sigma^{\infty}_{m=0} \gamma^m R_{t+m} ;  0 \leq \gamma < 1$$

**State-Action Value Function(Q-function)** measures the average discounted playoff given $t, a, x$. 

$$q^\pi_t(x,a) = E_\pi [G_t\vert x, a] = r(x,a) + \gamma \Sigma_{x'} p(x' \vert x, a) v^{\pi}_{t+1}(x')$$

$v^{\pi}_t(x) = E_{\pi} [G_t \vert x]$ is **State Value Function**. Under our current assumption(stationary dynamics, rewards and policy), both are independent from $t$. 

## Bellman Expectation Equation

Bellman Expectation Equation reveals the recursive nature of state-action value functions (q functions) and state value functions(v functions). 

$$\begin{aligned}
& v^\pi(x) = E_\pi [G_0 \vert X_0 = x] \\
& = E_\pi [\Sigma_m \gamma^m R_m \vert X_0 = x] \\
& = E_\pi[R_0 \vert X_0 = x] + \gamma E_\pi [\Sigma_m \gamma^m R_{m+1} \vert X_0 = x] \\
& = r(x, \pi(x)) + \gamma p(x'\vert x , \pi(x)) E_\pi [\Sigma_m \gamma^m R_m \vert X_0 = x'] \\
& = r(x, \pi(x)) + \gamma \Sigma_{x'} p(x' \vert x, \pi(x)) v^\pi(x') \\
& = r(x, \pi(x)) + \gamma E_{x\vert x', \pi(x)} [v^\pi(x')]

\end{aligned}$$

*Think of evaluating the stochastic policies as traversing trees and evaluating the deterministic policies as walking down one branch of the tree*

For stochastic policies: 

1. The value function can be shown to be the expectation of the q function.

$$\begin{aligned}
& v^{\pi}(x) = \Sigma_a \pi(a\vert x) (r(x'\vert x, a)v^\pi(x')) \\
& = E_{a\sim \pi(x)}[q^\pi(x,a)]
\end{aligned}$$

2. Thus, the q function is recursive(since it's defined by a state-value function). 

$$\begin{aligned}
& q^{\pi}(x, a) = r(x,a) + \gamma \Sigma_{x'} p(x'\vert x, a) \Sigma_{a'} \pi(a' \vert x') q^\pi(x', a') \\
& = r(x, a) + \gamma E_{x'\vert x,a} E_{a' \sim \pi(x)}[q^\pi(x', a')]
\end{aligned}$$

At this point I would argue that deterministic policies simply are stochastic policies with a 100% possibility assigned to the optimal action(under a given state). However, we can also appreciate a more simple notation as the book listed: 

$$v^{\pi}(x) = q^{\pi}(x,\pi(x))$$

## Policy Evaluation

This section focus on computing the $v^\pi$ for a given policy. 

The Bellman Equation in vector/matrix form(for n possible states):

$$\boldsymbol{v^\pi} = \boldsymbol{r^\pi} + \gamma \boldsymbol{P^\pi} \boldsymbol{v^\pi}$$
$$\boldsymbol{v^\pi} = (I - \gamma \boldsymbol{P^\pi})^{-1}\boldsymbol{r^\pi}$$

However, this involves matrix inverse $(O(N^3))$ and thus costly. 

We can approximate by **Fixed Point Iteration**

*Fixed Point Iteration is about rewriting equations to $x=f(x)$, and solve by starting from a $x_0$ and iteratively updating $x_{t+1} = f(x_t)$ until convergence.*

For example(from gemini), we can solve $x^2 - x - 1 = 0$ by iterative updating $x_{t+1} = \sqrt{x_t+1}$.

In our problem, we wish to write some $v = Bv$, where B is affine transformation(linear transformation + translation), so we can apply this method. We have already done that.  

$$v^\pi\leftarrow B^\pi\boldsymbol{v^\pi} = \boldsymbol{r^\pi} + \gamma \boldsymbol{P^\pi} \boldsymbol{v^\pi}$$

We need to prove that the solution $v^\pi$ is a unique fixed-point of $B^\pi$(only one solution) to use this iterative method. Note, this is a looser requirement than the problem is convex. 

Because we can write $x = f(x)$, that means that it's a fixed point and it'll eventually converge(for defined transformation $f$, there exists some $x$ that its value won't change after the transformation). 

We just need to show that for our problem $v^\pi$ is unique via showing the transformation $B$ is a contraction: by applying the transformation, the distance to any fixed points shrinks(and thus converges to 0, and this is not possible for multiple fixed points). 

**Proof of Contraction**

$$\begin{aligned}
& \vert \vert B^\pi v - B^\pi v' \vert \vert_{\infty} = \gamma \vert \vert P^\pi(v-v') \vert \vert_{\infty}\\
& \leq \gamma \max_x \Sigma_x' p(x'\vert x, \pi(x))\vert v(x) -v(x')\vert\\
& \leq \gamma \vert \vert v-v' \vert \vert_{\infty} \quad 0<\gamma < 1
\end{aligned}$$

*You might have already noticed that in our previous example, $x^2 - x -1 = 0$ has two solutions. And $x = \sqrt{x+1}$ would only select the positive one. Another formulation, $x = x^2 -1$ would not converge to either in iterative setting since it is not a contraction. For selecting the nagative roots: $x = \frac{1}{x+1}$ with a proper $x$. This is a local contraction(thus doesn't garuantee convergence, however, only converges to the negative root if any).*

We can, by induction, show that the convergence is exponentially fast. 

$$\vert \vert v_t^\pi - v^\pi \vert \vert_\infty \leq \gamma \vert \vert v_{t-1}^\pi - v^\pi \vert \vert_\infty =  \gamma^t  \vert \vert v_{0}^\pi - v^\pi \vert \vert_\infty $$

## Policy Optimization

We return to our goal of finding the optimal policy, $\pi$.  A policy is superior if it has a higher state value.

$$v^*(x) = max_\pi v^\pi(x); \quad q^*(x,a) = max_\pi q^\pi(x,a)$$

**From Greedy Policy to Bellman Optimality Equation**(*10.3.1*, *10.3.2*)

Note the greedy policy here is not about maximizing one step reward, but the long term value of an action(thus q or v value). 

$$\pi_q(x) = argmax_a q(x,a)$$

$$\pi_v(x) = argmax_a r(x,a) + \gamma \Sigma_{x'} p(x' \vert x,a) v(x')$$

We should be reminded that q and v functions are dependent on a specific policy. When we shift to a new policy, we have new q and v functions as well. 

It is intuitive that this optimization problem is an iterative process(fixed point iteration). There are two view of this optimization problem: 

- Policy Iteration: $\pi^*$ is the fixed point of the dependency between the greedy policy and the value function. This stress that: *The optimal policy is the one which maximize itself's q and v function(and thus $q^*$ abd $v^*$)*.

$$\pi^*(x) = argmax_a q^*(x,a)$$

- Value Iteration: $v^*$ is the fixed point of *Bellman Update* This stress that: *The value of the state under an optimal policy must be the expected return of the best action*.

$$v^*(x) = \max_a q^*(x,a) = \max_a r(x,a) + \gamma E_{x'\vert x,a}[v^*(x)]$$

$$q^*(x,a) = r(x,a) + \gamma E_{x'\vert x,a}[\max_{a'} q^*(x',a')]$$

Both perspectives should reach the same result(the optima), but with tradeoffs in other aspects.

**Policy Iteration** (*10.3.3*)

Consult the *Algorithm 10.14* for pesudo-code. Essentially, we start from a policy $\pi_0$ and iteratively compute the v-function $v^{\pi_t}$ and the optimal policy under the v-function $\pi_{t+1} \leftarrow \pi_v$.

Here, note that the book use two different notation for iteration steps: t for policy iteration(the outer loop) and T($\tau$) for policy evaluation(the inner loop). 

$$v_0 = v^{\pi_t}; \lim_{T\to \infty} v_T = v^{\pi_{t+1}}$$

We want to prove the convergence for this algorithm:

-  We know that we are working on deterministic polcies and we have one optimal solution(see beginning of the chapter)

- We wish to show proof of improvement(essentially the same as what we did last section to show that the affine transformation was a contraction). Monotonic Improvement implies that: (1) for all states, the resulting value is not worse; (2) And for at least some states, the resulting value is better.  

- For the policy evaluation step, we have already shown the v-function converges to the correct evaluation of a specified policy. We wish to plug in the definition of policy update and show(via induction) that the on every step, the evaluation of the new policy has a higher state value(v-value). And thus the new policy achieves a better state value than the starting policy. 

- If that is proved, it is evident for outer loop: the resulting policy of the new iteration is "superior" than the policy from previous iteration (except for cases where both are optima). (Monotonic Improvement)

$$v^{\pi_{t+1}} \geq v^{\pi_T}$$

*Proof of Improvement*

Base Case: 

$$v_1 = (Bv^{\pi_t})(x) = \max_a q^{\pi_t}(x,a) \geq q^{\pi_t}(x,a) = v^{\pi_t}(x) = v_0$$

Inductive step

$$\begin{aligned}
& v_{T+1}(x) = r(x, \pi_{t+1}(x)) + \gamma \Sigma_{x'} p(x' \vert x, \pi_{t+1}(x)) v_{T}(x')\\
& \geq r(x, \pi_{t+1}(x)) + \gamma \Sigma_{x'} p(x' \vert x, \pi_{t+1}(x)) v_{T-1}(x')\\
& = v_T(x)\\
\end{aligned}$$

Result: 

$$v^{\pi_{t+1}} = \lim_{T\to \infty} v_T \geq v_0 = v^{\pi_t}$$

$$v^{\pi_{t+1}} = v^{\pi_t} \space iff \space v^{\pi_{t+1}}  = v^{\pi_t} = v^*$$

<!-- the proof uses the old policy's value function  to define and justify the one-step improvement of the new policy, and then uses the new policy's Bellman operator to show that this improvement holds throughout the entire policy evaluation process. -->

**Value Iteration** (*10.3.4*)

For value iteration, we would focus on approximating the expected return of each actions. And after we converges to the correct v-value, we finalize our policy by selecting the best actions. Consult *Algorithm 10.17* for pseudo-code: the initialized v-value would be the maximum of action rewards. 

$$v(x) \leftarrow (B^*v)(x) = \max_a q(x,a)$$

We use the same Bellman update as before(in policy iteration) and we wish to prove that it is a contraction and thus $v^*$ is a unique fixed point. 

The book provides another view about the policy iteration: $v_t(x)$ is maximum expected reward when start from x and with a horizon t. This focus on the "reachable" states puts less computational effort in each iteration at the cost of not converging in finite steps. 

*Proof of Convergence*

$$\begin{aligned}
& \vert \vert B^*v - B^* v' \vert \vert_\infty \\
& = \max_x \vert B^*v(x) - B^* v'(x) \vert\\
& = \max_x \vert \max_a q(x,a) - \max_a q'(x,a)\vert\\
& \leq \max_x \max_a \vert q(x,a) - q'(x,a)\vert \\
& \leq \gamma \max_x \max_a \Sigma_{x'} p(x'\vert x,a) \vert v(x')-v'(x')\vert\\
& \leq \gamma \vert\vert v-v' \vert\vert_\infty
\end{aligned}$$

## Partial Observabable MDP (*10.4*)

We extend to partial observable rewards(noisy rewards), similiar to the noisy observations in GP. (PO-MDP)

Basically, we add observations $Y$ and observation probabilities: $o(y\vert x) = P(Y_t = y \vert X_t = x)$

And we have to introduce belief about current state, conditioned on observations and past actions: $b_t(x) = P(X_t = x\vert y_{1:t}, a_{1:t-1})$

PAI mentioned the Viterbi algorithm here, which we moved to the end of the note as the algorithm is not directly used. However, since both problems utilize an HMM setting. It might be helpful to read that part first. 

**Belief Update In PO-MDP**

Our primary goal here is not trying to recover a full hidden-state sequence(which Viterbi solves). Rather, we are trying to compute the current belief state.

The belief state can be updated using Bayes Rule:

$$\begin{aligned}
& b_{t+1} (x) = \frac{1}{Z} P(y_{t+1}\vert X_{t+1} = x) P(X_{t+1}\vert \{y, a\}_{1:t})\\
& = \frac{1}{Z} o(y_{t+1}\vert x) \Sigma_{x'} p(x\vert x', a_t) P(X_t = x'\vert \{y, a\}_{1:t})\\
& = \frac{1}{Z} o(y_{t+1}\vert x) \Sigma_{x'}  p(x\vert x', a_t) b_t(x')
\end{aligned}$$

The update only needs: $y_{t+1}, a_t$. You can find it very similiar to the dynamic programming idea in Viterbi.  

This update is deterministic: given the same action, observation and prior belief, we reach the same belief. 

*Although "beliefs" describes the uncertainty about the partially observable states, we are certain about how we shoule update it.*

**Belief State MDP**

With belief and belief update, we want to transform the PO-MDP problem, so that we can use algorithms we 
previously introduced in MDP. 

For each PO-MDP, we can see our belief state as observable states for an MDP. The reward of an action would be the expected reward(because we are uncertain about our beliefs).  Belief state are defined as a distribution on a time step:

$$B_t = X_t \vert y_{1:t},a_{1:t-1}$$
$$B_t \in \textbf{B} = \Delta^X =\{b\in R^{\vert X \vert}:b\geq 0, \Sigma_i b(i) = 1\}$$

Let us re-visit the components of MDP: a state, an action, a reward, and transition probability. Actions remains to be the same actions for PO-MDP, and state is now belief states. The reward would be the expected reward given our beliefs about the unexpected states. The transition probability can be defined using bayes rule as the transition probability between belief states given an action. 

Note, the computation of the belief update and reward are both deterministic. We would thus define a Belief State MDP for PO-MDP(see PAI for full definition):

*Reward*

$$\rho( b,a) = E_{x\sim b} [r(x,a)] = \Sigma_x b(x)r(x,a)$$

*Transition Probability*

$$\tau(b'\vert b,a) = P(B_{t+1} = b' \vert B_t = b, A_t = a)$$

$$\begin{aligned}
& \tau(b_{t+1} \vert b_t, a_t) = P(b_{t+1}\vert b_t, a_t)\\
& = \Sigma_{y_{t+1}} P(b_{t+1} \vert b_t, a_t, y_{t+1}) P(y_{t+1}\vert b_t, a_t) \\
\\
& P(b_{t+1} \vert b_t, a_t, y_{t+1}) = I_{b_{t+1} = b_{t+1}(x)} \\
& P(y_{t+1}\vert b_t, a_t) = E_{x\sim b_t} [E_{x\vert x', a_t}[o(y_{t+1} \vert x')]] \\
& = \Sigma_x b_t(x) \Sigma_{x'} p(x' \vert x, a_t) o(y_{t+1} \vert x')
\end{aligned}$$

We have shown that by we can map the PO-MDP problem to belief state MDP. 

*Up til now, we have solved for MDP(and PO-MDP) porblems with known transition probability and rewards.*

**Viterbi Algorithm to Solve HMM: A Side Quest (remark 10.21)**

I have been exposed to Viterbi Decoding, HMMs, etc. in [statistical NLP(Gildea)](https://www.cs.rochester.edu/~gildea/2022_Fall/notes.pdf). As an addition to the PAI book, you can find the pseudocode for Viterbi in *Algorithm 1: Viterbi Decoding* of that link. Let me, however, draw analogy between our current PO-MDP problem and the POS tagging problem in the note, and try my best to document viterbi algorithm. 

```
Our previous problem, MDP, was a Markov Chain(a sequence of observable states with transitional probability)

Hidden Markov Models(HMMs) introduces unobservable states(thus hidden states) and observations about the states at each time step. You can expect we know transitional probability between states and conditional probability between a state and an observation. The problem is to infer from the observations about the state sequence. 

The POS tagging is such a problem where you put grammmer tags(hidden states) to each words(observations) in a sentence. The transitional probability would be, for example, it is more likely for an object tag than a subject tag to appear after a verb tag. The conditional probability is that the word "dog" is more likely to be used as a subject or object rather than a verb. 

You can find these components for our PO-MDP problem. One thing is slightly different: the transitional probability is conditioned on a state and additionally an action(because an agent/policy is introduced!). 

You will have a prior for the starting state in our problem, and in the POS tagging, you might start from a flat prior, where all states are equally possible, or a certain "start of speech/sentence" token. 

The Viterbi Algorithm solve the HMM in tractable time by Dynamic Programming. Consider this data structure, we have a table(matrix) with each rows representing a probable states, and each columns represent a step in the sequence. The problem is thus transformed in to finding the most possible path within the table. (See the following figure)
```
![Science Direct Viterbi](https://ars.els-cdn.com/content/image/3-s2.0-B012227410500394X-gr14.gif)

All nodes in the table is initialized to be $-\infty$, thus equally impossible. The possibility is measured by log-likelihood in the table. 

For nodes in current layer(column), we would traverse through the node in the previous layer, and update the current node's log likelihood: $\max[\delta_{t,x}, \delta_{t-1,x}+tr_{x',x}]$

This will assign every node inside the table their largest log-likelihood, and allowing us to retrieve a path with the largest log-likelihood value. 

For decoding the sequence after registering the possibility of each nodes, we would start from the most probable node in the last layer and work our ways backward to retrieve nodes in the previous layers one by one. 

*The Viterbi Algorithm is tractable because it stores the internal values for calculating log-likelihood for each possible sequences in a data structure. These values are heavily reused(dynamic programming).*
