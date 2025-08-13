---
layout: page
title: "PAI 4: Tabular RL"
permalink: /pai/tabular-rl
group: pai
---

We'll now step into the reinforcement learning problem, where the environments(rewards, transition probability) is unknown. 

- This brings us back to the exploitation-exploration dilemma. And we wish to minimize regret. 
- The observed data(rewards and noisy observations about state) is dependent on played actions. 
- We stick to the vanilla MDP, where the state is known. However, it has been shown that PO-MDP can be mapped to MDP. 

For this chapter, we focus on the case where number of states and actions are small. So we can hold a look-up table for Q or V value(thus "tabular"). 

## RL: On-policy vs. Off-policy Methods

**Trajectory** is a sequence of transitions:

$$\tau_i = (x_i, a_i, r_i, x_{i+1})$$

The transitions are not i.i.d. Howeverm new observed states and rewards are conditionally independent (across items/transitions) to old states and actions. 

- Not i.i.d.(each state is dependent on previous states, etc.) violates the assumption of many supervised ML algorithms. 
- However, conditional independence is good enought for *Law of Big Numbers* and *Hoeffding's inequality*(we'll briefly go through this later). 

**Data Collection: Episodic vs. Continuous**

- Episodic setting means the agent goes through a sequence of training rounds. It's like learning by playing a game multiple rounds, and reset to the intialized position at the start of every rounds. 

- Continuous settings means the agent learns online, in a stimulated or realworld environment. It's like learning while playing a long game.  

**Data Collection: On-policy vs. Off-policy**

- On-policy methods means that the agent has control over which policy it follows. 

- Off-policy methods means that the agent cannot choose its own action. Rather, it learns from observational data. 

**Model-based vs Model-free Approaches**

- Model-based approaches learns the underlying MDP

- Model-free approaches learn the value function directly

## Model-Based Approaches(11.2, 11.3)

We have following MLE: 

*for transitions*

$$\hat p (x' \vert x,a) = \frac{N(x'\vert x,a)}{N(a\vert x)}$$

*for rewards*

$$\hat r (x,a) = \frac{1}{N(a\vert x)} \Sigma_t r_t(x_t=x, a_t=a)$$

*Note: law of large numbers is good enough for acquiring simple statistics like means. Essentially, Hoeffding's inequality simply states that the probability of a new observation to deviate from mean more than $\epsilon$ is smaller than some value(and as sample size grows larger, that values grows exponentially smaller).*

However, the problem of balancing explorations and eploitations arised from introducing uncertain environments remains to be answered. 

Here are a few strategies

**$\epsilon$-greedy**

Consult *Algorithm 11.2* for pseudo-code. 

Essentially, sample a value from a $(0,1)$ uniform distribution. If the sampled value is smaller than a threshold $\epsilon_t$, explore by picking a random action; else, exploit by selecting the best action. 

Obviously:

- It's simple.
- Longer into the game, you can tune $\epsilon$ to smaller values to encourage exploitation. (See remark 11.3 for the RM-conditions which the sequence should satisfy to converge to an optimal policy)
- We can utilize the inforation we have about the environment to decide what we are uncertain about and should explore(like we did in active learning), instead of randomly choosing an action to explore. 

**Softmax Exploration/Blotzman Exploration**

<!-- Don't be intimidated. This has nothing to do with Boltzman machine LOL -->

Consider this an update to $\epsilon$-greedy. 

We sample an action with probability(the probability is calculated divided this term by a normalizer):

$$\pi_\lambda (a\vert x) \propto \exp(\frac{1}{\lambda} Q^*(x,a))$$

As $\lambda$(temperature parameter) goes to zero, the weight assigned to the Q functions grows larger as the normalizer, and the model leans toward exploitation. 

- This is an update because when exploration, it prioritize the actions with better estimated Q-values. 

**$R_{\max}$ Algorithm: "Optimism in the face of Uncertainty"**

See *Algorithm 11.6* for pseudo-code. 

Essentially, we imagine a fantasy state $x^*$. The reward of fantasy state is the largest reward $R_{\max}$; and you stays there after reaching the state. This means that trying its best to reach this state is the best thing your agent can hope for. 

$$\forall a, r(x^*, a) = R_{\max}; \forall a, p(x^*\vert x^*,a)=1$$

Optimism here means that you believe that not fully explored actions would lead to some hidden big prizes(the fantasy state you imagined). 

To encourage exploration, we initialize our model to be:

- Every action from every state would lead to this fantasy state
- Every state has a reward of $R_{\max}$
- compute a exploration-favored policy based on that

Then, we iterate to execute policy for several rounds, update rewards and transition; and compute new policies. 

Our model will "drop the fantasy" over time and lean to exloitation after enough exploration. 

*Hoeffding's Inequality* gives a lower bound for each state-action pair to be "visited enough", and thus garuantee convergence. 

$$\begin{aligned}
& P(X_n - E X \geq \epsilon) \leq 2 \exp(-\frac{n \epsilon^2}{2 \sigma^2}) \\
& n \geq \frac{2\sigma^2}{\epsilon^2} log \frac{2}{\delta} \\
& \to n_{a\vert x} \geq \frac{R^2_{\max}}{\epsilon^2}log \frac{2}{\delta}
\end{aligned}
$$

**Drawbacks**

- The tables has $O(n^2 m)$ entries: for transition $p(x'
 \vert x, a)$, it has to store for each combination of current state, new state and actions. 

- We need to solve MDP(policy or value iteration) for optimal policy. We need to compute the best policy $O(nm)$ times(to have enough observation for state-value pairs). On the other hand, model-free approaches would store the q or v values directly. 

## Model-Free Approaches

In model-based methods, learning transition and rewards are very expensive. However, these are general information about the game and can be used for deriving any policy. Most of our previous introduced model-based methods are in on-policy learning settings. However, there are existing trajectories in the off-policy setting, and you can acquire MLE for rewards and transitions from there(and compute policy for on-line play). 

In model-free setting, we are learning value of each states, and as seen in the previous chapter, The state and state-action value are specific to policies. Some model-free methods are on-policy and others are off-policy. 

**Temporal Difference Learning(On-policy Value Estimation)** (*11.4.1*)

Here, TD-learning is addressing how to update our estimate about V or Q function for state when playing a game with a certain policy and observing a transition. 

You can compare TD-learning to two things: Monte Carlo Sampling or Exponential Moving Averages(EMA). 

When using Monte Carlo sampling to estimate a function value, we would sample a random variable from a distribution, evaluate the function of interest, repeat the process, and take average. 

The problem here is that:

- We are dealing with a somewhat black box of state value, given a policy, without any models of reward or transition. And the best we can do is to *sample one action at a time from the policy* and observe reward. We don't even have observation about the state value of the state or next state. 
- To calculate the state value with a known reward, you need to know the state value of the next state. 

To solve that: we use previous estimations for next state to update the estimation of this state. Over playing, we observe many data and we "hope" for the convergence. (The book garuantees the convergence if the learning rate sequence satisfies the RM-condition). 

Also, we mix in the empirical value, and use a learning rate to slowly update the estimated value. 

This is the value update in TD-learning(*Algorithm 11.10*):

$$V^\pi(x) \leftarrow (1-\alpha_t) V^\pi(x) + \alpha_t(r+\gamma V^{\pi} (x'))$$

*Note: we are essentially taking EMA of the value function(if the learning rate is constant).*

**SARSA(On-Policy Control)** (*11.4.2*)

We'll still need to use some sort of policy iteration to find an optimal policy. 

To expand the TD-learning to Q functions, the problem is that we don't have an explicit model of transition. So, we cannot compute the expected value of next state value for one action. 

However, if we observe two consecutive transitions(so next state's estimated state-action value can be retrieved), we can write:

$$Q^\pi(x,a)\leftarrow (1-\alpha_t) Q^\pi (x,a) + \alpha_t (r+\gamma Q^\pi (x',a))$$

Same guarantee of convergence actually holds for SARSA as well. This makes sense because as you observes enough transitions, you are naturally approximating the expected state value for next state. 

The estimated Q-value of SARSA is policy-specific. We use these values in policy iteration and cannot reuse the estimations between iterations: 

- We estimate $Q^{\pi_t}$ with SARSA
- We update $\pi_t$ greedily and repeat the process

*Note*:

- The method may not explore enough with finite samples
- We can apply noise($\epsilon$-greedy) when choosing actions as compensation

**Off-Policy Value Estimation**(*11.4.3*)

We wish to update SARSA to off-policy settings. So that we can reuse the transitions observed in previous policy iterations. This upate rule is off-policy:

$$Q^\pi(x,a)\leftarrow (1-\alpha_t) Q^\pi (x,a) + \alpha_t (r+\gamma \Sigma_{a'} \pi(a'\vert x') Q^\pi (x',a'))$$

Note that we would only need one observed transition for this update. The estimated q-value is still dependent on policies it follows.

**Q-learning(Off-Policy Control)**(*11.4.4*)

For learning q functions without a policy(thus no $\pi(a\vert x)$), we choose the action with maximum estimated q-value. This idea is similiar to value iteration. 

$$Q^\pi(x,a)\leftarrow (1-\alpha_t) Q^\pi (x,a) + \alpha_t (r+\gamma \max_{a'} Q^* (x',a'))$$

Consult *Algorithm 11.12*, in this Q-learning setting, the agent learns from observation instead of playing the game itself. 

**Optimistic Q-learning**(*11.4.5*)

We can introduce "optimism in the face of uncertainty" into Q-learning. 

We add an assumption $0 \leq r \leq R_{\max}$. 

Consult *Algorithm 11.14*. In on-policy setting, we explore the space by initializing state-action values uniformly to large value, and update the estimation over iterations and picking the action with largest state-action value. (Like in $R_{\max}$ algorithm, the agent will transition into exploitation after enough exploration). 

The initialization is(repeated multiplication to inflate the value for it to not decay very fast over iteration):

$$Q^*(x,a) \leftarrow V_{\max} \Pi^{T_{init}}_{t=1}(1-\alpha_t)^{-1}$$
$$V_{\max} =\frac{R_{\max}}{1-\gamma} \geq \max q^*(x,a)$$

The book proves that for initialization steps(of each state-action pair), $Q_t\geq V_{\max} \geq q^*(x,a)$. (*Lemma 11.15*)

*Note*

Q-learning requires $O(nm)$ memory(slot for each state-action pair) and $O(Tm)$ time complexity because we need to take maximum q-value over actions for each iteration. 

For next two chapters, the book moves on to games with big state and action spaces. 
