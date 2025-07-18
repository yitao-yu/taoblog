---
layout: page
title: "PAI 2: Active Learning(Entropy, Information Gain, Inductive learning and Transductive learning)"
permalink: /pai/active-learning-2
group: pai
---

For a start, active learning is about actively choosing data points to label next. Since the learner is in control of the data acquisition, we'll introduce several concepts to quantify "useful information". 

[Jump to this book mark if you are familiar with concepts in information theory(entropy, cross entropy, conditional entropy, mutual information, etc.)](#bookmark_activelearning)

## Entropy, etc.(Ch 5.4)

**Entropy is expected surprise:** $H[p] = E_{x\sim p}[S[p(x)]] = E_{x\sim p}[-log(p(x))]$

- discrete case: $H[p] = -\Sigma_x p(x)\log p(x)$
- continuous case: $H[p] = -\int p(x)\log p(x) dx$

Surprise is mapped from probability(both measures uncertainty): $S[u] = -log[u]$ 

Why this mapping? *Practically*, computers are better at adding very large numbers than multiplying very small floatingpoint numbers:

- more surprise means less occuring events(means "more surprise"): $u<v \to S[u] > S[v]; $

- Continuous and defined on $(0, \infty)$; ()

- $S[uv] = S[u] +S[v]$

**Cross Entropy** is average surprise of *data following assumed distribution*(q) and *data following groundtruth distribution*(p)

$H[p \vert q] = E_{x\sim p} [S[q(x)]] = E_{x\sim p} [-\log q(x)] = H[p] + KL(p\vert \vert q)$

Cross Entropy is also entropy of the groundtruth plus how different two distribution are(KL divergence). 

**KL-divergence**$(0,\infty)$ (measures difference between two probability distribution) is difference in cross entropy and entropy. 

$KL(p\vert \vert q) = H[p\vert \vert q] - H[p] = E_{\theta \sim p}[\log \frac{p\theta}{q\theta}]$

KL-divergence is not symmetric: $KL(q\vert \vert p)$ is reverse KL(as opposed to forward KL). 

**Minimizing KL and Reverse KL** (*5.4.5*)

We'll take it as an empirical fact that minimizing KL is more desirable because it's a more conservative objective. The book use the following example: 

For a ground truth distribution(p) of Gaussian Mixture(stacking of two gaussian distribution, resulting a distribution with two mode), minimizing KL would cause $q$ to select variance(center around mean); minimizing reverse KL would cause $q$ to select mode(greedy). (Figure 5.9)

However, that is because the true posterior is not contained in variational family $p \notin Q$ in the example. For a "less crazy" groundtruth, minimizing reverse KL still yields the true posterior. (eq 5.44)

$$argmin KL(q\vert \vert p(.\vert [x,y]_{1:n})) = p(.\vert [x,y]_{1:n})$$

**Minimizing KL is MLE**(*5.4.6*)

Suppose the likelihood is a parameterized model $q_\lambda(x)= q(x\vert \lambda)$, we want to show that minimizing KL is MLE with an infinitely large and iid dataset, $x$. 

{% raw %}
$argmin KL(p\vert \vert q_\lambda) = \argmax_\lambda lim_{n\to \infty} \frac{1}{n} \Sigma^n_i \log q(x_i\vert \lambda)$
{% endraw %}

Note that $H[p]$ is not parameterized on $\lambda$.

$$\begin{aligned}
& argmin_\lambda KL(p\vert \vert q) = argmin_\lambda H[p \vert \vert q] - H[p]\\
& = argmin_\lambda H[p\vert \vert q] + const \\
& H[p\vert \vert q] = E_{x\sim p} [-\log q(x)] = -\lim_n \frac{1}{n} \Sigma_n log(q(x_i \vert \lambda))
\end{aligned}$$

## Extending Entropy(Ch 8.1)

**Conditional Entropy**
$$\begin{aligned}
&H[X\vert Y] = E_{y \sim p(y)} [H[X \vert Y=y]]\\
& \quad = E_{x,y\sim p(x,y)}[-log(p(x\vert y))] \\
& H[X \vert Y=y] = E_{x \sim p(x \vert y)}[-log(p(x\vert y))] 
\end{aligned}$$

Conditional Entropy $H[X\vert Y]$ is measuring how much uncertainty will remain after that we (are yet to) observe y. So, **$p(y)$ is involved**. 

$H[X \vert Y=y]$ is average surprise of samples from the conditional distribution, or a probabilistic update of uncertainty in X given Y=y. In this case, **y is taken as a known variable**. 

Conditional Entropy is not symmetric(see next section). 

**Joint Entropy**

- Joint Entropy and chain rule

$$\begin{aligned}
& H(X,Y) = E_{x,y\sim p(x,y)}[-log(p(x, y))]\\ 
& \quad = H[Y] + H[X \vert Y] = H[X] + H[Y \vert X] \\
\end{aligned}$$

- Bayes Rule for entropy: 
$$\begin{aligned}
&H[X\vert Y] = H[X,Y] - H[Y] = H[Y\vert X]+H[X] - H[Y]\\
& \to H[X] - H[X\vert Y] = H[Y] - H[Y\vert X] 
\end{aligned}$$

The bayes rule is basically the same as probability, just as a reference: $P(X\vert Y) = P(X,Y)/P(Y) = \frac{P(Y\vert X)P(X)}{P(Y)}$

We finally find some symetric thing(mutual information) in information theory and we'll continue in next section. 

- information never hurts

Consulting probability, we can know that *information never hurts*(less surprise/uncertainty with more relevant observation): $H[X\vert Y] \leq H[X]$

## Mutual Information/Information Gain(Ch 8.2)

> Mutual Information measures the approximation error("information loss") when assuming X and Y are independent from each other.

$$\begin{aligned}
&I(X;Y) = H[X] - H[X\vert Y] = H[Y] - H[Y\vert X]\\ 
&= H[X] + H[Y] - H[X,Y] = I(Y;X)
\end{aligned}$$

**Information never hurts:** $I(X,Y) =  H[X] - H[X\vert Y] \geq 0$

Here's a visualization equivalent to fig 8.2 from a paper on research gate(*author: Choi et al.*). 

![Venn Diagram](https://www.researchgate.net/profile/Jae-Hoon-Cho/publication/263401621/figure/fig2/AS:667835558793221@1536235813351/Relation-between-entropy-and-mutual-information.png)

The Joint Entropy(total expected surprise) consists of three parts: mutual information and two conditional entropy. If X and Y are independent, $H(X\vert Y) = H(X); H(Y\vert X) = H(Y); I(X;Y) = 0$.

**Conditional Mutual Information** (*8.2.1*)

$$\begin{aligned}
& I(X;Y\vert Z) = H[X \vert Z] - H[X\vert Y, Z]\\ 
& = H[X,Z] + H[Y,Z] - H[Z] - H[X,Y,Z]\\
& = \textcolor{red}{H[X]} - I(X;Z) - \textcolor{red}{H[X\vert Y,Z]} = \textcolor{red}{I(X;Y,Z)} - I(X;Z)\\
& = I(Y;X, Z) - I(Y;Z) = I(Y;X\vert Z)
\end{aligned}$$

Here is a Venn diagram from wikipedia. 

![Wiki Venn diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5d/VennInfo3Var.svg/500px-VennInfo3Var.svg.png)

This is, however, *not a good illustration* because certain areas can be negative in this diagram. 

"Information never hurts" doesn't hold for Conditional Mutual Information. $I(A;B\vert C) \nleq I(A;B)$

When A, B are mutually independent, and are however correlated when conditional on C, $I(A;B\vert C) > I(A;B)$. This is the case when C is dependent on both A and B(generated by some calculation involves A and B), creating a path A and B on the casual graph between. In this case, $I(A;B\vert C) > 0$ because when considitioned on C, you are losing information to assume conditional independence. However, $I(A;B) = 0$. 

```
    [A]       [B]
      \       /
       \     /
         [C]
```

A, B are independent binary variable; $C = A\oplus B$(ChatGPT)

Or A, B are independent continuous random variable; $C=A+B$

Or $C=(A+B)mod2$([MathStack Exchange](https://math.stackexchange.com/questions/1219345/conditioning-reduces-mutual-information), , [MacKay Exercise 8.8](https://www.inference.org.uk/itprnn/book.pdf))

**Interaction Information and Synergy/Redundancy** (*8.2.1*)

$$I(X;Y;Z) = I(X;Y) - I(X;Y \vert Z)$$

Given above counter-example, interaction information can be larger or smaller than 0. 

*Also note, this is not symmetric.* You can use information theory to reconstruct some undirected graph(it cannot reveal the "casual" relation) between variables. 

- **Redundancy** between Y and Z with respect to X if $I(X;Y;Z) = I(X;Y) - I(X;Y \vert Z) > 0$

- **Synergy** between Y and Z with respect to X if $I(X;Y;Z) < 0$

## Maximizing Mutual Information(8.2.2, 8.3, 8.4)

<a id="bookmark_activelearning"></a>

**We wish to maximize mutual information in the active learning scenario:** $I(S) = I(f_S;y_S) = H[f_S] - H[f_S\vert y_S]$ where $y_S = f_S + \epsilon$. 

This is equivalent to optimal decision for this reward: 

$$r(y_S,S) = H[f_S] - H[f_S \vert Y_S=y_S]; $$
$$I[S] = E_{y_S}[r(y_S,S)]$$

Note: *$y_S$*, however, might not be observable at the time of selection. 

**Maximizing Mutual Information Over the Choice of a Subset is NP-hard**

There are $nCk$ possible subsets, which is a lot. 

A greedy approach(iteratively select the data item with max information gain) would be tractable. However, there is not yet a garuantee that it would reach global or local maxima.

**Marginal Gain:** describes how much "adding" x to a set A increases F. 

$$x\in X, A \subseteq X; \Delta_F(x\vert A) = F(A\cup \{x\}) - F(A)$$

For information gain:

$$\Delta_I(x\vert A) = I(f_x; y_x\vert y_A) = H[y_x\vert y_A] -H [\epsilon]$$

**Submodularity:** a function is submodular iff for $A \subseteq B \subseteq X$

$$\Delta_F(x\vert A) \geq \Delta_F(x\vert B)$$

For information gain(and our problem), this means that for a larger set as opposed to a smaller set, the information gain of knowing an additional observation has "diminishing return". 

**Monotone:** a function is monotone iff for $A \subseteq B \subseteq X$

$$F(A) \leq F(B)$$

For information gain, this means that having more data would only reduce uncertainty. 

Both $\Delta_I(x\vert A) \geq \Delta_I(x\vert B)$ and $I(A) \leq I(B)$ are satisfied for mutual information. (See proof for Theorem 8.11) This justifies the greedy approach. 

The book proves that the greedy approach will reach a decent local maxima(>60% of global maxima) if the objective is monotone submodular, we'll skip this part. (*8.4*, Theorem 8.12)

$$x_{t+1} = argmax_x \Delta_I(x \vert S_t) = argmax_x I(f_x;y_x\vert y_{S_t})$$

**Mutual Information of Gaussians** (*Example 8.4, Equation 8.13*)

Suppose $X \sim N(\mu, \Sigma)$ and $Y = X+\epsilon, \epsilon \sim (0,\sigma^2_n I)$

$$\begin{aligned}
& I(X;Y) = I(Y;X) = H[Y] - H[Y\vert X] \\
& = H[Y] - H[\epsilon] \\
& = \frac{1}{2} log((2\pi e )^d \det[\Sigma+\sigma_n^2I]) - \frac{1}{2}log((2\pi e )^d \det[\sigma_n^2I]) \\
& = \frac{1}{2} log \det[I+\sigma^{-2}_n\Sigma]
\end{aligned}$$
<!-- det(AB) = det(A)det(B) -->

The larger the noise(noise variance), the smaller information gain of observation. 

**Uncertainty Sampling**(*8.4.1*): We simply pick points with largest variance at each step for our greedy algorithm. 

Given previous result, assuming Gaussian **as well as label(data generation) noise is independent of x(homoscedastic)**: 

$$\begin{aligned}
% &x_{t+1} = argmax_x I(f_x;y_x,y_{S_t}) - I(f_x; y_{S_t})\\
& x_{t+1} = argmax_x \frac{1}{2} log(1+\frac{\sigma^2_t(x)}{\sigma^2_n} )\\
& = argmax_x \sigma^2_t(x)
\end{aligned}$$

**Heteroscedastic Label noise**: with out the homoscedastic noise assumption(which can be strong), we wish to find 

$$\begin{aligned}
&x_{t+1} = argmax_x \frac{1}{2} log(1+\frac{\sigma^2_t(x)}{\sigma^2_n(x)}) = argmax_x \frac{\sigma^2_t(x)}{\sigma^2_n(x)}\\
\end{aligned}$$

Note that one is aleacratic uncertainty $\sigma^2_n(x)$ and the other is epistemic uncertainty $\sigma^2_t(x)$. We have a tradeoff and wish to find locations where epistemic uncertainty is large and aleacratic uncertainty is small.

**Classification(Discrete Case)**: for discrete classes(as opposed to regression), we wish to select samples that maximize the entropy of the predicted label. 

$$\begin{aligned}
& x_{t+1} = argmax_x H[y_x\vert x_{1:t}, y{1:t}] \\
& = argmax_x I(\theta; y_x \vert x_{t+1}, y_{1:t}) \\
& = argmax_x H[y_x \vert x_{1:t}, y_{1:t}] - H[y_x \vert  \theta, x_{1:t}, x_{1:t}, y_{1:t}] \\
& = argmax_x H[y_x \vert x_{1:t}, y_{1:t}] - E_{\theta \vert x_{1:t}, y_{1:t}} H[y_x \vert \theta, x_{1:t}, y_{1:t}]  \\
& = argmax_x H[y_x \vert x_{1:t}, y_{1:t}] -  E_{\theta \vert x_{1:t}, y_{1:t}} H[y_x \vert \theta]
\end{aligned}$$

The last step assumes iid between all data points. 

- The first term is entropy of predictive posterior. 

- The second term is entropy of likelihood, penalizing aleatoric uncertainty. 

## Tranductive Active Learning

If we wish to optimize for the best performance of f(x^*) at a specific location $x^*$: 

**Inductive Learning**: select a diverse set of example, allowing the model to compress most general information. 


**Transductive Learning**: trade-off between *relevance*(to $x^*$) and diversity. The fist term measures relevance and the second term measures redundancy. 

$$I(f_{x^*};y_x\vert x_{1:t}, y_{1:t}) = I[f^*_x; y_x] - I(f^*_x; y_x; y_{1:t}\vert x_{1:t})$$

Also, we can express the optimization using conditional entropy(entropy search): 

$$argmax_x I(f_{x^*};y_x\vert x_{1:t}, y_{1:t}) = argmin_x H[f_{x^*} \vert x_{1:t},y_{1:t},y_x]$$

