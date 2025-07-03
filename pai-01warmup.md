---
layout: page
title: "PAI 1: Warmup(Bayes Linear Reg, Kernel Trick, GP)"
permalink: /pai/warmup-1
group: pai
---

## Decision Theory(*Ch 1.4*)

Choosing the optimal action is about choosing the action that would maximize the expected reward. *We are interested in learning the reward function in later parts.*

**Optimal Decision Rule**: $a^{*}(x) = \text{argmax}_a E[y\|x] [r(y,a)]$. 

Equivalently, to see $-r$ as loss, we can also also acquire the optimal decision by minimizing the loss and it's a regression problem if we are making a decision in a continuous space. 

The book discussed mostly how square loss(symetrical reward) and asymmetrical loss would affect the optimal decision. (Example 1.29)

It is quite clear and intuitive that the optimal action with a symetrical reward is the mean: $a^⋆(x) = E[y \| x]$ where loss is $-r(y,a) = (y-a)^2$. The derivation should be similiar to the following case. 

*Problem 1.13*: We then consider the asymmetrical loss/reward: $-r(y,a) = c_1 * max[y-a,0] + c_2 * max[a-y,0]$

The expected reward is:  

$$E[r] = c_1 \int_a^\infty (y-a) p(y\|x) dy + c_2 \int_{-\infty}^a (a-y) p(y\|x) dy$$

Here we assume that the random variable $y\|x \sim N(\mu, \sigma^2)$ (the observation noise follows a normal distribution). We would further map this distribution to standard gaussian: $z = \frac{y-\mu}{\sigma} \sim Z(0,1)$.

And we wish to find local maxima of $\alpha$ (action/prediction mapped to standard gaussian) so that $\frac{d E[r(y,\alpha)]}{d\alpha}  = 0$, where $\alpha = \frac{a - \mu}{\sigma}$. And we can rewrite the reward into: 

$$E[r] = c_1 \int_\alpha^\infty (z-\alpha) \phi(z) dz + c_2 \int_{-\infty}^\alpha (\alpha-z) \phi(z) dz$$

We need to rely on the Leibniz Rule [*1*](#leibniz-rule) to calculate the derivative: 

$$\frac{d }{d \alpha} \int^{b(\alpha)}_{a(\alpha)} f(z,\alpha) dz = f(b(\alpha), \alpha) \frac{db}{d\alpha} - f(a(\alpha), \alpha) \frac{da}{d\alpha} + \int^{b(\alpha)}_{a(\alpha)} \frac {\delta f } {\delta \alpha} dz$$

First Term, $a(\alpha) = \alpha, b(\alpha) = \infty, f = (z-\alpha)\phi(z); L_1 = c_1 \int_\alpha^\infty(z-\alpha)\phi(z)dz$: 

$$\frac{d L_1}{d \alpha} = c_1*(-f(\alpha, \alpha)+\int_\alpha^\infty -\phi(z)dz) =  - c_1*(1-\Phi(\alpha))$$

Second Term, Similarly, $a = -\infty, b = \alpha, f = (\alpha-z)\phi(z)$: 

$$\frac{d L_2}{d \alpha} =  c_2(-f(\alpha, \alpha)+\int_{-\infty}^\alpha\phi(z)dz) = c_2\Phi(\alpha)$$

Sum of the derivative is zero would give us the optimal decision(mapped from optimal $z$): 

$$\begin{align*}
c_2\Phi(\alpha) - c_1*(1-\Phi(\alpha)) = 0 \\
\to\Phi(\alpha) = c_1/(c_1+c_2) \\
\to a^* = \mu + \sigma*\Phi^{-1}[\frac{c_1}{c_1+c_2}]
\end{align*}$$

#### Leibniz Rule

Case 1(constant bounds) is intuitive: 

$u'(x) = \int_a^b f_x(x,t) dt$ where $u(x) = \int_a^bf(x,t)dt$

$$\begin{align*}
u'(x) = lim_{h\to0} \frac{u(x+h) - u(h)}{h} \\
 = \int_a^b lim_{h\to0}\frac{f(x+h,t)-f(h,t)}{h}dt \\
 = \int_a^b \frac{df(x,t)}{dx} dt\end{align*}$$

Case 2(function bounds) can be derived with the same method: 

$$u'=\frac{d}{d\alpha}\int_{a(\alpha)}^{b(\alpha)}f(z,\alpha)dz = lim_{h\to0}\frac{1}{h}[\int_{a(\alpha+h)}^{b(\alpha+h)} f(z,\alpha+h) dz- \int_{a(\alpha)}^{b(\alpha)} f(z,\alpha) dz]$$

By adding and subtracting $\int_{a(\alpha)}^{b(\alpha)}f(z,\alpha+h)dz$: 

$$\begin{align*}
u'= lim_{h\to0}\frac{1}{h}[\int_{a(\alpha+h)}^{b(\alpha+h)} f(z,\alpha+h) dz - \int_{a(\alpha)}^{b(\alpha)}f(z,\alpha+h)dz \\
+ \int_{a(\alpha)}^{b(\alpha)}f(z,\alpha+h)dz - \int_{a(\alpha)}^{b(\alpha)} f(z,\alpha) dz] \\
= f(b(\alpha,\alpha)) \frac{db}{d\alpha} −f(a(α),α) \frac{da}{d\alpha} + \int_{a(\alpha)}^{b(\alpha)}\frac{\delta f(z,\alpha)} {\delta \alpha} dz
\end{align*}$$

## Linear Regression

The section II of the book is about fitting model to the perception(or data) about the world. The first method introduced is, of course, linear regression: $f(x;w) = w^T x$. 

This part of the book is about introducing concepts like MLE vs MAP, and aleatoric vs epistemic uncertainty. The book shows that the MLE estimate of the weight in the linear regression model is equivalent to ordinary least square(or minimizing MSE). And the MAP estimate(with a Gaussian prior) is equivalent to minimizing the ridge loss. 

For proving that, key assumptions are:
- dataset is i.i.d. 
- Gaussian noise

By Bayes rule, we can write the full posterior that we are minimizing. (MLE would just minimize the likelihood term, marked red, and MAP would minimize the full posterior term)

$$\log p(w | x_{1:n}, y_{1:n}) = \log p(w) + \textcolor{red}{\log p(y_{1:n}|x_{1:n}, w)} + const$$

**Deriving least square:**

$$\hat{w}_{ls} = argmin_w ||y-X_w||^2_2 = (X^TX)^{-1}X^Ty$$

The sum of each individual term is what we wish to minimize(i.i.d.) in MLE. 

$\log p(y_i\|x_i,w) = log \frac{1}{\sqrt(2 \pi \sigma_n^2)} \exp(-\frac{1}{2 \sigma^2_n} (y_i - w^T x_i)^2)$ 

Problem(2.2): Prove the variance for the noise term is: $\sigma_n^2 = \frac{1}{n} \Sigma_i (y_i-w^Tx_i)^2$

$$\log p = -\frac{n}{2} \log(2\pi \sigma_n^2) - \frac{1}{2\sigma_n^2} \Sigma_i(y_i - w^T x_i)^2 \\
\to \frac{d \log L}{d \sigma_n^2} = -\frac{n}{2\sigma_n^2}+\frac{\Sigma_i(...)}{2(\sigma_n^2)^2} = 0 \\
\to \sigma_n^2 = \frac{1}{n} \Sigma_i(y_i-w^Tx_i)^2$$

*We would skip writing the actual MLE estimate since it'll be similiar to writing MAP estimate(which we'll do later): taking derivative of loss.*

**Deriving ridge regression:**

*The book(2.9) expanded the term of full posterior and showed that the full posterior is also gaussian(quadratic form: $w^TAw+B^Tw+C$).(The grey part is merged into the constant term)*

$$
\begin{aligned}
&\log p(w | x_{1:n}, y_{1:n}) = -\frac{1}{2} [\sigma_p^{-2} ||w||^2_2 + \sigma_n^{-2} \textcolor{red}{||y-Xw||^2_2}] \\
&\to \textcolor{red}{||y-Xw||^2_2} =  w^TX^TXw + 2y^TXw + \textcolor{grey}{y^Ty} \\
&\to \log p(w | x_{1:n}, y_{1:n}) = -\frac{1}{2} w^T \Sigma^{-1} w + w^T \Sigma^{-1}\mu + \text{const} \\
&\to w_{|x,y} \sim N(\mu, \Sigma) \\ 
&\mu = \sigma_n^{-2} \Sigma X^Ty \\
&\Sigma = [\sigma_n^{-2}X^TX+\sigma_p^{-2}I]^{-1}
\end{aligned}
$$

We go back to deriving the ridge loss(and the $\lambda$ term): 

$$\hat{w}_{ridge} = argmin_w ||y-X_w||^2_2 + \lambda ||w||^2_2 = (X^TX + \lambda I)^{-1} X^T y$$

$$\hat{w}_{MAP} = argmax -\frac{1}{2} \sigma_p^{-2} ||w|| + \sigma_n^{-2} ||y-Xw||+const \\ = argmin ||y-Xw|| + \frac{\sigma_n^2}{\sigma_p^2}$$

*We would skip Lasso(Example 2.2)*: If we would assume a Lasso prior $w\sim Laplace(0,h)$, we'll acquire a Lasso term instead of ridge term: $\frac{\sigma_n^2}{h}\|\|w\|\|^1_1$.

Recall the expanded loss: $\frac{1}{2} w^T A w - b^Tw$ where $A = \sigma_n^{-2}X^TX + \sigma_p^{-2}I; b = \sigma_n^{-2}X^Ty$ 

And we can write the MAP estimate: 

$$\Delta_w L = Aw-b = 0 \to Aw = b \\ [X^TX + \frac{\sigma_n^2}{\sigma_p^2}I]w = X^T y \\ \to \hat{w}_{MAP} = [X^TX +\frac{\sigma_n^2}{\sigma_p^2}I]X^Ty$$

**Prediction with the full weight posterior(2.1.2): Bayeisna LR**

The book has also mentioned that instead of using a point estimate of the weight, we can use the full posterior of the weight to acquire a distribution of possible target values, instead of the best estimated target value. 

$$f^* | x^*, x_{1:n}, y_{1:n} \sim N(\mu^T x^*, x^{*T} \Sigma x^*) \\
y^*| x^*, x_{1:n}, y_{1:n} \sim  N(\mu^T x^*, x^{*T} \Sigma x^*+\sigma_n^2)$$

$f^*$ is the **actual distribution or BLR prediction**, however, $y^*$ is the **predicted label**, taking account of the the data generation noise(gaussian noise). I want to help with this distinction:

$$
p(f^*|x^*,x_{1:n},y_{1:n}) = \int p(f^*|w,x^*) p(w|x_{1:n},y_{1:n}) dw\\

p(y^*|x^*,x_{1:n},y_{1:n}) = \int p(y^*|f*) p(f^*|x^*,x_{1:n},y_{1:n}) dw
$$

where $p(y^*\|f*)$ is the data generation process(normal distribution). In other word, $f^*$ is the theoretical distribution of the target value. However, this is made more uncertain by the data generation noise. $f^*$(BLR prediction) is the prior for $y^*$(the actual prediction) and the data generation distribution is the likelihood. 

This would allow us to have a **varying distribution over the feature space**: more certain when there are more observed data around the test point, less certain when there are less. The book didn't go into a lot of detail and I wish to show this feature. 

First we recall that w's posterior distribution is Gaussian.  This is not entirely the same because we chose to use a isotropic prior: $w\sim N(\mu,\Sigma_0); \Sigma_0 = \alpha^{-1}I$. 

$$
w_{|x,y} \sim N(\mu_N, \Sigma_N) \\
\mu_N = \Sigma_N (\Sigma_0^{-1}\mu_0 + \sigma_n^{-2}X^Ty)\\
\Sigma_N = (\Sigma_0^{-1} + \sigma_n^{-2}X^TX)^{-1}
$$

<!-- 
$$w_{|x,y} \sim N(\mu, \Sigma) \\ 
\mu = \sigma_n^{-2} \Sigma X^Ty; \\
\Sigma = [\sigma_n^{-2}X^TX+\sigma_p^{-2}I]^{-1}$$ 
-->

$X^TX$ would represent the density of observed data point. With more data points(larger $i$),$\Sigma_i x_i x_i^T$ tends to have a large value, and thus a larger $X^T X$ matrix. This would result in smaller eigen value for $\Sigma_N$, thus lower variance along the principal axes of the distribution. 

In this process, the center of the posterior distribution or data generation noise are not changed. The change in variance is brought in merely by the $X^TX$ term. 

*Here we skip the online recursive BLR, mostly the part about its computational complexity(2.1.3) because it is not entirely clear to me.*

**Aleatoric and Epistemic Uncertainty(2.2)**

Aleatoric and Epistemic Uncertainty is different from bias-variance tradeoff from applied ML or data science. Bias-variance tradeoff is about choosing the "right" complexity of the model. Aleatoric and Epistemic Uncertainty is about how uncertain we are about our prediction and where does these uncertainty comes from. 

Aleatoric and Epistemic Uncertainty are caused by different things. 

Recall from BLR: 
$$y^*| x^*, x_{1:n}, y_{1:n} \sim  N(\mu^T x^*,\textcolor{blue}{\sigma_n^2}+\textcolor{red}{x^{*T} \Sigma x^*})$$

The blue part(the data generation noise term) is aleatoric uncertainty, or uncertainty caused by labels. The blue part(previously the variance for BLR prediction) is epistemic uncertainty due to lack of data(it grows larger when there are less data);. 

Let's rewrite the two term of the variance as the book did(2.18) so that we can expand the idea to models other than LR: 

$$Var(y^*|x^*) = \textcolor{blue}{E_\theta[Var_{y^*}[y^*|x^*,\theta]}+\textcolor{red}{Var_{\theta}[E_{y^*}[y^*|x^*,\theta]]}$$

I quote the book(with some modification): "Aleatoric uncertainty is expected variance of $y^*$ across all models, and Epistemic uncertainty is the expected variance of mean prediction under each model. "

<!-- Come back Later -->

## Nonlinear LR: Kernel Trick

Most of the times our observed feature and target is not linear(or they might only be linear after we mapped them to a feature space). 

## Karman Filter

## Gaussian Process

*We would skip the Bayesian NN part in the book(for now). However, I'm sure that I will come back to this part because the topic is interesting to me.*