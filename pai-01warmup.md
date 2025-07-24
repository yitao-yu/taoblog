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
\to \sigma_n^2 = \frac{1}{n} \Sigma_i(y_i-w^Tx_i)^2 \\$$

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

**Prediction with the full weight posterior(2.1.2): Bayeisna LR** (*This part needs edit because I think I introduced a prior when I shouldn't need to*)

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
\begin{aligned}
&w_{|x,y} \sim N(\mu_N, \Sigma_N) \\
&\mu_N = \Sigma_N (\Sigma_0^{-1}\mu_0 + \sigma_n^{-2}X^Ty)\\
&\Sigma_N = (\Sigma_0^{-1} + \sigma_n^{-2}X^TX)^{-1}
\end{aligned}
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

Most of the times our observed feature, $x$ , and target is not linear, but they might be linear in some feature space, $\phi(x)$. We would transform the linear regression into: $f=\phi w$, $w \sim N(0,\sigma_p^2 I)$. We would acquire this prior about the function given input: 

$$f| X \sim N(0,\sigma_p^2 \phi \phi^T)$$

It can be, however, costly to figure out a best set of feature and compute the feature vector for each entry during training. 

The book used these letter to denote dimensions: 

$d$: size of the observed features. 

$e$: size of the mapped features. (or equivalently, weight vector). $e$ can be $O(d^{power})$ if using polynomial features, so it can be large. 

$n$: size of the dataset/entry. Actually on a second thought this can also be a large number. Maybe it can be 100, 1k or 10k times $d$, so $O(d)$, to prevent overfit? 

The feature matrix(concatenation of feature vector for each entry) is $n\times e$; the Kernel Matrix(see below) is $n \times n$. The dataset it self we worked with in simple LR is $n \times d$. 

**Kernel Function**

$$k(x,x') = \sigma_p^2 \phi(x)^T \phi(x') = Conv(f(x), f(x'))$$

**Kernel Matrix(nxn)**

The covariance matrix for the previous prior is Kernel Matrix. The fact that it is $n\times n$ crucial since **we can operate with this kernel matrix directly with the training dataset, instead of computing the feature vectors $\phi$ for every entry and compute kernel matrix from them**. This is the "Kernel Trick". **Also, since the feature vector is not computed at all, we are not finding optimal weights, rather, we are finding an optimal Kernel Matrix.** 

$$K = \sigma_p^2 \phi \phi^T \\ K_{ij} = k(x_i, x_j); i,j<n$$

*The book says kernel matrix is a finite view of the kernel function. It means that kernel matrix contains values for every entry pair in the training dataset. At inference or testing, we will work with unobserved input(outside of the training dataset), and the book will take us into it(2.4.1 and 4.1).*

*Also, note that this is just a prior and the actual target value donot have to be center around zero, just like the mean weight of a simple LR model won't necessarily be zero.*

The book listed the potential benefit of the kernel trick(2.4.2). Kernel trick will be revisited in Ch. 4(GP) so we'll come back to it. 

<!-- >1. For certain feature transformations φ, we may be able to find an easier to compute expression equivalent to φ(x)⊤φ(x′). 
>2. If this is not possible, we could approximate the inner product by an easier to compute expression. 
>3. Or, alternatively, we may decide not to care very much about the exact feature transformation and simply experiment with kernels that induce some feature space (which may even be infinitely dimensional).  -->


**Kernelized Bayesian LR(2.4.1)**

> **Feature matrix:** $\hat{\Phi} = \begin{bmatrix} \Phi_{n \times e} \\ \phi(x^*)^T \end{bmatrix}$

$n\times e$ design matrix/training dataset, and the vecotr for unobserved data point

>**target groundtruth:** $\hat{y} = \begin{bmatrix} y \\ y^* \end{bmatrix}$

*$y^*$ is not observable, unless in evaluation/testing.*

>**noise-free prediction:** $\hat{f} = \begin{bmatrix}f \\ f^* \end{bmatrix}$

>(joint) kernel matrix for training and testing**: $\hat{K} = \sigma_p^2 \hat{\Phi} \hat{\Phi}^T$

>Prior adding data genereation noise: $\hat{y}\|X,x^* \sim N(0,\hat{K}+\sigma_n^2 I)$

**Problems to be addressed:**

- Here, the distribution is not conditioned on any observed $y$. We'll do that in GP. 

- Also, the point of the kernel trick is that we don't have to compute the feature vector on the training set(which is larger than a test set or a few unobserved points). A large chunk of $\hat{K}$ is actually $K$, so we can simply reuse these values. In reality, we probably don't even need to compute feature vector at all(even for this linear kernel): 

> 1. For certain feature transformations φ, we may be able to find an easier to compute expression equivalent to $\phi(x)⊤\phi(x′)$. (2.4.2)

We'll come back to it. 

## Filtering/Kalman Filter(Ch 3)

*This chapter is relatively short. The idea of filtering is very similiar to markov chain(hidden state of world->perception->model state). I don't think I would add a lot of things to what the book said for this chapter. But I'll attach my notetaking:*

**Bayesian Filtering**

*x* is the hidden state sequence and *y* is the observation sequence. 

Start with $p(x_0)$. 

For each iteration, we have $p(x_t\|y_{1:t-1})$ and observe the new $y_t$. 

- Conditioning step(*3.8*): compute $p(x_t\|y_{1:t}) = p(x_1)p(y_1\|x_1) \Pi^t_{i=2}p(x_i\|x_{i-1})p(y_i\|x_i)$

- Prediction step(*3.9*): $p(x_{t+1}\|y_{1:t}) = \int p(x_{t+1}\| x_t)p(x_t\|y_{1:t})dx_t$

*Bayesian Smoothing* is a relevant concept about computing the hidden state in the past: it estimates $X_k\|y_{1:t} \forall k < t$ 

Also, most Bayesian filters assume *conditional independence*($X_t$ is only dependent on $X_{t-1}$, the previous state, like Markov chain).

**Kalman Filter: with a Gaussian prior**

Kalman Filter is a Bayesian Filter with a Gaussian prior(also you can see that it assumes linear relationship). It can be expressed in closed form. 

$$\begin{aligned}
&X_0 \sim N(\mu, \Sigma) \\
&X_{t+1} = FX_t+\epsilon_{t} \\
&Y_t = HX_t+ \eta_{t} 
\end{aligned}$$

*Conditioning(Kalman Update, 3.2.1)*

$$\begin{aligned}
&X_{t+1}\|Y_{1:t+1} \sim N(\mu_{t+1}. \Sigma_{t+1})\\
&\mu_{t+1} = F\mu_t+\textcolor{red}{K_{t+1}}(y_{t+1}-HF\mu_t) \\
&\Sigma_{t+1} = (I-\textcolor{red}{K_{t+1}}H)(F\Sigma_tF^T+\Sigma_x)
\end{aligned}$$

$F\mu_t$: expected state

$HF\mu_t$: expected observation

$K_{t+1}$: Kalman gain(relevance of new observation to the prediction). $K_{t+1} = (F\Sigma_tF^T+\Sigma_x)H^T(H(F\Sigma_tF^T+\Sigma_x)H^T+\Sigma_y)^{-1} \in R^{d\times m}$

*Prediction(3.2.2)*

$$\begin{aligned}
\mu_{t+1} = F\mu_t \\
\Sigma_t = F\Sigma_tF^T+\Sigma_x \\
 = E[(X_{t+1}-\hat\mu_{t+1})(X_{t+1}-\hat\mu_{t+1})^T\|y_{1:t}] \\
 = FE[(x_t-\mu_t)(x_t-\mu_t)^T\|y_{1:t}]+E[\epsilon_t\epsilon_t^T]
\end{aligned}$$


## Gaussian Process(Ch 4)

As in LR, we are still working with a training dataset and interested in estimate a function value over $x$. However, we have different assumptions(compared to it's linear or it's linear on another feature space) thus a different model: 

> The collection of random variables, (who are unknown function values, $f_x = f(x) \forall x\in X$), are called GP if *any infinite subset* of them is jointly Gaussian.

> Ininite dimensional Guassian over functions: at any location $x$, GP yields a (Gaussian) distribution over $f(x)$ (*fig 4.1*)

However, if we have an infinite-dimensional Gaussian(without any constraint). We would have infinite parameters. We have $O(n)$ means and $O(n^2)$ parameters for the covariance matrix. 

This is where we would draw relationship between GP and kernelized Bayesian LR: (1) We would have a prior(usually center around 0) such that when we don't observe any data around $x$, we would estimate a fix value. For somewhat "observed value of x", we can derive a posterior about the mean with the data.  (2) We have seen that kernel matrix is a covariance matrix(for a Gaussian distribution), and is defined by a kernel function that probably can be described by limited amount of parameters or even written in closed form. 

In fact, the predicted values(on training and unseen observation) of kernelized BLR can be seen as a GP(it is infinitely dimensional, and the distribution is gaussian for any subset).  

A GP is thus defined by a mean function $\mu(.)$ and a covariance(kernel) function $k(.,.)$. 

$$\begin{aligned}
&f\sim GP(\mu, k)\\
&f_A = [f_{x_1} ... f_{x_m}]^T \sim N(\mu_A, K_{AA})\\
&\mu_A = [\mu(x_1)...]^T; K_{AA_{ij}} = k(x_i, x_j).
\end{aligned}$$

The distribution at a new given point $x^*$ can be written(with a defined mean and covariance function), you can see the aleatoric and epistemic terms in the variance: 

$$y^*\|x^* \sim N(\mu(x^*), k(x^*, x^*)+\sigma^2_n)$$

*I am not 100% sure if this is a prior or predictive posterior. I lean to the idea that it is a prior because it is not conditioned on training data. Futhermore, the book will later obtain a posterior with $\mu'(.)$ and $k'(.,.)$.*

The previously referenced fig 4.1 depicted a predictive posterior for sure. However, the existence of the two variance terms is also depicted in the above variance term.

**Learning and Inference(*4.1*)**

The joint distribution between observations(training) and Noise-free prediction: 

$$\begin{aligned}
& \begin{bmatrix}y \\ f^* \end{bmatrix} | x^*,X  \sim N(\hat\mu, \hat K) \\
& \hat K = \begin{bmatrix} K_AA & k_{x^*, A}\\k^T_{x^*, A} & k(x^*,x^*)\end{bmatrix}; k(x,A) = \begin{bmatrix} k(x,x_1)\\... \end{bmatrix}
\end{aligned}$$

As you may recall from kernelized BLR, we just need to calculate $k(x^*, A)$ (the kernel function for new data point and our observations) and $k(x^*, x^*)$. 

From the joint distribution, we can derive the predictive posterior(the conditional distribution), which is also the predictive posterior of kernelized BLR:

$$\begin{aligned}
&f|x_{1:n},y_{1:n} \sim GP(\mu', k') \\
&\to f^*|x^*,x_{1:n},y_{1:n} \sim N(\mu'(x^*),k'(x^*,x^*))\\
&\quad \mu'(x) = \mu(x)+k^T_{x,A}\textcolor{red}{(K_{AA}+\sigma^2_nI)^{-1}}(y_A-\mu_A)\\
&\quad k'(x,x') = k(x,x') - k^T_{x,A}(K_{AA}+\sigma^2_nI)^{-1}k_{x',A}
\end{aligned}$$

Book's derivation utilized 1.53(*1.2.3* pp21), in our case, A is the distribution of testing($f$) and B is the distribution of the training($[x,y]_{1:n}$): 

$$\begin{aligned}
&X_A \sim N(\mu_A,\Sigma_A);  \\
& \to X_A|X_B = x_B \sim N(\mu_{A\|B}, \Sigma_{A\|B})\\
& \quad \mu_{A|B} = \mu_A + \Sigma_{AB}\Sigma^{-1}_{BB}(x_B-\mu_B)\\
& \quad \Sigma_{A|B} = \Sigma_{AA} - \Sigma_{AB}\Sigma^{-1}_{BB}\Sigma_{BA}
\end{aligned}$$

I'll explain briefly about the colored part: $\Sigma_{BB}$ is the prior with the data noise(it showed up before in our notes). My idea is that although the generation noise/error has been observed for the training set, it still can add to the uncertainty about the uunobserved underlying distribution.(?)

I won't provide a proof for 1.53 but provide a short explanation on top of the book:

> $Σ_{AB}Σ^{−1}_{BB}(x_B − \mu_B)$ represents “how
different” $x_B$ is from what was
expected.

$Σ_{AB}Σ^{−1}_{BB}$ can be seen as "effect of observing B to our expectation of A". Therefore, we adjust our expectation by adding an additional term to the mean, and our expectation become less uncertain.

Note that: 

- The inverse of a covariance matrix is a precision matrix: $P = \Sigma^{-1}$; scalar: $p = 1/\sigma^2$

-  $\Sigma^T_{AB} = \Sigma_{BA}$ (Covariance Matrix).

- Thus, The univariate form: 

$$N(\mu_A+\frac{\sigma_{AB}}{\sigma^2_{B}}(x_B-\mu_B), \sigma^2_{A}-\frac{\sigma^2_{AB}}{\sigma^2_{B}})$$

This should ring some bell if you have seen it (least square estimate of the slope in univarate LR is $\frac{\sigma_{AB}}{\sigma^2_{B}}$). 

If you have not, this might be reasonable because $\sigma_{AB}$ is the covariance and $\sigma^2_{B}$ is the variance of B, and for a change in B to have a large impact on our expectation of A: (1) the covariance needs to be large, (2) the variance of B should be small. 

<!-- ALSO, if linearly dependent, cannot take inverse. (iid data) -->

**Kernel Functions(4.4)**

Before we start, here's a great [GP visualization website](https://www.infinitecuriosity.org/vizgp/) that should help with understanding how observed data or choice of kernel/parameters might affect the smoothness of GP. 

- *Linear*

$k(x,x'; \phi) = \phi(x)^T \phi(x')$

You might be able to define a kernel with a transformation $\phi$ up to some power. However, we will go beyond this since we don't have to think about feature space with the following kernels. 

- *Gaussian(RBF)*

$k(x,x'; h) = exp(-\frac{\|\|x-x'\|\|^2_2}{2h^2})$

$h$ would control the smoothness(larger -> smoother). Gaussian kernel is equivalent to infinitely dimensional feature space. It's an exercise in the book, however we'll not prove it here. 

- *Laplace(Exponential)*

$k(x,x'; h) = exp(-\frac{\|\|x-x'\|\|_2}{h})$

Laplace kernel are non-smooth. 

- *Matern: Gaussian and Laplace with a tradeoff parameter($v$)*

Book discussed how to combine/map existing kernels to create new kernels(*4.3.2*) and also the two property of the kernels(*4.3.3*): 

- stationary: kernel depends on relative locations of pts. $k(x,x') = \hat(k)(x - x')$

- isotropic: kernel depends only on distance of pts. $k(x,x') = \hat(k)(\|\|x - x'\|\|)$

*RBF is isotropic(and thus also stationary), EXP is only stationary. Linear kernel is neither.* 

We may go back to the benefits of kernel tricks(*2.4.2*), which we mentioned in Non-linear LR. 

```
1. For certain feature transformations ϕ, we may be able to find an easier to compute expression equivalent to ϕ(x)⊤ϕ(x′).

2. If this is not possible, we could approximate the inner product by an easier to compute expression.

3. Or, alternatively, we may decide not to care very much about the exact feature transformation and simply experiment with kernels that induce some feature space (which may even be infinitely dimensional).
```

2 and 3 are RBF and Laplace kernel. Both don't have an explicit form of feature vector or inner product during computation, yet can be proven to be infinite-dimensional. 

1 is a trick that we can use for linear(polynomial) kernel. 

> For a feature vector up degree m, and for a constant factor: $\phi(x)^T \phi(x') = (1+x^Tx')^m$(Fact 2.4, *2.4.2*)

This reduces the computational cost from $O(d^{power})$ to $O(d)$. 

**Sampling(4.2)**

Instead of full posterior, we might obtain samples in these two ways(both $O(n^3)$): 

- Affine trasnformation: $f = K^{1/2}\epsilon+ \mu$ (Square Root)

- Forward Sampling: $f_n \sim p(f_n\|f_{1,...,n-1})$ (Matrix Inverse in previous posterior)

<!-- Here, need to dig more -->

**Reproducing Kenel Hilbert Space(RKHS)**

*RKHS for a kernel $k:X\times X\to R$:* $H_k(X) = \{f(.) = \Sigma_i \alpha_i k(x_i, .)\}$

$$\forall x\in X \to k(x,.) \in H_k(X)$$

*Inner product of two function within the same RKHS(f,g are two RKHS):* $<f,g>_{k} = \Sigma_i \Sigma_j \alpha_i \alpha'_j k(x_i, x'_j)$; 

*Norm of a function measures the smoothness/complexity of f:* $\|\|f\|\|_k = \sqrt{<f,f>_k}$

*Reproducing Property*

$$\forall x\in X, f\in H_k(X), \to f(x) = <f(.),k(x,.)>_k$$

**Representer Theorem**

$$\begin{aligned}
&\hat f = argmin_{f\in H_k(x)} L(f(x_1)...f(x_n)) + \lambda ||f|| \\ 
&\to \hat f(x) = \hat\alpha^T k_{x,x_i} = \Sigma_i \hat\alpha_i k(x,x_i)
\end{aligned}$$

Representer Theorem are used across methods that use kernel trick. It shows that we reduced the MAP(point estimate) to a regularized LR. Thus, GP is tractable, even with infinite dimensional feature space. 

<!-- This is powerful enough before NN. -->

*We skip the proof for now. The book then goes on to train-validation split, and regularization term, etc. These are theoretical(sort of) justifications to common empirical techiniques and we would skip these parts.*

-------

*We would skip the Bayesian NN part in the book(for now).*