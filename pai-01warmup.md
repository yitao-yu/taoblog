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

$$c_2\Phi(\alpha) - c_1*(1-\Phi(\alpha)) = 0\\
\to\Phi(\alpha) = c_1/(c_1+c_2)\\
\to a^* = \mu + \sigma*\Phi^{-1}[\frac{c_1}{c_1+c_2}]$$

#### Leibniz Rule

Case 1(constant bounds) is intuitive: 

$u'(x) = \int_a^b f_x(x,t) dt$ where $u(x) = \int_a^bf(x,t)dt$

$u'(x) = lim_{h\to0} \frac{u(x+h) - u(h)}{h}\\ = \int_a^b lim_{h\to0}\frac{f(x+h,t)-f(h,t)}{h}dt\\ = \int_a^b \frac{df(x,t)}{dx} dt$

Case 2(function bounds) can be derived with the same method: 

$$u'=\frac{d}{d\alpha}\int_{a(\alpha)}^{b(\alpha)}f(z,\alpha)dz = lim_{h\to0}\frac{1}{h}[\int_{a(\alpha+h)}^{b(\alpha+h)} f(z,\alpha+h) dz- \int_{a(\alpha)}^{b(\alpha)} f(z,\alpha) dz]$$

By adding and subtracting $\int_{a(\alpha)}^{b(\alpha)}f(z,\alpha+h)dz$: 

$$u'= lim_{h\to0}\frac{1}{h}[\int_{a(\alpha+h)}^{b(\alpha+h)} f(z,\alpha+h) dz - \int_{a(\alpha)}^{b(\alpha)}f(z,\alpha+h)dz \\+ \int_{a(\alpha)}^{b(\alpha)}f(z,\alpha+h)dz - \int_{a(\alpha)}^{b(\alpha)} f(z,\alpha) dz] \\= f(b(\alpha,\alpha)) \frac{db}{d\alpha} −f(a(α),α) \frac{da}{d\alpha} + \int_{a(\alpha)}^{b(\alpha)}\frac{\delta f(z,\alpha)} {\delta \alpha} dz$$


## Linear Regression

## Kernel Trick

## Gaussian Process
