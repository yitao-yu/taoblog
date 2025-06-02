---
layout: page
title: "PAI 1: Warmup(Bayes Linear Reg, Kernel Trick, GP)"
permalink: /pai/warmup-1
group: pai
---

## Decision Theory(*1.4*)

Choosing the optimal action is about choosing the action that would maximize the expected reward. *We are interested in learning the reward function in later parts.*

**Optimal Decision Rule**: $a^*(x) = {argmax}_a E_{y\|x} [r(y,a)]$. 

Equivalently, to see $-r$ as loss, we can also also acquire the optimal decision by minimizing the loss and it's a regression problem if we are making a decision in a continuous space. 

The book discussed mostly how square loss(symetrical reward) and asymmetrical loss would affect the optimal decision. 