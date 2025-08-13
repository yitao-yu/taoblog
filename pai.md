---
layout: page
title: "Study Notes: Probabilistic AI"
permalink: /pai/
group: pai
---

## Motivation and Thoughts

Hi! I'm Yitao. This is a list of my notes while reading the book *"Probabilistic Aritificial Intelligence"* by Andreas Krause and Jonas Hübotter, which I'll refer to as PAI. It is [available on arxiv](https://arxiv.org/abs/2502.05244). I believe that there is also a [course website](https://las.inf.ethz.ch/teaching/pai-f24). 

<!-- There seems to be [repo](https://github.com/angelognazzo/Probabilistic-Artificial-Intelligence) of previous semesters' homework of that course! -->

There are many approaches to learn reinforcement learning, and I believe that the book is friendly to people with background in machine learning. It is also a good chance to brush up memories from books like PRML. 

Ultimately, our goal from reading the book is to know about concepts like Gaussian Process, Multi-arm Bandit and Q-learning, and how methods like markov chain and Neural Networks are used in solving the RL problem. 

Another course that might be of interest is [*Reinforcement Learning and Optimal Control*](https://www.mit.edu/~dimitrib/RLbook.html) by Dimitri P. Bertsekas, which would provide more details about different problems in Reinforcement Learning. This course might also be more friendly to people with background in control theory. 

My notetaking is mostly a cheatsheet for future me and I'll go through some of the proofs in the book that is not entirely clear to me(and some of them are left in the book as exercise). A major part of the note is generated from back and forth conversation with LLMs while reading the book. Hopefully, my note can be helpful to you. 

## My notes

Plan: We would skip Bayesian Neural Network for now and come back later, and at the end, we might rely on few external sources to learn about topics like RLHF. 

- 01: [Warmup(Bayes Linear Regression, Kernel Trick, Gaussian Process)]({{site.url}}{{site.baseurl}}/pai/warmup-1); Chapter 1.4, 2,3,4 of PAI, Chapter 7 to be added in future
- 02: [Active Learning]({{site.url}}{{site.baseurl}}/pai/active-learning-2); Chapter 8 of PAI
- 03: [Bayesian Optimization(Bandits, etc.), Markov Decision Process(Policy Gradient, etc.)]({{site.url}}{{site.baseurl}}/pai/bayes-opt);Chapter 9, 10 of PAI
- 04:  [Tabular RL]({{site.url}}{{site.baseurl}}/pai/tabular-rl);Chapter 11 of PAI
- 05: [Model-free RL(large action space)]({{site.url}}{{site.baseurl}}/pai/model-free);Chapter 12 of PAI
- 06: [Model-based RL(large action space)]({{site.url}}{{site.baseurl}}/pai/model-based);Chapter 13 of PAI
- Extension: To Be Decided