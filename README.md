# Privacy Audit: Membership Inference Attacks on Generative Models

## Abstract

This project demonstrates privacy vulnerabilities in generative models by performing 
a membership inference attack on a GAN trained on MNIST. We show that an attacker 
can exploit the discriminator's memorization of training data to distinguish members 
from non-members with above-random accuracy. We then apply Differential Privacy via 
DP-SGD as a defense and demonstrate that privacy leakage is reduced to essentially 
zero, with attack accuracy dropping below random guessing.

---

## 1. Introduction

### The Problem

Modern generative models are trained on sensitive data. Medical records, location 
traces, financial transactions, personal images — all are potential training inputs 
for AI systems. But training creates a fundamental privacy risk: models memorize 
their training data.

This memorization enables **membership inference attacks** — where an adversary 
queries a model to determine whether a specific individual's data was used in 
training. Even without access to the training data itself, an attacker can exploit 
the statistical gap between how a model responds to data it has seen versus data 
it hasn't.

This problem is not theoretical. In 2021, researchers demonstrated that GPT-2 
could be prompted to reproduce verbatim training data including personal information. 
The same fundamental vulnerability exists in all generative models — GANs, VAEs, 
and large language models alike.

### Research Questions

1. Can membership inference attacks successfully exploit privacy leakage in a GAN?
2. Does Differential Privacy eliminate this leakage, and at what cost?
3. How does the privacy-utility tradeoff manifest in practice?

---