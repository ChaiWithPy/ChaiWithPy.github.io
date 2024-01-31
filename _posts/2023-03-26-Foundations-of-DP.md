---
layout: post
title: "Foundations of Differential Privacy"
date: 2023-03-26T00:00:00-00:00
author: Aman Priyanshu
categories: technical-blogs
---

<!-- <b>View Counter:</b><a href="{{ page.url | prepend: 'https://hits.sh/chaiwithpy.github.io' }}"><img alt="Hits" src="{{ page.url | prepend: 'https://hits.sh/chaiwithpy.github.io' | append: '.svg?label=Views'}}"/></a> -->

## Introduction

### A Technically, Non-Technical definition of Privacy

Imagine you have a special box where you keep all your secrets, like your favorite collectibles and even your money. You don't want anyone to know what's inside because it's private and personal to you.

Now, let's say there's a new series of collectibles, and you really want one, but you need to give your special box to a stranger for them to decide the most suitable collectible for you. However, you might feel unsure about giving away your box because you don't know if the stranger will look inside and find out your money/wallet.

This is similar to how privacy works in the digital world. Just like how you keep your secrets in a special box, people keep their personal information, like their name, address, and favorite websites, private and secure. However, when we use certain apps or services, such as recommendation systems, they may need access to some of our data for improved customer satisfaction. However, this may lead to the leakage of sensitive information, which may not have been necessary.

AI is like a brilliant computer that can learn and make decisions on its own. It can use personal information to recommend movies or products we might like. However, if we're not careful, the AI might use our data in ways we don't want it to, like sharing it with strangers or targeting us with ads.

Continuing on, we can delve deeper into more formal and technical descriptions of privacy and privacy-preserving technology.

### What is Privacy?

Privacy is the ability to keep information about ourselves secure or have authority/control over it. Everything from who can access your data to who can utilize it is encompassed within the Privacy. It is essential because it enables us to maintain control over our personal information and avoid unwanted intrusions into our private lives. This can range from your favorite movies, books, and comics to your location, banking details, internet history, and even the pictures you take from your mobile devices.

### What is Privacy-preserving Technology?

Now, privacy-preserving technology is a set of tools and techniques that enable us to protect our personal information while still being able to use digital services. These technologies use cryptography, differential privacy, federated learning, and other techniques to enable secure data sharing and processing while maintaining the privacy of individuals' data. In this article, we refer specifically to PPML paradigms within privacy-preserving technology.

### Examples where Privacy is necessary

1. **Recommendation Systems (Internet History):** Your personal data, history, and click rates can potentially describe everything from your *next favorite movie to your political polarity*. Therefore, these fields must have robust privacy conservation modules integrated into them.

2. **Epidemic Detection:** While having access to medical data might be incredibly useful for applications such as Epidemic Detection, it comes at the cost of leaking your personal medical records. These documents can be used by third parties like *insurance companies, adversaries, and competing businesses.*

### Understanding the need for Modern Privacy

As we describe the motivations behind privacy, we must also understand that analytics through AI is still necessary. Therefore, developing tools to augment privacy into these methodologies is required.

So a straightforward implementation of this privacy protection can simply be removing or hiding sensitive attributes while retaining the rest of the data and all its precise analytics.

Introducing **Anonymization:** While original data may not be suitable for applications, sensitive information from these datasets can be anonymized. For example, removing peoples' names or bucketing groups of ages together `(0-5, 6-10, 11-17, 18-25,... etc.)`.

However, these methodologies may not secure the sensitive data entirely and may be unsafe due to plausible attacks, such as:

1. **Deanonymization Attacks:** Using analytics, suggestions, logical conjectures, relational datasets, or other sources to deanonymize the anonymization algorithm.
2. **Reconstruction Attacks:** It is a stronger and more developed expansion of deanonymization attacks.
3. **Differencing Attacks:** Employing statistics/analytical techniques to extract sensitive attributes about a target by still asking aggregate questions.

So, understanding that methods which augment privacy on data post-collection may not be suitable. At the same time, delivering precise analytics, which may allow differencing attacks, may also not be ideal for robust security. Therefore, a need for a mathematically rigorous method is necessary.

## Discovering Reconstruction Attacks

Now, one of the primary propositions discussed above is aggregate statistics as a solution to privacy attacks. However, without the consideration of its precise analytics/statistics extraction framework, this attribute could instead be employed to extract sensitive data.

In essence,
> Given a dataset *X*, analytical functions *F = f_1, f_2, ..., f_k*, and their responses *A = a_1, a_2, ..., a_k*; we can extract *X'* which is a subset of all possible combinations of *X* such that they follow all the constraints of *F* and *A*.

There's also two popular implementations of reconstruction attacks:
- Reconstruction Attack
![Reconstruction Attack](https://chaiwithpy.github.io/assets/recon.png)
- Linear-Programming based Reconstruction Attack
![LP-Reconstruction Attack](https://chaiwithpy.github.io/assets/lp_recon.png)

Therefore, a methodology that returns accurate/precise aggregate statistics is still insufficient for privacy conservation. Therefore, a process that can, by engineering, introduce noisy responses is more valuable and robust against reconstruction attacks.

## Differential Privacy

> An analysis is private if the data analyst knows almost no more about a target after the analysis than if the analyst would have known had he conducted the same analysis on an identical database with the target's data replaced.

This property states that upon changing a single record does not alter the output distribution of the original dataset.

**ε-Differentially Private Dataset** mathematically denotes the expanses/difference in probability of an event happening based on the existence of a random row in dataset D and its replacement in D'.

### Properties:

1. **A Privacy Loss Measure (ε):** Provides strong mathematical bounds describing privacy.
2. **Resilience to Post-Processing Attacks:** 
* Resistance to re-identification attacks.
* Due to sampling replacement, background knowledge-based inference becomes irrelevant.
3. **Easy Programmability:** construction of simple privacy-preserving building blocks can be used to build complicated analytical systems.

## Conclusion

Privacy is essential for individuals because it enables us to maintain control over our personal information and avoid unwanted intrusions into our lives. Privacy-preserving technology, like differential privacy, enables us to protect our personal information while still being able to use digital services.