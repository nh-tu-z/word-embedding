# Word Embedding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17d1Dj6kW_C0aYlDtoUjexto7c87ykENI?usp=sharing#scrollTo=9TnVbaH9hOgo)

## Instructor
PhD. Nguyễn An Khương

## Group 13
- Ngô Hoài Tú - 2570536
- Nguyễn Hoàng Kiên - 2570435
- Trần Quốc Việt - 2570154
- Huỳnh Đức Nhâm - 2570276

## About

This repository hosts a research-focused collection of notebooks and supporting code for exploring classical word-embedding methods in natural language processing, following the pedagogical treatment in the Dive into Deep Learning (d2l) chapter on NLP pretraining (https://d2l.ai/chapter_natural-language-processing-pretraining/index.html).

The experiments reproduce and compare two foundational shallow embedding models:
- Skip-gram (SG): models the probability of context words given a center word, optimized so nearby words share similar representations.
- Continuous Bag-of-Words (CBOW): predicts a center word from its surrounding context (bag-of-words), producing dense word vectors efficient for many downstream tasks.

Because exact softmax across large vocabularies is costly, the notebooks implement and evaluate two common approximation strategies used in practice:
- Negative sampling: a sampled objective that trains the model to distinguish target-context pairs from randomly sampled noise pairs, greatly speeding up training while producing high-quality vectors.
- Hierarchical softmax: uses a binary tree decomposition of the vocabulary to compute approximate softmax probabilities in O(log V) time per update.

Research goals
- Reproduce key behaviors and trade-offs described in d2l for SG vs. CBOW and for negative sampling vs. hierarchical softmax.
- Measure embedding quality on intrinsic tasks (word similarity, analogy) and visualize learned spaces (t-SNE / UMAP).
- Provide reproducible, notebook-driven experiments that make hyperparameters, seeds, and training procedures explicit so results can be replicated and extended.

What you’ll find in the notebooks
- Data preprocessing and vocabulary construction suitable for shallow embedding training.
- Implementations (and gensim-based baselines) of Skip-gram and CBOW.
- Training recipes for negative sampling and hierarchical softmax, plus comparisons of convergence and wall-clock performance.
- Evaluation scripts and plotting utilities to inspect nearest neighbors, analogies, and embedding geometry.
- Notes and guidance on hyperparameter choices (embedding dimension, window size, negatives, learning rate) and reproducibility.

Reference
- Dive into Deep Learning (d2l) — Natural Language Processing: Pretraining: https://d2l.ai/chapter_natural-language-processing-pretraining/index.html
