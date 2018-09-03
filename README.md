# Hidden Markov Model

For a walkthrough, see the Jupyter notebook file. The Python code in the files is slightly different and the program ha a different interface, but it's all the same mathematics underneath.

This is a two-state HMM plus a small English corpus. The goal of the project was to determine which initializations of the model would result in the clustering of consonant and vowel segments. It also tries to print a heat map after sampling the distribution.

All of the source code is in hmm.py. It's written in Python 2.6 and is exectuable. Included packages are math, copy, and random, which (to my knowledge) come with the default distribution. Simply run it and follow the prompts. The script will initialize A, B, and Pi as random distributions, and then will allow the user a few options:

- To produce the most likely underlying state sequence corresponding to a user-input word (Viterbi algorithm);
- To enter a specific word to test the probability computation of the forward and backward algorithms;
- To read the corpus in english1000.txt once, compute softcounts and iterate until the change in likelihood is less that 0.001, printing the results if flagged (Baum-Welch algorithm);
- To sample the space of possible a distributions, performing the above task with each possible distribution within a 7x7 grid centered inside the unit square with spacing 1/8 and margin 1/16, then printing the distribution and cell yielding the lowest plog sum and printing a full report to sample.txt. It then uses matplotlib to display a heatmap of likelihoods with respect to random initializations of A. This is still in development as I teach myself matplotlib.
