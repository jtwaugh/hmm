README
--

Hidden Markov Model project
James Waugh
CMSC 25020

--

This project contains three files and writes a fourth:
	* README.txt
	* hmm.py
	* english1000.txt
	* sample.txt

All of the source code is in hmm.py. It's written in Python 2.6 and is exectuable. Included packages are math, copy, and random, which (to my knowledge) come with the default distribution. Simply run it and follow the prompts. The script will initialize A, B, and Pi as random distributions, and then will allow the user a few options:
 	* To enter a specific word to test the probability computation of the forward and backward algorithms;
	* To read the corpus in english1000.txt once, compute softcounts and iterate until the change in plog sum is less that 0.001, printin the results if flagged;
	* To sample the space of possible a distributions, performing the above task with each possible distribution within a 7x7 grid centered inside the unit square with spacing 1/8 and margin 1/16, then printing the distribution and cell yielding the lowest plog sum and printing a full report to sample.txt.

The program prints smoothed log ratios to avoid division by zero.