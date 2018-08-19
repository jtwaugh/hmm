import random
from math import log, fabs
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt

# Flag variables

# Word-level printing
verbose_flag = False
# Corpus-level printing
print_flag = False


# Functions to initialize distributions

def distribute_default(distribution, from_states, to_states):
	magnitude = len(from_states) * len(to_states)
	for from_state in from_states:
		for to_state in to_states:
			distribution[(from_state, to_state)] = 1.0 / float(magnitude)


def distribute_random(distribution, from_states, to_states):
	for from_state in from_states:
		magnitude = 0.0
		for to_state in to_states:
			distribution[(from_state, to_state)] = random.random()
			magnitude += distribution[(from_state, to_state)]
		for to_state in to_states:
			distribution[(from_state, to_state)] /= magnitude
		

def distribute_pi(pi, from_states):
	magnitude = 0.0
	for from_state in from_states:
		pi[from_state] = random.random()
		magnitude += pi[from_state]
	for from_state in from_states:
		pi[from_state] /= magnitude


# Model values

class HMM():

	def __init__(self, _hidden_states, filename):
		self.A = dict()
		self.B = dict()
		self.Pi = dict()
		self.softcount = dict()
		self.hidden_states = _hidden_states
		self.corpus_name = filename
		self.word_count = 1
		self.longest = 0

		# Determine the output states from the corpus
		chars = set(['#'])
		corpus_file = open(self.corpus_name, "r")
		line = corpus_file.readline().strip('\n').lower()
		while line:
			chars |= set(line)
			self.word_count += 1
			self.longest = max(len(line), self.longest)
			line = corpus_file.readline().strip('\n').lower()
		self.output_states = sorted(list(chars))

		distribute_random(self.A, self.hidden_states, self.hidden_states)
		distribute_random(self.B, self.hidden_states, self.output_states)
		distribute_pi(self.Pi, self.hidden_states)

		if verbose_flag:
			print "\n\n--\n\n========================"
			print "==   Initialization   =="
			print "========================"

		for from_state in self.hidden_states:
			if verbose_flag:
				print "\n--\n\nCreating state " + from_state.__str__() + "\n\n"
				print "Transitions:\n"
			transition_sum = 0.0
			for to_state in self.hidden_states:
				if verbose_flag:
					print "    To state     " + to_state.__str__() + ":    " + self.A[(from_state, to_state)].__str__()
				transition_sum += self.A[(from_state, to_state)]
			if verbose_flag:
				print "\nTotal: " + transition_sum.__str__()
				print "\n\nEmissions:\n"
			emission_sum = 0.0
			for to_state in self.output_states:
				if verbose_flag:
					print "    To letter    " + to_state.__str__() + ":    " + self.B[(from_state, to_state)].__str__()
				emission_sum += self.B[(from_state, to_state)]
			if verbose_flag:
				print "\nTotal: " + emission_sum.__str__()

		if verbose_flag:
			print "\n--\n"
			print " Starting distribution:\n"
		start_sum = 0.0
		for state in self.hidden_states:
			if verbose_flag:
				print "    For state    " + state.__str__() + ":    " + self.Pi[state].__str__()
			start_sum += self.Pi[state]
		if verbose_flag:
			print "\nTotal: " + start_sum.__str__()


	# Compute probabilities for an individual word

	def forward(self, word):
		# Initialize the distribution
		alpha = dict()

		# Print if flagged
		if verbose_flag:
			print "\n--\n\nComputing forward probabilities."

		# Initilize \alpha values to the initial distribution
		for state in self.hidden_states:
			alpha[(state, 0)] = self.Pi[state]

		# Moving forward, compute new alpha values from probability products
		for t in range(1, len(word) + 1):

			# Print if flagged
			if verbose_flag:
				print "\n\n    Time " + t.__str__() + ": \'" + word[t-1] + "\'"

			# Keep a running sum at each time t
			t_sum = 0.0
			# Run through posssible next states
			for to_state in self.hidden_states:
				alpha[(to_state, t)] = 0

				# Print if flagged
				if verbose_flag:
					print "        To state " + to_state.__str__()

				# Find the forward probability given the next letter
				for from_state in self.hidden_states:
					increment = alpha[(from_state, t-1)] * self.B[(from_state, word[t-1])] * self.A[(from_state, to_state)]
					alpha[(to_state, t)] += increment

					# Print if flagged
					if verbose_flag:
						print "            From state " + from_state.__str__() + ": \\alpha_{" + from_state.__str__() + ", " + (t-1).__str__() + "} \cdot b_{" + from_state.__str__() + ", " +  word[t-1] + "} \cdot a_{" + from_state.__str__() + ", " + to_state.__str__() + "} = " + increment.__str__() 
				
				# Print if flagged
				if verbose_flag:
					print "        \\alpha_{" + to_state.__str__() + ", " + t.__str__() + "} = " +  alpha[(to_state, t)].__str__()
				
				# Add the probability from the current state to the sum for t
				t_sum += alpha[(to_state, t)]
			
			# Print if flagged
			if verbose_flag:
				print "\n    \sum_{x \in X} \\alpha_{x, " + t.__str__() + "} = " + t_sum.__str__()

		# Print if flagged
		if verbose_flag:
			print "\n--\n\n"		
			for t in range(0, len(word)+1):
				print "Time " + t.__str__() + ":"
				for state in self.hidden_states:
					print "    \\alpha_{" + state.__str__() + ", " + t.__str__() + "} = " + alpha[(state, t)].__str__()

		return alpha


	def backward(self, word):
		# Initialize the distribution
		beta = dict()

		# Print if flagged
		if verbose_flag:
			print "\n--\n\nComputing backward probabilities."

		for s in self.hidden_states:
			beta[(s, len(word))] = 1

		for t in range(len(word), 0, -1):
			
			# Print if flagged
			if verbose_flag:
				print "\n\n    Time " + t.__str__() + ": \'" + word[t-1] + "\'"
			
			# Keep a running sum at each time t
			t_sum = 0.0
			
			for from_state in self.hidden_states:
				# Initialize \beta
				beta[(from_state, t-1)] = 0.0

				# Print if flagged
				if verbose_flag:
					print "        From state " + from_state.__str__()

				# Find the backward probability given the last letter
				for to_state in self.hidden_states:
					increment = beta[(to_state, t)] * self.B[(from_state, word[t-1])] * self.A[(from_state, to_state)]
					beta[(from_state, t-1)] += increment

					# Print if flagged
					if verbose_flag:
						print "            To state " + to_state.__str__() + ": \\beta_{" + to_state.__str__() + ", " + (t+1).__str__() + "} \cdot b_{" + from_state.__str__() + ", " +  word[t-1] + "} \cdot a_{" + from_state.__str__() + ", " + to_state.__str__() + "} = " + increment.__str__() 
				
				# Add the probability from the current state to the sum for t
				t_sum += beta[(from_state, t-1)]

				# Print if flagged
				if verbose_flag:
					print "\n    \sum_{x \in X} \\beta_{x, " + t.__str__() + "} = " + t_sum.__str__()
			
		

		# Print if flagged
		if verbose_flag:
			print "\n--\n\n"
			for t in range(0, len(word)+1):
				print "Time " + t.__str__() + ":"
				for state in self.hidden_states:
					print "    \\beta_{" + state.__str__() + ", " + t.__str__() + "} = " + beta[(state, t)].__str__()
		return beta


	def forward_probability(self, alpha, length):
		alpha_sum = 0.0
		for state in self.hidden_states:
			alpha_sum += alpha[(state, length)]
		return alpha_sum


	def backward_probability(self, beta):
		beta_sum = 0.0
		for state in self.hidden_states:
			beta_sum += self.Pi[state] * beta[(state, 0)]
		return beta_sum


	# E-M functions

	def expectation(self):
		self.softcount = dict()
		# Set initial values
		plog_sum = 0.0
		# Open and read file
		corpus_file = open(self.corpus_name, "r")
		line = corpus_file.readline().strip('\n').lower()
		if print_flag:
			print "\n\nPlogs:\n"
		while line:
			# Append endline character
			line += "#"
			
			# Compute probabilities
			alpha = self.forward(line)
			beta = self.backward(line)
			f_prob = self.forward_probability(alpha, len(line))
			b_prob = self.backward_probability(beta)
			
			# Run through the word and tabulate softcounts
			for t in range(len(line)):
				for from_state in self.hidden_states:
					for to_state in self.hidden_states:
						if (t, line[t], from_state, to_state) not in self.softcount:
							self.softcount[(t, line[t], from_state, to_state)] = 0.0	
						self.softcount[(t, line[t], from_state, to_state)] += (alpha[(from_state, t)] * self.A[(from_state, to_state)] * self.B[(from_state, line[t])] * beta[(to_state, t+1)]) / f_prob

			# If we have agreement on the probailities, more or less
			if (fabs(f_prob - b_prob) < 0.00001):
				plog = -1 * log(f_prob, 2)
				plog_sum += plog

				# Print if flagged
				if print_flag:
					print "plog(\"" + line + "\") = " + plog.__str__()

			else:
				print "Unacceptable probability mismatch at word " + line + ": forward (" + f_prob.__str__() + ") != backward (" + b_prob.__str__() + ")."	
			line = corpus_file.readline().strip('\n').lower()
		
		# Print if flagged
		if print_flag:
			print "\n--\n\nSum of positive logs: " + plog_sum.__str__() + "\n\n--\nSoftcounts:\n"

			for t in range(self.longest):
				print "\n    At time " + t.__str__() + ": "
				mysum = 0.0
				for from_state in self.hidden_states:
					print "\n        From state " + from_state.__str__() + ": "
					for to_state in self.hidden_states:
						print "\n            To state " + to_state.__str__() + ": "
						for char in self.output_states:
							if (t, char, from_state, to_state) in self.softcount:
								print "                Emitting " + char + ": " + self.softcount[(t, char, from_state, to_state)].__str__()
								mysum += self.softcount[(t, char, from_state, to_state)]
				print "Sum = " + mysum.__str__()
		
		# Return
		return plog_sum

	def maximization(self):
		# Reset Pi

		# Print if flagged
		if print_flag:
			print "Distribution \\Pi:"
		
		for from_state in self.hidden_states:
			# Print if flagged
			if print_flag:
				print "    For state " + from_state.__str__() + " was    " + self.Pi[from_state].__str__()
				print "Recomputing... \n"

			softcount_i = sum([self.softcount[(0, char, from_state, to_state)] for char in self.output_states for to_state in self.hidden_states if (0, char, from_state, to_state) in self.softcount])

			self.Pi[from_state] = 1/float(self.word_count) * softcount_i 
		
			# Print if flagged
			if print_flag:
				print "    For state " + from_state.__str__() + " is now " + self.Pi[from_state].__str__() + "\n"


		# For each (i, j), assign A_{i, j}
		
		# Print if flagged
		if print_flag:
			print "\nDistribution A:"
		for from_state in self.hidden_states:
			
			# Print if flagged
			if print_flag:
				print "\n    From state " + from_state.__str__() + ":\n"
			
			a_denom = sum([self.softcount[(t, l, from_state, k)] for t in range(self.longest) for l in self.output_states for k in self.hidden_states if (t, l, from_state, k) in self.softcount])

			if print_flag:
				print "    Computed the denominator at state i = " + from_state.__str__() + " (sum over hidden_states, output_states, and t): " + a_denom.__str__() + "\n"

			for to_state in self.hidden_states:
				a_num = sum([self.softcount[(t, l, from_state, to_state)] for t in range(self.longest) for l in self.output_states if (t, l, from_state, to_state) in self.softcount])

				# Print if flagged
				if print_flag:
					print "\n        Computed the numerator at states i = " + from_state.__str__() + "; j = " + to_state.__str__() + " (sum over output_states and t): " + a_num.__str__() + "\n"
					print "        To state " + to_state.__str__() + " was    " + self.A[(from_state, to_state)].__str__()

				self.A[(from_state, to_state)] = a_num / a_denom
				
				# Print if flagged
				if print_flag:
					print "        To state " + to_state.__str__() + " is now " + self.A[(from_state, to_state)].__str__()


		# For each (i, l), assign B_{i, l}
		
		# Print if flagged
		if print_flag:
			print "\nDistribution B:"
		for from_state in self.hidden_states:
			# Print if flagged
			if print_flag:
				print "\n    From state " + from_state.__str__() + ": "

			b_denom = sum([self.softcount[(t, m, from_state, j)] for t in range(self.longest) for m in self.output_states for j in self.hidden_states if (t, m, from_state, j) in self.softcount])

			# Print if flagged
			if print_flag:
				print "    Computed the denominator at state i = " + from_state.__str__() + " (sum over hidden_states, output_states, and t): " + b_denom.__str__() + "\n"

			for char in self.output_states:
				b_num = sum([self.softcount[(t, char, from_state, j)] for t in range(self.longest) for j in self.hidden_states if (t, char, from_state, j) in self.softcount])

				# Print if flagged
				if print_flag:
					print "\n        Computed the numerator at states i = " + from_state.__str__() + "; \\ell = " + char + " (sum over output_states and t): " + b_num.__str__() + "\n"
					print "        To state " + char + " was    " + self.B[(from_state, char)].__str__()

				self.B[(from_state, char)] = b_num / b_denom

				# Print if flagged
				if print_flag:
					print "        To state " + char + " is now " + self.B[(from_state, char)].__str__()

		# Print if flagged
		if print_flag:
			print "\n\n--\n"

	def print_softcount():
		print "\n--\n\nSoftcounts:\n"
		for t in range(longest):
			print "\n--\n\nAt time " + t.__str__() + ":\n"
			for char in output_states:
				char_sum = 0.0
				print "\nCharacter \'" + char + "\':"
				for from_state in hidden_states:
					for to_state in hidden_states:
						if (t, char, from_state, to_state) in softcount or (t == 0):
							print "    From state " + from_state.__str__() + " to state " + to_state.__str__() + ":    " + softcount[(t, char, from_state, to_state)].__str__()
							char_sum += softcount[(t, char, from_state, to_state)]
				print "Total (equal to number of words O with O_" + t.__str__() + " = " + char +"): " + char_sum.__str__()	


	def viterbi_parse(self, word):
		path = [None for k in range(len(word))]
		
		# Keep track of the best guesses
		max_probability = dict()
		argmax_state = dict()

		# Keep track of the initial states
		for state in self.hidden_states:
			max_probability[(state, 0)] = self.Pi[state] * self.B[(state, word[0])]
			argmax_state[(state, 0)] = state

		# Moving forward, memoize the probability-maximizing next state given each possible underlying state and the inferred emission probability
		for i in range(1, len(word)):
			for state in self.hidden_states:
				func = lambda k : max_probability[(k, i-1)] * self.A[(k, state)] * self.B[(state, word[i])]
				max_probability[(state, i)] = max(map(func, self.hidden_states))
				argmax_state[(state, i)] = max(self.hidden_states, key=func)
		
		# Connect the path
		path[len(word) - 1] = max(self.hidden_states, key=(lambda k : max_probability[(k, len(word) - 1)]))
		for i in (range(1, len(word))[::-1]):
			path[i-1] = argmax_state[(path[i], i)]
		print "Viterbi parse: " + path.__str__()

# Test the probability of a single word

def test_probability(hmm):
		# Get a word and sanity check
		valid_word = False
		while not valid_word:
			valid_word = True
			word = raw_input("Enter word to compute probability:    ")

			if word[-1] != "#":
				print "Please send your word with a word boundary (\'#\')"
				valid_word = False
			for char in word:
				if not char in set(hmm.output_states):
					print "Character " + char + " is not in the corpus!"
					valid_word = False

		# Compute its probability
		alpha = hmm.forward(word)
		beta = hmm.backward(word)

		alpha_sum = hmm.forward_probability(alpha, len(word))
		beta_sum = hmm.backward_probability(beta)

		print "\n--\n\nProbability from forward computation: " + alpha_sum.__str__() + "\nplog from forward computation: " + (-1 * log(alpha_sum)).__str__() + "\n\nProbability from backward computation: " + beta_sum.__str__() + "\nplog from backward computation: " + (-1 * log(beta_sum)).__str__()


# Main routine

print "========================"
print "==   HMM:  Part III   =="
print "========================\n\n"

print "This program will randomly assign transition and emission probabilities to each of two underlying states and to output states determined from unique characters in a corpus. After assigning random distributions, it will compute the conditional probability distributions \\alpha and \\beta and use these to determine the probabilities of each word in the corpus. These will be printed as plogs.\n\nThis process will iterate, and, if indicted, the computed probabilities will be printed at every iteration until the change in the total plog sum of the corpus changes by less that \\Delta = 0.001.\n"

verbose = raw_input("Type \'W\' for word-level verbosity:    ")
if verbose == "W":
	verbose_flag = True
verbose = raw_input("Type \'C\' for corpus-level verbosity:    ")
if verbose == "C":
	print_flag = True

corpus_filename = "english1000.txt"
hidden_states = [ 0, 1 ]

hmm = HMM(hidden_states, corpus_filename)

print "\n--\n\nType \'T\' to test the probability of a sample word from both directions. Type \'C\' to run through the plogs in the corpus (not advised in verbose mode) and then Viterbi parse user-specified words. Type \'Q\' to sample the unit square in terms of starting distributions for A. This will print the full output in terms of the log ratios of characters in the distribution B to a file \"sample.txt\" and will print out the distribution corresponding to the lowest plog sum."
input_string = raw_input(">    ")
if input_string == "T":
	print "\n--\n\n"
	test_probability(hmm)
if input_string == "C":
	# Begin expectation-maximization
	print "\n--\n\n"
	plog_sum = hmm.expectation()
	delta = plog_sum
	i = 0

	if print_flag:
		print "\n\n--\n\nplog sum at iteration " + i.__str__() + ": " + plog_sum.__str__()
		print "\\Delta = " + delta.__str__()
		print "\n\n--\n"

	# Run until the plog doesn't change very much
	while delta > 0.001:
		i += 1

		if print_flag:
			print "\n\n--\nITERATION " + i.__str__() + ":\n"

		# Run E-M
		hmm.maximization()
		new_plog = hmm.expectation()

		# Consider the change in plog
		delta = fabs(new_plog - plog_sum)
		plog_sum = new_plog
		#if print_flag:
		print "\n\n--\n\nplog sum at iteration " + i.__str__() + ": " + plog_sum.__str__()
		print "\\Delta = " + delta.__str__()
		# Printing this here because the HMM class doesn't know that it has two states
		print "\n\nLog emission ratios ((log(B_{l, 0} + 0.001) / (B_{l, 1} + 0.001))):\n"
		ratios = sorted([(log((hmm.B[(0, char)] + 0.001) / (hmm.B[(1, char)] + 0.001), 2), char) for char in hmm.output_states])
		for (ratio, char) in ratios:
			print "    " + char + ": " + ratio.__str__()
		print ""
		print "\n\n--"

	print "HMM terminated after " + i.__str__() + " iterations; total plog = " + plog_sum.__str__() + "\n\nThe program will now parse words you enter.\n\n"

	while True:
		word = raw_input("Enter a word to Viterbi-parse its underlying vocality: ")
		hmm.viterbi_parse(word)


if input_string == "Q":
	# Sample an 8x8 grid of distributions

	output_file = open("sample.txt", 'w')

	lowest_plog = 99999999999999999999
	lowest_cell = (0, 0)
	highest_plog = 0
	start_B = deepcopy(hmm.B)
	start_Pi = deepcopy(hmm.Pi)
	lowest_B = hmm.B

	plot_labels = [(1/float(16) + float(n) / 8).__str__() for n in range(8)]
	values = [[0 for j in range(8)] for i in range(8)]

	for i in range(8):
		for j in range(8):
			hmm.B = deepcopy(start_B)
			hmm.Pi = deepcopy(start_Pi)
			hmm.softcounts = dict()

			hmm.A[0, 0] = (1/float(16)) + i * (1/float(8))
			hmm.A[0, 1] = 1.0 - hmm.A[0, 0]

			hmm.A[1, 1] = (1/float(16)) + j * (1/float(8))
			hmm.A[1, 0] = 1.0 - hmm.A[1, 0]

			plog_sum = hmm.expectation()
			delta = plog_sum

			k = 0

			while delta > 0.001 and k < 100:
				k += 1

				# Run E-M
				hmm.maximization()
				new_plog = hmm.expectation()

				# Consider the change in plog
				delta = fabs(new_plog - plog_sum)
				plog_sum = new_plog


			output_file.write("\n\nFor cell (" + i.__str__() + ", " + j.__str__() + "):\n")
			output_file.write("Total plog = " + plog_sum.__str__() + "\n")
			output_file.write("Smoothed log emission ratios ((log(B_{l, 0} + 0.0001) / (B_{l, 1} + 0.0001))):\n")
			ratios = sorted([(log((hmm.B[(0, char)] + 0.001) / (hmm.B[(1, char)] + 0.001), 2), char) for char in hmm.output_states])
			for (ratio, char) in ratios:
				output_file.write("    " + char + ": " + ratio.__str__())

			if plog_sum < lowest_plog:
				lowest_plog = plog_sum
				lowest_cell = (i, j)
				lowest_B = hmm.B

			if plog_sum > highest_plog:
				highest_plog = plog_sum

			values[i][j] = plog_sum

	fig, ax = plt.subplots()
	im = ax.imshow(values)
 
	ax.set_xticks(range(8))
	ax.set_yticks(range(8))
	ax.set_xticklabels(plot_labels)
	ax.set_yticklabels(plot_labels)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(8):
	    for j in range(8):
	        text = ax.text(j, i, values[i][j], ha="center", va="center", color="w")

	ax.set_title("plog values with respect to hmm.A[0, 0] and hmm.A[1, 1] when pi(0) = " + hmm.Pi[0])
	fig.tight_layout()
	plt.show()

	print "\n\nLowest plog is " + lowest_plog.__str__() + " found at cell " + lowest_cell.__str__() + "\n"
	print "Smoothed log emission ratios ((log(B_{l, 0} + 0.001) / (B_{l, 1} + 0.001))):\n"
	ratios = sorted([(log((hmm.B[(0, char)] + 0.001) / (hmm.B[(1, char)] + 0.001), 2), char) for char in hmm.output_states])
	for (ratio, char) in ratios:
		print "    " + char + ": " + ratio.__str__()
		


no_terminate = raw_input("\n--\n\nPress enter key to exit.")