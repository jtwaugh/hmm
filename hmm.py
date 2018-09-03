import random
from distribute import distribute_random, distribute_pi
from math import log, fabs

# Model values

class HMM():

	def __init__(self, _hidden_states, filename, _verbose_flag, _print_flag):
		
		# Essential mathematical components
		self.A = dict()
		self.B = dict()
		self.Pi = dict()
		self.hidden_states = _hidden_states
		self.output_states = set([])

		# For the sake of convenience
		self.softcount = dict()
		self.verbose_flag = _verbose_flag
		self.print_flag = _print_flag
		
		# For the corpus
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

		# Randomly initialize the distributions
		distribute_random(self.A, self.hidden_states, self.hidden_states)
		distribute_random(self.B, self.hidden_states, self.output_states)
		distribute_pi(self.Pi, self.hidden_states)


		# Print out the whole process if we're flagged
		if self.verbose_flag:
			print "\n\n--\n\n========================"
			print "==   Initialization   =="
			print "========================"

		for from_state in self.hidden_states:
			if self.verbose_flag:
				print "\n--\n\nCreating state " + from_state.__str__() + "\n\n"
				print "Transitions:\n"
			transition_sum = 0.0
			for to_state in self.hidden_states:
				if self.verbose_flag:
					print "    To state     " + to_state.__str__() + ":    " + self.A[(from_state, to_state)].__str__()
				transition_sum += self.A[(from_state, to_state)]
			if self.verbose_flag:
				print "\nTotal: " + transition_sum.__str__()
				print "\n\nEmissions:\n"
			emission_sum = 0.0
			for to_state in self.output_states:
				if self.verbose_flag:
					print "    To letter    " + to_state.__str__() + ":    " + self.B[(from_state, to_state)].__str__()
				emission_sum += self.B[(from_state, to_state)]
			if self.verbose_flag:
				print "\nTotal: " + emission_sum.__str__()

		if self.verbose_flag:
			print "\n--\n"
			print " Starting distribution:\n"
		start_sum = 0.0
		for state in self.hidden_states:
			if self.verbose_flag:
				print "    For state    " + state.__str__() + ":    " + self.Pi[state].__str__()
			start_sum += self.Pi[state]
		if self.verbose_flag:
			print "\nTotal: " + start_sum.__str__()


	# Helper function to print

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


	# Compute probabilities for an individual word

	def forward(self, word):
		# Initialize the distribution
		alpha = dict()

		# Print if flagged
		if self.verbose_flag:
			print "\n--\n\nComputing forward probabilities."

		# Initialize \alpha values to the initial distribution
		for state in self.hidden_states:
			alpha[(state, 0)] = self.Pi[state]

		# Moving forward, compute new alpha values from probability products
		for t in range(1, len(word) + 1):

			# Print if flagged
			if self.verbose_flag:
				print "\n\n    Time " + t.__str__() + ": \'" + word[t-1] + "\'"

			# Keep a running sum at each time t
			t_sum = 0.0
			# Run through posssible next states - do this explicitly for the sake of debug printing
			for to_state in self.hidden_states:
				alpha[(to_state, t)] = 0

				# Print if flagged
				if self.verbose_flag:
					print "        To state " + to_state.__str__()

				# Find the forward probability given the next letter
				for from_state in self.hidden_states:
					increment = alpha[(from_state, t-1)] * self.B[(from_state, word[t-1])] * self.A[(from_state, to_state)]
					alpha[(to_state, t)] += increment

					# Print if flagged
					if self.verbose_flag:
						print "            From state " + from_state.__str__() + ": \\alpha_{" + from_state.__str__() + ", " + (t-1).__str__() + "} \cdot b_{" + from_state.__str__() + ", " +  word[t-1] + "} \cdot a_{" + from_state.__str__() + ", " + to_state.__str__() + "} = " + increment.__str__() 
				
				# Print if flagged
				if self.verbose_flag:
					print "        \\alpha_{" + to_state.__str__() + ", " + t.__str__() + "} = " +  alpha[(to_state, t)].__str__()
				
				# Add the probability from the current state to the sum for t
				t_sum += alpha[(to_state, t)]
			
			# Print if flagged
			if self.verbose_flag:
				print "\n    \sum_{x \in X} \\alpha_{x, " + t.__str__() + "} = " + t_sum.__str__()

		# Print if flagged
		if self.verbose_flag:
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
		if self.verbose_flag:
			print "\n--\n\nComputing backward probabilities."

		for s in self.hidden_states:
			beta[(s, len(word))] = 1

		# Run through the word explicitly for the sake of debug printing
		for t in range(len(word), 0, -1):
			
			# Print if flagged
			if self.verbose_flag:
				print "\n\n    Time " + t.__str__() + ": \'" + word[t-1] + "\'"
			
			# Keep a running sum at each time t
			t_sum = 0.0
			
			for from_state in self.hidden_states:
				# Initialize \beta
				beta[(from_state, t-1)] = 0.0

				# Print if flagged
				if self.verbose_flag:
					print "        From state " + from_state.__str__()

				# Find the backward probability given the last letter
				for to_state in self.hidden_states:
					increment = beta[(to_state, t)] * self.B[(from_state, word[t-1])] * self.A[(from_state, to_state)]
					beta[(from_state, t-1)] += increment

					# Print if flagged
					if self.verbose_flag:
						print "            To state " + to_state.__str__() + ": \\beta_{" + to_state.__str__() + ", " + (t+1).__str__() + "} \cdot b_{" + from_state.__str__() + ", " +  word[t-1] + "} \cdot a_{" + from_state.__str__() + ", " + to_state.__str__() + "} = " + increment.__str__() 
				
				# Add the probability from the current state to the sum for t
				t_sum += beta[(from_state, t-1)]

				# Print if flagged
				if self.verbose_flag:
					print "\n    \sum_{x \in X} \\beta_{x, " + t.__str__() + "} = " + t_sum.__str__()

		# Print if flagged
		if self.verbose_flag:
			print "\n--\n\n"
			for t in range(0, len(word)+1):
				print "Time " + t.__str__() + ":"
				for state in self.hidden_states:
					print "    \\beta_{" + state.__str__() + ", " + t.__str__() + "} = " + beta[(state, t)].__str__()
		
		return beta


	# E-M functions

	def expectation(self):
		# Reset softcounts
		self.softcount = dict()
		
		# Set initial values
		likelihood = 0.0

		# Open and read file
		corpus_file = open(self.corpus_name, "r")
		line = corpus_file.readline().strip('\n').lower()
		
		# Print if flagged
		if self.print_flag:
			print "\n\nPlogs:\n"
		
		# Hit the corpus
		while line:
			# Append endline character
			line += "#"
			
			# Compute probabilities
			alpha = self.forward(line)
			beta = self.backward(line)
			f_prob = sum([alpha[(state, len(line))] for state in self.hidden_states])
			b_prob = sum([self.Pi[state] * beta[(state, 0)] for state in self.hidden_states])
			
			# Run through the word and tabulate softcounts
			for t in range(len(line)):
				for from_state in self.hidden_states:
					for to_state in self.hidden_states:
						if (t, line[t], from_state, to_state) not in self.softcount:
							self.softcount[(t, line[t], from_state, to_state)] = 0.0	
						self.softcount[(t, line[t], from_state, to_state)] += (alpha[(from_state, t)] * self.A[(from_state, to_state)] * self.B[(from_state, line[t])] * beta[(to_state, t+1)]) / f_prob

			# If we have agreement on the probailities, more or less
			if (fabs(f_prob - b_prob) < 0.00001):
				plog = -1 * log(f_prob)
				likelihood += plog

				# Print if flagged
				if self.print_flag:
					print "plog(\"" + line + "\") = " + plog.__str__()

			else:
				# Print for error if flagged
				print "Unacceptable probability mismatch at word " + line + ": forward (" + f_prob.__str__() + ") != backward (" + b_prob.__str__() + ")."	
			
			line = corpus_file.readline().strip('\n').lower()
		
		# Print if flagged
		if self.print_flag:
			print "\n--\n\nSum of positive logs: " + likelihood.__str__() + "\n\n--\nSoftcounts:\n"

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
		return likelihood

	def maximization(self):
		# Print first if flagged
		if self.print_flag:
			print "Distribution \\Pi:"
		# Reset Pi
		for from_state in self.hidden_states:
			softcount_i = sum([self.softcount[(0, char, from_state, to_state)] for char in self.output_states for to_state in self.hidden_states if (0, char, from_state, to_state) in self.softcount])
			old_pi = self.Pi[from_state]
			self.Pi[from_state] = 1/float(self.word_count) * softcount_i 
		
			# Print if flagged
			if self.print_flag:
				print "    For state " + from_state.__str__() + " was    " + old_pi.__str__() + "\nRecomputing... \n\n    For state " + from_state.__str__() + " is now " + self.Pi[from_state].__str__() + "\n"

		# Print first if flagged
		if self.print_flag:
			print "\nDistribution A:"
		# For each (i, j), assign A_{i, j}	
		for from_state in self.hidden_states:
			a_denom = sum([self.softcount[(t, l, from_state, k)] for t in range(self.longest) for l in self.output_states for k in self.hidden_states if (t, l, from_state, k) in self.softcount])

			# Print if flagged
			if self.print_flag:
				print "\n    From state " + from_state.__str__() + ":\n\n    Computed the denominator at state i = " + from_state.__str__() + " (sum over hidden_states, output_states, and t): " + a_denom.__str__() + "\n"

			for to_state in self.hidden_states:
				a_num = sum([self.softcount[(t, l, from_state, to_state)] for t in range(self.longest) for l in self.output_states if (t, l, from_state, to_state) in self.softcount])
				old_a = self.A[(from_state, to_state)]
				self.A[(from_state, to_state)] = a_num / a_denom

				# Print if flagged
				if self.print_flag:
					print "\n        Computed the numerator at states i = " + from_state.__str__() + "; j = " + to_state.__str__() + " (sum over output_states and t): " + a_num.__str__() + "\n\n        To state " + to_state.__str__() + " was    " + old_a.__str__() + "\n        To state " + to_state.__str__() + " is now " + self.A[(from_state, to_state)].__str__()

		# Print first if flagged
		if self.print_flag:
			print "\nDistribution B:"
		# For each (i, l), assign B_{i, l}
		for from_state in self.hidden_states:
			b_denom = sum([self.softcount[(t, m, from_state, j)] for t in range(self.longest) for m in self.output_states for j in self.hidden_states if (t, m, from_state, j) in self.softcount])
			
			# Print if flagged
			if self.print_flag:
				print "\n    From state " + from_state.__str__() + ":\n    Computed the denominator at state i = " + from_state.__str__() + " (sum over hidden_states, output_states, and t): " + b_denom.__str__() + "\n"			

			for char in self.output_states:
				b_num = sum([self.softcount[(t, char, from_state, j)] for t in range(self.longest) for j in self.hidden_states if (t, char, from_state, j) in self.softcount])
				old_b = self.B[(from_state, char)]
				self.B[(from_state, char)] = b_num / b_denom

				# Print if flagged
				if self.print_flag:
					print "\n        Computed the numerator at states i = " + from_state.__str__() + "; \\ell = " + char + " (sum over output_states and t): " + b_num.__str__() + "\n\n        To state " + char + " was    " + old_b.__str__() + "\n        To state " + char + " is now " + self.B[(from_state, char)].__str__()

		# Print if flagged
		if self.print_flag:
			print "\n\n--\n"


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
				probability = lambda k : max_probability[(k, i-1)] * self.A[(k, state)] * self.B[(state, word[i])]
				max_probability[(state, i)] = max(map(probability, self.hidden_states))
				argmax_state[(state, i)] = max(self.hidden_states, key=probability)
		
		# Connect the path
		path[len(word) - 1] = max(self.hidden_states, key=(lambda k : max_probability[(k, len(word) - 1)]))
		for i in (range(1, len(word))[::-1]):
			path[i-1] = argmax_state[(path[i], i)]
		print "Viterbi parse: " + path.__str__()
