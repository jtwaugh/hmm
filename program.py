from hmm import HMM, print_flag, verbose_flag
from math import log, fabs

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

print "========================="
print "== Hidden Markov Model =="
print "=========================\n\n"

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

while True:
	print "\n--\n\nOptions:\n* Type \'T\' to train the model on the corpus;\n* Type \'P\' to test the probability of an output string;\n* Type \'V\' to Viterbi-parse a hypothetical output string;\n* Type \'S\' to sample a model across the parameter space A.\n"
	input_string = raw_input(">    ")
	if input_string == "P":
		print "\n--\n\n"
		test_probability(hmm)
	if input_string == "V":
		valid = False
		while not valid:
			word = raw_input("\n\nEnter word to parse (please no invalid characters): ").lower()
			if set(word).issubset(hmm.output_states):
				valid = True
		hmm.viterbi_parse(word)
	if input_string == "T":
		# Begin expectation-maximization
		print "\n--\n\n"
		likelihood = hmm.expectation()
		delta = likelihood
		i = 0

		if print_flag:
			print "\n\n--\n\nplog sum at iteration " + i.__str__() + ": " + likelihood.__str__()
			print "\\Delta = " + delta.__str__()
			print "\n\n--\n"

		# Run until the plog doesn't change very much
		while delta > 0.001:
			i += 1

			if print_flag:
				print "\n\n--\nITERATION " + i.__str__() + ":\n"

			# Run E-M
			hmm.maximization()
			new_likelihood = hmm.expectation()

			# Consider the change in plog
			delta = fabs(new_likelihood - likelihood)
			likelihood = new_likelihood
			#if print_flag:
			print "\n\n--\n\nplog sum at iteration " + i.__str__() + ": " + likelihood.__str__()
			print "\\Delta = " + delta.__str__()
			# Printing this here because the HMM class doesn't know that it has two states
			print "\n\nLog emission ratios ((log(B_{l, 0} + 0.001) / (B_{l, 1} + 0.001))):\n"
			ratios = sorted([(log((hmm.B[(0, char)] + 0.001) / (hmm.B[(1, char)] + 0.001), 2), char) for char in hmm.output_states])
			for (ratio, char) in ratios:
				print "    " + char + ": " + ratio.__str__()
			print ""
			print "\n\n--"

		print "HMM terminated after " + i.__str__() + " iterations; total plog = " + likelihood.__str__() + "\n\nThe program will now parse words you enter.\n\n"

	if input_string == "S":
		sample(hmm, 4)

no_terminate = raw_input("\n--\n\nPress enter key to exit.")