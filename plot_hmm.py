from hmm import HMM
from copy import deepcopy
from math import log, fabs
import matplotlib
import matplotlib.pyplot as plt

def sample(hmm, resolution):
	output_file = open("sample.txt", 'w')

	lowest_plog = 99999999999999999999
	lowest_cell = (0, 0)
	highest_plog = 0
	start_B = deepcopy(hmm.B)
	start_Pi = deepcopy(hmm.Pi)
	lowest_B = hmm.B

	plot_labels = [(1/float(resolution * 2) + float(n) / resolution).__str__() for n in range(resolution)]
	values = [[0 for j in range(resolution)] for i in range(resolution)]

	for i in range(resolution):
		for j in range(resolution):
			hmm.B = deepcopy(start_B)
			hmm.Pi = deepcopy(start_Pi)
			hmm.softcounts = dict()

			hmm.A[0, 0] = (1/float(resolution * 2)) + i * (1/float(resolution))
			hmm.A[0, 1] = 1.0 - hmm.A[0, 0]

			hmm.A[1, 1] = (1/float(resolution * 2)) + j * (1/float(resolution))
			hmm.A[1, 0] = 1.0 - hmm.A[1, 0]

			likelihood = hmm.expectation()
			delta = likelihood

			k = 0

			while delta > 0.001 and k < 100:
				k += 1

				# Run E-M
				hmm.maximization()
				new_plog = hmm.expectation()

				# Consider the change in plog
				delta = fabs(new_plog - likelihood)
				likelihood = new_plog


			output_file.write("\n\nFor cell (" + i.__str__() + ", " + j.__str__() + "):\n")
			output_file.write("Total plog = " + likelihood.__str__() + "\n")
			output_file.write("Smoothed log emission ratios ((log(B_{l, 0} + 0.0001) / (B_{l, 1} + 0.0001))):\n")
			ratios = sorted([(log((hmm.B[(0, char)] + 0.001) / (hmm.B[(1, char)] + 0.001), 2), char) for char in hmm.output_states])
			for (ratio, char) in ratios:
				output_file.write("    " + char + ": " + ratio.__str__())

			if likelihood < lowest_plog:
				lowest_plog = likelihood
				lowest_cell = (i, j)
				lowest_B = hmm.B

			if likelihood > highest_plog:
				highest_plog = likelihood

			values[i][j] = likelihood

	fig, ax = plt.subplots()
	im = ax.imshow(values)
 
	ax.set_xticks(range(resolution))
	ax.set_yticks(range(resolution))
	ax.set_xticklabels(plot_labels)
	ax.set_yticklabels(plot_labels)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(resolution):
	    for j in range(resolution):
	        text = ax.text(j, i, values[i][j], ha="center", va="center", color="w")

	ax.set_title("plog values with respect to hmm.A[0, 0] and hmm.A[1, 1] when pi(0) = " + hmm.Pi[0])
	fig.tight_layout()
	plt.show()

	print "\n\nLowest plog is " + lowest_plog.__str__() + " found at cell " + lowest_cell.__str__() + "\n"
	print "Smoothed log emission ratios ((log(B_{l, 0} + 0.001) / (B_{l, 1} + 0.001))):\n"
	ratios = sorted([(log((hmm.B[(0, char)] + 0.001) / (hmm.B[(1, char)] + 0.001), 2), char) for char in hmm.output_states])
	for (ratio, char) in ratios:
		print "    " + char + ": " + ratio.__str__()