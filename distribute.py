import random

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