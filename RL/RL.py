import numpy as np
from IPython.display import clear_output


from Utils.Helpers import *
from CNN.CNN import *
from SNN.SNN import *


###q-learning
q_table = np.zeros([3,4])
print(q_table)


# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

goal = 2

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 300):
	state = np.random.choice([0,1,2])

	epochs, penalties, reward, = 0, 0, 0
	done = False

	while not done:

		if np.random.uniform(0, 1) < epsilon:
			action =  np.random.choice([1,2,3,4])# Explore action space
		else:
			action = np.argmax(q_table[state]) # Exploit learned values

		inp_feat = {'V':np.array([float(action)])}
		for j in range(memory):
			name = 'S'+str(j-memory)
			inp_feat[name] = np.array([float(state)])


		next_state = int(SNN.runSNN2(snn_model,inp_feat))
		if next_state == goal:
			done = True
		reward = next_state - goal 

		old_value = q_table[state, action-1]
		next_max = np.max(q_table[next_state])

		new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
		q_table[state, action-1] = new_value


		state = next_state
		epochs += 1
		print(i)

	if i % 100 == 0:
		clear_output(wait=True)
		print(f"Episode: {i}")

print("Training finished.\n")

print(q_table)
