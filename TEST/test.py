def encode(state,k):
	s = 0
	m = len(state)
	for i in range(m):
		j = m - i -1
		s += state[j]*k**i
	return s

def stateDecoder(k,state,m):
	'''
	Method that decodes stae from number to 
	input vector.
	'''
	done = False
	out = []
	q,r = 0,0
	q = state
	while not done:
		new_q = q // k
		print(new_q)
		r = q % k
		q = new_q
		out.append(r)
		if new_q == 0:
			done = True
	while len(out) < m:
		out.append(0)

	print(out)
	out = out[::-1]
	return out

k = 3
m = 1
state = [2]
print("Inital state: ",state)
num = encode(state,k)
print("Encoded state: ",num)
print("Decoded State: ",stateDecoder(k,num,m))