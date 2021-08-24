import numpy as np

time_stamp = 0
init = 0
vol_lvl = 4
step = 3

inp = [[time_stamp,
		float(init),
		vol_lvl]]
inp = np.transpose([inp[-1]] * step)

new = [[1,0,4]]

inp = np.append(inp,np.transpose(new),axis=1)

inp = np.delete(inp,0,1)

print(inp)