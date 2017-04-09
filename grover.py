import numpy as np
from numpy import dot, outer, ceil, identity, log2, array, zeros, sqrt, pi

N = 2**11; wanted = 5;

translated = array([np.zeros(N) for j in range(0, N)])
for j in range(0, N): translated[j][j] = 1

state = sum(translated)/sqrt(N)

diffusion = 2*outer(state, state) - identity(N)
oracle = identity(N) - 2*outer(translated[wanted], translated[wanted])

for j in range(0, int(pi*sqrt(N)/4.)):
	print str(j)+": "+str(dot(state, translated[wanted]) ** 2);
	state = dot(diffusion, dot(oracle, state))