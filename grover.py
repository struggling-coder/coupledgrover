import numpy as np
from numpy import dot, outer, ceil, identity, log2, array, zeros, sqrt, pi
import matplotlib.pyplot as plt

N = 2**8; wanted = 5;
a = []; b = []


translated = array([np.zeros(N) for j in range(0, N)])
for j in range(0, N): translated[j][j] = 1

state = sum(translated)/sqrt(N)
i = state

diffusion = 2*outer(state, state) - identity(N)
oracle = identity(N) - 2*outer(translated[wanted], translated[wanted])

for j in range(0, int(pi*sqrt(N)/4.)):
	a.append((dot(state, translated[wanted]) ** 2))
	print (dot(state, translated[wanted]) ** 2)
	b.append((dot(state, translated[wanted+1]) ** 2))
	state = dot(diffusion, dot(oracle, state))

p1, = plt.plot(range(len(a)), a, 'r.')
p2, = plt.plot(range(len(b)), b, 'b.')
plt.legend([p1,p2],["Target","Non-target"],loc = 'best')
plt.xlabel(r'Step number')
plt.ylabel(r'Probability')
plt.title("Grover\'s Algorithm")
plt.show()
