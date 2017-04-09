from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pygame

class Mass:
    def __init__(self,(x,y),radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.colour = (255, 0, 0)
        self.thickness = 0
    def display(self):
        pygame.draw.circle(screen, self.colour, (self.x, self.y), self.radius, self.thickness)

def rhs(state, t):
    dydx = np.zeros_like(state)
    for i in range(N):
        #print i
        dydx[i] = state[i+N]
        j = i + N
        dydx[j] = -(k/m)*(state[i] - state[2*N])
    dydx[2*N] = state[len(state)-1]
    dydx[len(state)-1] = -(K/M)*state[2*N] + (k/M)*np.sum(state[:N] - state[2*N])
    return dydx

def solver():
    y = np.zeros((1,N))
    ydot = np.full((1,N),V)
    Y = Ydot = 0
    state = np.concatenate((y,ydot)) #intial conditions
    state = np.append(state,Y)
    state = np.append(state,Ydot)
    sol = np.zeros((1,2*N+2))
    sol[0,:] = state #enter initial conditions

    t_i = 0.0
    while t_i<t_end:
        t = np.arange(t_i, t_i+delta+dt, dt)
        ic = sol[int(t_i/dt),:]
        temp_sol = integrate.odeint(rhs,ic,t) #solving the ode between half periods
        t_i = t_i + delta
        sol = np.vstack((sol,temp_sol))
        sol[(int(t_i/dt),target+N)] = -1 * sol[(int(t_i/dt),target+N)] #inverting the velocity of target
    return sol

#physical parameters
M = 64.0;m = 15.0;k = 5.0;K = 12.0
N = 100 #number of m's
V = 1.0 #init vel of m's
wt = np.sqrt(k/m) #$\omega_{t}$
wp = 1.5*wt #$\omega_{+}$
wm = 0.5*wt #$\omega_{-}$

#Q = int(np.pi/4 * np.sqrt(N)) #theoretical limit of no. of iterations
target = 3 #target entry out of [0,1,2,3,...]
delta = 2*np.pi/wt #time period of tapping = half the time period of target
Q = int( np.sqrt(N)) #theoretical limit of no. of iterations
t_end = delta*Q #corresponding time limit
res = 20 #resolution
dt = delta/res #each half cycle is divided with a resolution of res

solution = solver()
#print solution[res*Q-1] #test

t = np.arange(0,dt*len(solution),dt)
plt.plot(t,(solution[:,target+N]),'r--')
plt.plot(t,(solution[:,target+N-1]),'b--')
plt.xlabel("t")
plt.ylabel("velocity")
plt.show()
#graphics
'''
(width,height) = (800,600)
rad = 10 #radius of each m
x0 = width/(N+1)
y0 = height/2

background_colour = (20,255,200)
screen = pygame.display.set_mode((width, height))

pygame.display.set_caption('Grover\'s Algorithm with Coupled Oscillators')
screen.fill(background_colour)

masses = []
for i in range(N):
    masses.append(Mass((x0+i*x0,y0),rad))
for mass in masses:
    mass.display()

pygame.display.flip() #to display the window
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
'''
