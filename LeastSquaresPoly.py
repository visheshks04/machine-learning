import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fit(X, Y, degree=1):
    
    B = np.array([sum(X**i*Y) for i in range(degree, -1,-1)])
    A = []

    for i in range(degree+1):
        power = 2*degree-i
        A.append([sum(X**(power-j)) for j in range(degree+1)])

    A_inverse = np.linalg.inv(A)
    unknown_matrix = A_inverse.dot(B)

    return unknown_matrix

number_of_points = 50
points = [np.arange(-number_of_points,number_of_points), np.array([-x**3+2*x**2+3+40000*np.random.random() for x in range(-number_of_points,number_of_points)])]
degree = 3

X = fit(points[0],points[1], degree = degree)

print("m = %s and c = %s"%(X[:-2],X[-1]))
#PLOT
x = np.array([i for i in np.arange(-number_of_points,number_of_points,0.001)])

y = 0
for i in range(degree+1):
    y+=X[i]*(x**(degree-i))

plt.style.use('dark_background')
plt.scatter(points[0], points[1], color = "chartreuse", s=10)
plt.plot(x,y, color = "white")
plt.show()
