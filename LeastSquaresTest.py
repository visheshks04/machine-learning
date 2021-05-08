import matplotlib.pyplot as plt
import numpy as np

number_of_points = 75 #Number of random points to test
points = [np.array([i for i in range(number_of_points)]), np.array([0.4*i+3+3*np.random.normal() for i in range(number_of_points)])]

sum_of_x = sum(points[0])
sum_of_x_square = sum(points[0]**2)
sum_of_y = sum(points[1])
sum_of_xiyi = sum(points[0]*points[1])


#Solving AX = B for X
A = np.array([[sum_of_x_square, sum_of_x],[sum_of_x, number_of_points]])
B = np.array([sum_of_xiyi, sum_of_y])
A_inverse = np.linalg.inv(A)
X = A_inverse.dot(B)
m,c = X[0],X[1]

print("m = %s and c = %s"%(m,c))


#PLOT
x = np.array([i for i in range(number_of_points)])
y = m*x+c
plt.style.use('dark_background')
plt.scatter(points[0], points[1], color = "chartreuse", s=10)
plt.plot(x,y, color = "white")
plt.show()
