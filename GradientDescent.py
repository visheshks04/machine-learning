import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(X, y):
    n = len(y)
    m = 0
    b = 0

    iterations = 10000
    learning_rate = 0.001

    for _ in range(iterations):

        mean_squared_error = sum(y - (m*X + b))/n
        derivative_m = (-2/n)*sum(X*(y - (m*X + b)))
        derivative_b = (-2/n)*sum(y - (m*X + b))
    
        m = m-learning_rate*derivative_m
        b = b-learning_rate*derivative_b

        print("m = {},\tb = {},\tMSE = {}".format(m,b,mean_squared_error))
    
    return m,b

number_of_points = 20
points = [np.array([i for i in range(number_of_points)]), np.array([0.4*i+3+6*np.random.random() for i in range(number_of_points)])]

m,b = gradient_descent(points[0], points[1])
#PLOT
x = np.array([i for i in range(number_of_points)])
y = m*x+b
plt.style.use('dark_background')
plt.scatter(points[0], points[1], color = "chartreuse", s=10)
plt.plot(x,y, color = "white")
plt.show()