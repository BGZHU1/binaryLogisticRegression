
from scipy.optimize import check_grad



def function(x, a, b) :
    print(x)
    return x[0]**2 - 0.5 * x[1]**3 + a**2 + b

def grad(x, a, b):
    print(x)
    return [2 * x[0], -1.5 * x[1]**2 + 2*a + b]

a = 5
b = 10
result=check_grad(function, grad, [1.5, -1.5], a, b)
print (result)
