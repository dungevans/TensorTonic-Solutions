def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x0 = 0
    for i in range (steps) : 
        df = 2*a*x0 + b 
        x0 = x0 -lr*df
    return x0  
    