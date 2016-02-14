from __future__ import division
import warnings
import math


def square_root(a,x,max_iter=100):
    """
    Use netwton method to compute square root of a, x is the initiali geuss
    a = Number
    x = best geuss at square root
    """
    print("\nNewtone Method Go!")
    
    counter = 0
    while True:
       
        # update rule
        y = (x + (a/x)) / 2
        
        print(y)
        
        
        # Convergence
        if abs(y-x)<0.00000000000001:
            print('Iterations: '+str(counter))
            return y 
        
        # Too many iterations
        if counter > max_iter:
            print('Iterations: '+str(counter))
            warnings.warn("Convergence Failed after "+str(max_iter)+" iterations")
            return y
        
        # Update Guess
        x = y
        
        
        counter += 1
        
    return y

if __name__ == "__main__":        
    print square_root(4,2.5)
    print square_root(10,4)
    print square_root(10,-765357)
    print math.sqrt(10)