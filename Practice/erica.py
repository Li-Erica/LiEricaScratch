from __future__ import division
import numpy as np
import sklearn as sk


def main():
    #inc(5)
    combo(5,A=[0,1],B=[0,10])   
    
    return None
    
def inc(n):
    """
    """
    result = []
    A=1
    B=0
    i=0
    
    b = float(1) / n
    while i <= n:
        result = [A-b*i,B+b*i]
        print(result)
        i += 1
        
    return result
    
def combo(n,A=[0,1],B=[0,1]):
    """
    """
    stepA = (max(A)-min(A))/(n-1)
    stepB = (max(B)-min(B))/(n-1)
    
    x = np.arange(start=A[0] , stop=A[1]+stepA , step=stepA)
    y = np.arange(start=B[0] , stop=B[1]+stepB , step=stepB)
    
    for i in x:
        for j in y:
            # Evaluate Learning model for paramas A and B
            print('A:'+str(i)+' B:'+str(j))
    
    
    return None
        
        
   
    
        
    


if __name__=="__main__":
    main()










