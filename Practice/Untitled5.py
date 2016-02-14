''' estimate pi value
'''
import math
# factorial calculation
def fac(x):
    if x==0:
        return 1
    if x==1:
        return 1
    if x%1==0:
        recursion=fac(x-1)
        result=x*recursion
        return result
    if x%1 !=0:
        print 'wrong input'

# square root calculation
def sq(x,a):
    while True:
        y=(x+a/x)/2
        if abs(y-x)<0.0000001:
            break
        else:
            x=y*1.0
    return y
        
def sqrt(x):
    return sq(3,x)
    
#power function
def power(x,y):
    result=1
    while y>0:
        itr=x
        result=result*itr
        y=y-1
    return result 

   

def estimate_pi():
    k=0
    res=0
    
    fact=2 * math.sqrt(2) / 9801
    
    while True:
          lumpsum=fac(4*k) * (1103 + 26390*k)/(fac(k)**4 * 396**(4*k))
          term=fact*lumpsum
          res+=term
          if abs(term)<1e-15:
              break
          k+=1
    return 1/res

print estimate_pi()
   
    


