'''
find the sq of a using guessing parameter x
'''
import math
def sq(a,x):
     count =0
     while True:
         
         #newton's method 
         y=(x+a/x)/2
         
         if abs(x-y)<0.000000000001:
             break
         else:
             x=y*1.0
            
             count+=1
              
     return y
#the difference between sq function and built-in function for int from 1 to 10     
a=1          
while a<=10:
          print a, sq(a,3),math.sqrt(a),abs(sq(a,3)-math.sqrt(a))
          a +=1
print 10e-15
