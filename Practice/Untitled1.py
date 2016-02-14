import math 
a,b= raw_input('input a,b').split()
a,b= [int(a),int(b)]


def fermat_check(a,b):
    if math.pow(a,4)+math.pow(b,4)==math.pow(3,4):
         print ('fermat is wrong')
    else:
        print('this is not working')
        
        return 
fermat_check(a,b)