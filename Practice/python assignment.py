''' use fuzzy logic table joining to predict the 13th person's group
'''
import numpy as np
import panda as pd
import editdistance as ed

#use editdistance to creat unique keys to join data
def similarity(A,B):
    vec=[ed.eval(a,b) for a,b in zip(A,B)]
    print vec
    res=np.sum(vec)
    return res
''' check similarity function    
X=['banana','bahama']
Y=['apple','banana']
print similarity(X,Y)
'''
#macth function to match the key in the table to minimize similarity score 
def macth(X,Y):
    