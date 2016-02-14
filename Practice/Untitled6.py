def find(word,letter,n):
    index=n
    while index < len(word):
         
        if word[index]==letter:
            
            print index
        index=index+1
    
find('hwuseware','w',3)

def count(word,n):
    count =0
    for letter in word:
        if letter == n:
            count=count+1
    return count
    
print count('banana','a')

help eval()
