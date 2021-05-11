import numpy as np

# evaluate the top three resturned results
def evaluateTopThree(data):
    j = 0
    returned = 0
    for file,rankings in data:
        if file[0]=='c':
            true = file[0:15]
        else:
            true = file[0:9] 
        for i in range(len(rankings)):
            if i<3:
                predicted = rankings[i]
                if true == predicted[0:-8]:
                   returned += 1
        j += 1
    x = returned/j             
    return x

# find MAP score
def evaluate(data):
    evaluate = []
    j = 0
    avg_p_sum = 0
    for file,rankings in data:
        row = []
        rel_sum = 0
        p_sum = 0
        avg_p = 0
        if file[0]=='c':
            true = file[0:15]
        else:
            true = file[0:9] 
        for i in range(len(rankings)):
            rank = i+1
            predicted = rankings[i]
            rel = relevance(true, predicted)
            p = precision(rank, rel, rel_sum)
            r = recall(rank,rel,rel_sum)
            if p==0 and r==0:
                f = 0
            else:
                f = ((2*p*r)/(p+r))
            rel_sum += rel
            p_sum += p
            row.append((rank,rel,p,r,f))
        
        avg_p = avg_precision(row)
        evaluate.append((row,avg_p))
        avg_p_sum += avg_p
        j += 1
    
    m = (avg_p_sum/j)
    return evaluate, m


# find relevance score
def relevance(true,predicted):
    if true == predicted[0:-8]:
        rel = 1
    else:
        rel = 0
    return rel
   
       
   # find precision score
def precision(rank,rel,rel_sum):
    p = (rel_sum+rel)/rank
    return p


# find recall score
def recall(rank,rel,rel_sum):
    r = rel_sum+rel
    return r


# find average precision of multiple queries
def avg_precision(row):
    avg_p = 0
    for i in range(len(row)):
        rel = row[i][1]
        p = row[i][2]
        avg_p += (rel*p)
    return avg_p
        

def execute():

    array = np.load("count.npy",allow_pickle=True)
    evaluated_rankings, MAP = evaluate(array)
    print("Mean Average Precision = ",MAP)
    print("Returned song accuracy = ",x)
    