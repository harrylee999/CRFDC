# Simplified MI computation code from https://github.com/ravidziv/IDNNs
import numpy as np
import pandas as pd 
from collections import Counter
def MI_cal(label_matrix, layer_T, NUM_TEST_MASK):
    '''
    Inputs:  
    - size_of_test: (N,) how many test samples have be given. since every input is different
      we only care the number.
    -  label_matrix: (N,C)  the label_matrix created by creat_label_matrix.py.
    -  layer_T:  (N,H) H is the size of hidden layer
    Outputs:
    - MI_XT : the mutual information I(X,T)
    - MI_TY : the mutual information I(T,Y)
    '''
    MI_XT=0
    MI_TY=0
    layer_T = Discretize(layer_T)
    XT_matrix = np.zeros((NUM_TEST_MASK,NUM_TEST_MASK))
    Non_repeat=[]
    mark_list=[]
    for i in range(NUM_TEST_MASK):
        pre_mark_size = len(mark_list)
        if i==0:
            Non_repeat.append(i)
            mark_list.append(i)
            XT_matrix[i,i]=1
        else:
            for j in range(len(Non_repeat)):
                if (layer_T[i] ==layer_T[ Non_repeat[j] ]).all():
                    mark_list.append(Non_repeat[j])
                    XT_matrix[i,Non_repeat[j]]=1
                    break
        if pre_mark_size == len(mark_list):
            Non_repeat.append(i)
            mark_list.append(Non_repeat[-1])
            XT_matrix[i,Non_repeat[-1]]=1          
    P_layer_T = np.sum(XT_matrix,axis=0)/float(NUM_TEST_MASK)
    P_sample_x = 1/float(NUM_TEST_MASK)
    for i in range(NUM_TEST_MASK):
        MI_XT+=P_sample_x*np.log2(1.0/P_layer_T[mark_list[i]])
    pdf_y = Counter();  pdf_yt = Counter();pdf_t = Counter()
    for i in range(NUM_TEST_MASK):
        pdf_t[(label_matrix[i],)+tuple(layer_T[i].astype(int))] = P_layer_T[mark_list[i]]
        pdf_y[label_matrix[i]] += 1/float(NUM_TEST_MASK)
        pdf_yt[(label_matrix[i],)+tuple(layer_T[i].astype(int))] += 1/float(NUM_TEST_MASK)
  
    MI_TY= 0
    for i in pdf_yt:
        # P(t,y), P(t) and P(y)
        p_yt = pdf_yt[i]; p_t = pdf_t[i]; p_y = pdf_y[i[0]]
        # I(X;T)
        MI_TY += p_yt * np.log2(p_yt / p_t / p_y)

    return MI_XT,MI_TY

def Discretize(layer_T,NUM_INTERVALS = 10):
    '''
    Discretize the output of the neuron 
    Inputs:
    - layer_T:(N,H)
    Outputs:
    - layer_T:(N,H) the new layer_T after discretized
    '''

    
    layer_T = np.exp(layer_T - np.max(layer_T,axis=1,keepdims=True))
    layer_T /= np.sum( layer_T,axis=1,keepdims=True)


    labels = np.arange(NUM_INTERVALS)
    bins = np.arange(NUM_INTERVALS+1)
    bins = bins/float(NUM_INTERVALS)
    
    for i in range(layer_T.shape[1]):
        temp = pd.cut(layer_T[:,i],bins,labels=labels)
        layer_T[:,i] = np.array(temp)
    return layer_T
