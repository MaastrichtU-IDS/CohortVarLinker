'''
Created on August 15, 2024

@note Fetches topK similar nodes to each node from a similarity matrix
      provides two implementation per method: pythonic and tensor based;
      the former one is used with CPU and the latter one with GPU 
            
'''
import numpy as np
import tensorflow as tf

def get_listMLE_topK_BCH_py(result_matrix, topK):    
    '''                
        @return: top_indices: |batch|*topK matrix contians indices of topK nodes to each target node 
    '''      
    if topK > result_matrix.shape[1]:
        top_indices = np.argsort(-result_matrix, axis=1)[:, :result_matrix.shape[1]]                  
    else:
        top_indices = np.argsort(-result_matrix, axis=1)[:, :topK]           
    return top_indices

def get_listMLE_topK_BCH_tb(result_matrix,topK):    
    '''
        @note: the sorting is performed row by row by using tf.map_fn                
        @return: top_indices: |batch|*topK tensor contians descending order of nodes 
    '''     
    if topK > result_matrix.shape[1]:
        _, topk_indices = tf.nn.top_k(result_matrix, k=result_matrix.shape[1], sorted=True)
    else:
        _, topk_indices = tf.nn.top_k(result_matrix, k=topK, sorted=True)
    return topk_indices
