'''
Created on August 08, 2024; 

@NOTE: this file contains related implementation for BOTH "similarity computation" and "data preparation for graph embedding"
       1- Computation on CPU: BOTH similarity computation and selecting TopK nodes are run in CPU with multi-processing
       2- Computation on GPU: BOTH similarity computation and selecting TopK nodes are run in GPU                    
'''

import numpy as np
import math
import tensorflow as tf
import os
import fetch_topK_SPARSE_BCH as prData
import pickle
from tqdm import tqdm
from scipy.sparse import csr_matrix, csr_array
from sklearn.preprocessing import normalize
from concurrent.futures import ProcessPoolExecutor

def LINOW_bn_EMB (adj_mat=None, gr_type='', avg_deg=None, in_degrees=None, dataset_name='', iterations=5, damping_factor=0.2, bch_size_cpu=128, bch_size_gpu=128, prl_num=8, 
                  GPU=False, scaling_factor=0, loss='', write_topK_nodes=False, topK_save_path=''):
    '''       
        @param adj_mat: graph adjacency matrix
        @param gr_type: graph type
        @param avg_deg: average degree of nodes 
        @param in_degrees: in-degree of nodes
        @param dataset_name: graph name      
        @param iterations: number of iteration
        @param damping_factor: C
        @param bch_size_cpu: batch size for CPU computation 
        @param bch_size_gpu: batch size for GPU computation
        @param prl_num: number of processes for parallel computation  
        @param GPU: flag for using GPU   
        @param scaling_factor: The scaling factor t to compute topK 
        @param loss: the target loss funciton 
        @param write_topK_nodes: flag to save the topK nodes in a file 
        @param topK_save_path: address to save the topK in a file   
    '''    
    print("Batch similarity computation --> dataset:'{}', iterations:{}, C:{}, batch:{}, {}, {}".format(dataset_name,iterations,damping_factor,(bch_size_gpu if GPU else bch_size_cpu),('#processes:'+str(prl_num) if not GPU else 'single process'),('GPU' if GPU else 'CPU'))+'\n')

    ####################################################
        # Prepare weight matrix
    ####################################################
    if gr_type == 'directed': 
        topK = scaling_factor*avg_deg
    else: 
        topK = scaling_factor*2*avg_deg        
    weights = csr_matrix(1 / np.log(in_degrees + math.e)) 
    weight_matrix = csr_matrix(adj_mat.multiply(weights)).astype(dtype='float32', casting='unsafe')      
    del in_degrees
    del weights
    weight_matrix = normalize(weight_matrix, norm='l2', axis=0)
    weight_matrix = (damping_factor / 2) * weight_matrix  
    top_indices = [] ## keep the topK nodes
    num_nodes = adj_mat.shape[0]

    ####################################################
        # GPU Computation
    ####################################################            
    if GPU: 
        batches = [range(i, min(i + bch_size_gpu, num_nodes)) for i in range(0, num_nodes, bch_size_gpu)]                   
        rows, cols = csr_array.nonzero(weight_matrix)
        indices = list(zip(rows, cols))                           
        weight_matrix_tensor= tf.sparse.SparseTensor(indices=indices, values=weight_matrix.data, dense_shape=[num_nodes,num_nodes])
        for idx in tqdm(range(0, len(batches))):
            result_ = LINOW_bn_tensor_based_EMB(tf.constant(num_nodes, dtype=tf.int32), tf.constant(iterations, dtype=tf.int32),tf.constant(damping_factor, dtype=tf.float32), 
                                                  weight_matrix_tensor , tf.constant(batches[idx],dtype=tf.int32), tf.constant(topK, dtype=tf.int32))            
            with tf.device('/CPU:0'):                
                top_indices.extend(result_.numpy().tolist())

    ####################################################
        # CPU Computation
    ####################################################                        
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ## disabling GPU  
        batches = [range(i, min(i + bch_size_cpu, num_nodes)) for i in range(0, num_nodes, bch_size_cpu)]           
        for idx in tqdm(range(0, len(batches), prl_num)):                
            with ProcessPoolExecutor(max_workers=prl_num) as executor:
                futures = [executor.submit(LINOW_bn_pythonic_EMB, num_nodes, iterations, damping_factor, weight_matrix, batch, topK) for batch in batches[idx:idx + prl_num]]
                for i, future in enumerate(futures):
                    top_indices.extend(future.result().tolist()) 
                    
    ####################################################
        # Save the results if requested
    ####################################################      
    name_ = topK_save_path+dataset_name+'_LINOW_IT_'+str(iterations)+'_C_'+ str(int(damping_factor*10))+'_'+loss+'_Scl'+str(scaling_factor)
    if write_topK_nodes: ## serialized the topK for future usage     
        if not os.path.exists(topK_save_path):
            os.makedirs(topK_save_path) 
        with open(name_,'wb') as file_:
            pickle.dump(top_indices, file_)
        file_.close()    
        print('The topK nodes are saved into a binary file ...')
        
    ####################################################
        # Return the results
    ####################################################      
    return tf.constant(top_indices, dtype=tf.int32)     

def LINOW_bn_pythonic_EMB(n, K, C, W, q, topK):
    """
        Pythonic implementation for batch similarity computation and data preparation with CPU 

        @param n: number of nodes
        @param K: number of iterations
        @param C: damping factor
        @param W: column normalized weight matrix
        @param q: range of query nodes
        @param topK: number of top similar nodes  
        @return: numpy.ndarray in size |batch|*topK          
    """
    e_u = np.zeros((n, len(q)),  dtype=np.float32) ## query matrix e_u
    for i, query_node in enumerate(q):
        e_u[query_node, i] = 1

    zero = np.zeros((n, len(q)),  dtype=np.float32)            
    gamma = np.full((K + 2, K + 1), None, dtype=np.ndarray)
    for i in range(1, K + 2): 
        gamma[i - 1, i - 1] = zero
        gamma[i, 0] = W.dot(gamma[i - 1, 0]) + e_u
        for j in range(1, i):
            gamma[i, j] = W.dot(gamma[i - 1, j]) + gamma[i - 1, j - 1]
            gamma[i - 1, j - 1] = None  ## for saving memory

    Gamma = e_u
    for i in range(1, K + 1):
        Gamma = gamma[K + 1, K - i] + W.T.dot(Gamma)

    return prData.get_listMLE_topK_BCH_py(((1 - C) * Gamma).T, topK)
        
def LINOW_bn_tensor_based_EMB(n, K, C, W, q, topK):
    """
        Tensor-based implementation for batch similarity computation and data preparation with GPU

        @param n: number of nodes
        @param K: number of iterations
        @param C: damping factor
        @param W: column normalized weight matrix
        @param q: range of query nodes
        @param topK: number of top similar nodes  
        @return: RaggedTensor in size |batch|*topK          
    """
    def compute_(n, K, C, W, q, topK, e_u):        
        zero = tf.zeros((n, q.shape[0]), dtype=tf.float32)    
        gamma = tf.TensorArray(dtype=tf.float32, size=(K+2)*(K+1), infer_shape=False, dynamic_size=True,clear_after_read = False)  
        for i in range(1, K + 2):
            gamma = gamma.write((i-1)*(K+1)+(i-1), zero)
            gamma = gamma.write( i*(K+1), tf.sparse.sparse_dense_matmul(W,gamma.read((i-1)*(K+1)))+e_u)
            for j in range(1, i):
                gamma = gamma.write(i*(K+1)+j, tf.sparse.sparse_dense_matmul(W,gamma.read((i-1)*(K+1)+j)) +gamma.read((i-1)*(K+1)+(j-1)))
                gamma = gamma.write((i-1)*(K+1)+(j-1), 0.) # IMPORTANT: it is required for memory scalability 
        Gamma = e_u
        for i in tf.range(1, K + 1):
            Gamma = gamma.read((K+1)*(K+1)+(K-i)) + tf.sparse.sparse_dense_matmul(tf.sparse.transpose(W),Gamma)    
        
        return prData.get_listMLE_topK_BCH_tb( tf.transpose(tf.math.scalar_mul(1-C,Gamma)), topK)
                
    e_u = tf.Variable(tf.zeros([n, tf.shape(q)[0]]), dtype=np.float32)   
    row_idx = q
    col_idx = tf.range(tf.shape(q)[0])
    indices = tf.stack([row_idx, col_idx], axis=1)
    updates = tf.ones(tf.shape(row_idx), dtype=tf.float32)
    e_u.assign(tf.tensor_scatter_nd_update(e_u, indices, updates))        
    return compute_(n, K, C, W, q, topK, e_u)    

if __name__ == "__main__":
    pass
    
