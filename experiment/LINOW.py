'''
Created on August 08, 2024

@NOTE: LINOW Implementation (matrix form, linear matrix form, LINOW_LMF-sn, and LINOW_LMF-bn)   

'''
import numpy as np
import math
import time
import tensorflow as tf
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix, identity, csr_array
from sklearn.preprocessing import normalize
from concurrent.futures import ProcessPoolExecutor

def LINOW_MF (graph='', iterations=0, damping_factor=0.8):
    '''
        This is the implementation of the LINOW matrix form

        @param graph: the graph dataset
        @param iterations: number of iteration
        @param damping_factor: C
        @return: |V|*|V| matrix, each row i contains similarity scores for node i         
    '''
    print("Starting similarity computation on dataset '{}' with itrations={} and C={}...".format(graph,iterations,damping_factor))
    
    #============================================================================================
        # reading graph, computing weights matrix
    #============================================================================================    
    edges = np.loadtxt(graph, dtype=int)
    num_nodes = np.max(edges) + 1
    adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes), dtype='int32') 
    print('Adjacency matrix is constructed ...')    
    print("Number of nodes:",num_nodes)    
    degrees = adj.sum(axis=0).T     
    weights = csr_matrix(1/np.log(degrees+math.e))  
    weight_matrix = csr_matrix(adj.multiply(weights)).astype(dtype='float32',casting='unsafe')         
    del degrees
    del weights
    result_matrix = identity(weight_matrix.shape[0],dtype='float32', format='csr')
    weight_matrix = normalize(weight_matrix, norm='l1', axis=0) 
    for itr in range (1, iterations+1):
        print("Iteration "+str(itr)+' ...')
        temp = result_matrix * weight_matrix
        result_matrix =  damping_factor/2.0 * (temp + temp.T) 
        result_matrix.setdiag(1) ## disjunction operator
        
    # print(result_matrix.todense())
    print("LINOW Matrix Form: similarity computation is completed ... \n")    
    return result_matrix



def LINOW_LMF (graph='', iterations=0, damping_factor=0.8):
    '''
        This is the implementation of the LINOW linear matrix form

        @param graph: the graph dataset
        @param iterations: number of iteration
        @param damping_factor: C
        @return: |V|*|V| matrix, each row i contains similarity scores for node i         
    '''
    print("Starting similarity computation on dataset '{}' with itrations={} and C={}...".format(graph,iterations,damping_factor))
    
    #============================================================================================
        # reading graph, computing weights matrix
    #============================================================================================    
    edges = np.loadtxt(graph, dtype=int)
    num_nodes = np.max(edges) + 1
    adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes), dtype='int32') 
    print('Adjacency matrix is constructed ...')    
    print("Number of nodes:",num_nodes)    
    degrees = adj.sum(axis=0).T     
    weights = csr_matrix(1/np.log(degrees+math.e))  
    weight_matrix = csr_matrix(adj.multiply(weights)).astype(dtype='float32',casting='unsafe')         
    del degrees
    del weights
    iden_matrix = identity(weight_matrix.shape[0],dtype='float32', format='csr')
    iden_matrix.setdiag(1-damping_factor)
    result_matrix = iden_matrix
    weight_matrix = normalize(weight_matrix, norm='l1', axis=0) 
    for itr in range (1, iterations+1):
        print("Iteration "+str(itr)+' ...')
        temp = result_matrix * weight_matrix
        result_matrix =  damping_factor/2.0 * (temp + temp.T) + iden_matrix
    # print(result_matrix.todense())
    print("LINOW Linear Matrix Form: similarity computation is completed ... \n")    
    return result_matrix

def compute_LINOW_sn (graph='', iterations=0, damping_factor=0.8, target_nodes = []):
    '''
        Compute similarity w.r.t a single node        
        @param graph: the graph dataset
        @param iterations: number of iteration
        @param damping_factor: C
        @param target_nodes: the list of target nodes
        @return: len(target_nodes)*|V| matrix, each row i contains similarity scores for target_node[i] 
    '''
    print("Starting single source computation on dataset '{}' with itrations={} and C={}...".format(graph,iterations,damping_factor))    
    edges = np.loadtxt(graph, dtype=int)
    num_nodes = np.max(edges) + 1
    adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes), dtype='int32')
    print('Adjacency matrix is constructed ...')
    print("Number of nodes:", num_nodes)
    degrees = adj.sum(axis=0).T 
    weights = csr_matrix(1 / np.log(degrees + math.e))
    weight_matrix = csr_matrix(adj.multiply(weights)).astype(dtype='float32', casting='unsafe')
    del degrees
    del weights
    weight_matrix = normalize(weight_matrix, norm='l1', axis=0)  
    result_array = []
    for target_node in target_nodes:
        print('Compuation for node',target_node)
        similarity_array = LINOW_sn(num_nodes, iterations, damping_factor, weight_matrix, [target_node])
        result_array.append(similarity_array.T)
    result_matrix = np.matrix(result_array, dtype=np.float32)
    print(result_matrix)
    print("LINOW_LMF-SN: similarity computation is completed ... \n")        
    return result_matrix

def compute_LINOW_bn (graph='', iterations=0, damping_factor=0.8, bch_size=16, num_of_pr=1, GPU=False):
    '''
       Compute similarity w.r.t a batch of nodes
        
        @param graph: the graph dataset with size |V|*|V|
        @param iterations: number of iteration
        @param damping_factor: C
        @param bch_size: batch size
        @param num_of_pr: number of parallel computation  
        @param GPU: flag for using GPU   
    '''    
    
    print("Batch similarity computation --> dataset:'{}', iterations:{}, C:{}, batch:{}, {}, {}".format(graph,iterations,damping_factor,bch_size,('#processes: '+str(num_of_pr) if not GPU else 'single process'),('GPU' if GPU else 'CPU')))    
    edges = np.loadtxt(graph, dtype=int)
    num_nodes = np.max(edges) + 1
    adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes), dtype='int32')
    print('Adjacency matrix is constructed ...')
    print("Number of nodes:", num_nodes)
    degrees = adj.sum(axis=0).T 
    
    weights = csr_matrix(1 / np.log(degrees + math.e)) 
    weight_matrix = csr_matrix(adj.multiply(weights)).astype(dtype='float32', casting='unsafe')     
    del degrees
    del weights
    weight_matrix = normalize(weight_matrix, norm='l1', axis=0) 
    weight_matrix = (damping_factor / 2) * weight_matrix 
    batches = [range(i, min(i + bch_size, num_nodes)) for i in range(0, num_nodes, bch_size)]        
    
    if GPU: ## GPU computation
        rows, cols = csr_array.nonzero(weight_matrix)
        indices = list(zip(rows, cols))                           
        weight_matrix_tensor= tf.sparse.SparseTensor(indices=indices, values=weight_matrix.data, dense_shape=[num_nodes,num_nodes])
        for idx in tqdm(range(0, len(batches))):
            # start_time = time.time()                            
            result_matrix = LINOW_bn_tensor_based(tf.constant(num_nodes, dtype=tf.int32), tf.constant(iterations, dtype=tf.int32),tf.constant(damping_factor, dtype=tf.float32), weight_matrix_tensor , tf.constant(batches[idx],dtype=tf.int32))
            yield result_matrix.T                 
            # print("\nTIME GPU: "+str(round((time.time() - start_time)/60,3))+'\n')            
            
    else: ## CPU computation with multi-process
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ## disabling GPU  
        for idx in tqdm(range(0, len(batches), num_of_pr)):
            # start_time = time.time()                
            with ProcessPoolExecutor(max_workers=num_of_pr) as executor:
                futures = [executor.submit(LINOW_bn_pythonic, num_nodes, iterations, damping_factor, weight_matrix, batch) for batch in batches[idx:idx + num_of_pr]]
                for i, future in enumerate(futures):
                    result_matrix = future.result().T                        
                    yield result_matrix
            # print("\nTIME CPU/multi-process: "+str(round((time.time() - start_time)/60,3))+'\n')      
    print("LINOW_LMF-BN: similarity computation is completed ... \n")        
       
                       
def LINOW_sn(n, K, C, W, q):
    """
        Pythonic single node computation.
        
        @param n: number of nodes
        @param K: number of iterations
        @param C: damping factor
        @param W: column normalized weight matrix
        @param q: query node index
        @return: query node's similarity array
    """
    e_u = np.zeros(n, dtype=np.float32) 
    e_u[q] = 1
    zero = np.zeros(n, dtype=np.float32) 
    gamma = np.full((K + 2, K + 1), None, dtype=np.ndarray)     
    for i in range(K + 1):
        gamma[i, i] = zero
    gamma[K + 1, K] = e_u    
    W_buf = (C / 2) * W

    for i in range(1, K + 2):
        for j in range(i):
            if j == 0:
                gamma[i, j] = W_buf @ gamma[i - 1, j] + e_u
            else:
                gamma[i, j] = W_buf @ gamma[i - 1, j] + gamma[i - 1, j - 1]
                gamma[i - 1, j - 1] = None 

    W_T_buf = (C / 2) * W.T 
    Gamma = e_u
    for i in range(1, K + 1):
        Gamma = gamma[K + 1, K - i] + W_T_buf @ Gamma
    return (1 - C) * Gamma

def LINOW_bn_pythonic(n, K, C, W, q):
    """
        Pythonic implementation for batch similarity computation.

        @param n: number of nodes
        @param K: number of iterations
        @param C: damping factor
        @param W: column normalized weight matrix
        @param q: range of query nodes
    """
    
    e_u = np.zeros((n, len(q)),  dtype=np.float32) 
    for i, query_node in enumerate(q):
        e_u[query_node, i] = 1
    zero = np.zeros((n, len(q)),  dtype=np.float32)  
      
    gamma = np.full((K + 2, K + 1), None, dtype=np.ndarray)
    for i in range(1, K + 2):
        gamma[i - 1, i - 1] = zero
        gamma[i, 0] = W.dot(gamma[i - 1, 0]) + e_u
        for j in range(1, i):
            gamma[i, j] = W.dot(gamma[i - 1, j]) + gamma[i - 1, j - 1]            
            gamma[i - 1, j - 1] = None

    Gamma = e_u
    for i in range(1, K + 1):
        Gamma = gamma[K + 1, K - i] + W.T.dot(Gamma)     
    return (1 - C) * Gamma


def LINOW_bn_tensor_based(n, K, C, W, q):
    """
        Tensor-based implementation for batch similarity computation.

        @param n: number of nodes
        @param K: number of iterations
        @param C: damping factor
        @param W: column normalized weight matrix
        @param q: range of query nodes
    """
    def compute_(n, K, C, W, q, e_u):        
        zero = tf.zeros((n, q.shape[0]), dtype=tf.float32)    
        '''    
            infer_shape=False: entries can have different shape;
            clear_after_read = False: entries are kept after reading            
        '''
        gamma = tf.TensorArray(dtype=tf.float32, size=(K+2)*(K+1), infer_shape=False, dynamic_size=True,clear_after_read = False)  
        for i in tf.range(1, K + 2):
            gamma = gamma.write((i-1)*(K+1)+(i-1), zero)
            gamma = gamma.write( i*(K+1), tf.sparse.sparse_dense_matmul(W,gamma.read((i-1)*(K+1)))+e_u)
            for j in tf.range(1, i):
                gamma = gamma.write(i*(K+1)+j, tf.sparse.sparse_dense_matmul(W,gamma.read((i-1)*(K+1)+j)) +gamma.read((i-1)*(K+1)+(j-1)))
                gamma = gamma.write((i-1)*(K+1)+(j-1), 0.)  
        Gamma = e_u
        for i in tf.range(1, K + 1):
            Gamma = gamma.read((K+1)*(K+1)+(K-i)) + tf.sparse.sparse_dense_matmul(tf.sparse.transpose(W),Gamma)            
        return (tf.math.scalar_mul(1-C,Gamma)).numpy() 

    e_u = tf.Variable(tf.zeros([n, tf.shape(q)[0]]), dtype=np.float32)   
    row_idx = q
    col_idx = tf.range(tf.shape(q)[0])
    indices = tf.stack([row_idx, col_idx], axis=1)
    updates = tf.ones(tf.shape(row_idx), dtype=tf.float32)
    e_u.assign(tf.tensor_scatter_nd_update(e_u, indices, updates))        
        
    return compute_(n, K, C, W, q, e_u)    

if __name__ == "__main__":
    
    ## matrix multiplication
    # LINOW_LMF(graph="data/Cora_directed_graph.txt",
    #            iterations=5,
    #            damping_factor = 0.2
    #        )
    
    ## single node calculation
    # compute_LINOW_sn(graph='data/Cora_directed_graph.txt',                     
    #                  iterations=5,
    #                  damping_factor = 0.2,
    #                  target_nodes=[0],
    #                  )
    #

    
    ## batch CPU multi-process/GPU calculation
    for thread_result in compute_LINOW_bn(graph='data/Cora_directed_graph.txt',
                      iterations=5,
                      damping_factor = 0.2,
                      bch_size=128,
                      num_of_pr=6,
                      GPU=True
                      ):
        thread_result ## return result of each batch as completed
        # print(thread_result)
    

    
