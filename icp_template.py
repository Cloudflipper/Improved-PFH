#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
###YOUR IMPORTS HERE###
from numpy.linalg import svd
import tqdm
###YOUR IMPORTS HERE###

def get_tranform(p,q):
    p_mean = numpy.mean(p,axis=0)
    q_mean = numpy.mean(q,axis=0)
    p_centered = p-p_mean
    q_centered = q-q_mean
    S = numpy.matmul(p_centered.reshape(-1,3).T,q_centered.reshape(-1,3))
    u,sigma,vt = svd(S)
    third = numpy.linalg.det(numpy.matmul(u,vt))
    R = numpy.matmul(vt.T,numpy.array([[1, 0, 0],[0, 1, 0],[0, 0, third]]))
    R = numpy.matmul(R,u.T)
    return R,q_mean.reshape(3,1)-numpy.matmul(R,p_mean.reshape(3,1))

def dist(p,q):
        return numpy.linalg.norm(p-numpy.array(q))

def get_error(p_set,q_set,R,t):
    error = 0
    for p,q in zip(p_set,q_set):
        error += numpy.linalg.norm(numpy.matmul(R,p.T).T+t-q)**2
    return error
        
def process_icp_cycle(pc_source,kdtree_target,patience = 20):
    epoch = 0
    errors=[]
    patient = 0
    min_error = numpy.inf
    while True:
        p_set = []
        q_set = []
        for item in pc_source:
            item_ = numpy.array(item)
            p_set.append(item_)
            distance, index = kdtree_target.query(item_.reshape(3))
            q_set.append(pc_source[index])
        R, t = get_tranform(p_set,q_set)
        epoch +=1
        
        error = get_error(p_set,q_set,R,t)
        errors.append(error)
        print("Epoch:",epoch,"Loss:",error,min_error)
        if error < min_error*0.999:
            patient=0
            min_error=error
        elif min_error>error:
            min_error=error
            patient+=1
        else:
            patient+=1
        if patient>=patience:
            break
        new_source = []
        for item in pc_source:
            item_ = numpy.matrix(numpy.matmul(R,numpy.array(item).T)+t)
            new_source.append(item_)
        pc_source = new_source
    return pc_source,error

def main():
    #Import the cloud
    pc_original=pc_source = utils.load_pc('source_cloud.csv')
    
    ###YOUR CODE HERE###
    pc_target = utils.load_pc('target_cloud.csv') # Change this to load in a different target
    kdtree = KDTree(numpy.array(pc_target).reshape(-1,3))
    
    #print(pc_source)
    pc_source,errors = process_icp_cycle(pc_original,kdtree)
    
    utils.view_pc([pc_source, pc_target, pc_original], None, ['b', 'r','g'], ['o', '^','o'])
    #plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
    ###YOUR CODE HERE###

    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1,len(errors)+1), errors, marker='o', label='loss_vs_epoch')
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title("target 3")
    plt.legend()
    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
