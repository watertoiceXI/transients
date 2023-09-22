import numpy as np
import torch 


def rk_iter_nophi(dt,f,states):
    for i in range(len(states)):
        state = states[i]
        k1=dt*f(state)
        k2=dt*f(state+k1/2)
        k3=dt*f(state+k2/2)
        k4=dt*f(state+k3)
        states[i]=state+k1/6+k2/3+k3/3+k4/6
    return states
def step(states,f,dt,substeps=1):
    oldstates = np.copy(states)
    for j in range(substeps):
        states = rk_iter_nophi(dt,f,states)
    return states, oldstates[0]-states[0]

def normalizer(x):
    x-=np.min(x)
    x/=np.max(x)
    return x
def normalize(x,axis=None):
    x /= np.sqrt(np.sum(x**2,axis=axis,keepdims=True))
    return x
def normalize_vec(x):
    x /= np.sqrt(np.sum(x**2,keepdims=True))
    return x
def normalize_vec_batch(x):
    x /= torch.sqrt(torch.sum(torch.square(x),axis=1,keepdims=True))
    if len(x.shape)<3: #important!
        x=torch.unsqueeze(x,axis=2)
    return x
def rot_mat(u,v, normalized=False,check=True):
    if not normalized:
        u = normalize_vec(u)
        v = normalize_vec(v)
    r = u+v #needed for the below to work properly
    if np.sum(np.abs(r))<10e-6:
        #print('oopposites!')
        return -np.eye(len(r))
    if len(r.shape)<2:
        r=np.expand_dims(r,axis=1)
    R = 2*(r@r.T)/(r.T@r) - np.eye(len(r))
    if check:
        try:
            assert np.sum(R@R.T-np.eye(R.shape[1])) < 10**-4
        except AssertionError:
            print('rot mat error with inputs (u,v,R)')
            print([u,v,R])
    return R
def batch_eye(nbatch,s):
    x = torch.eye(s)
    x = x.reshape((1, s, s))
    x = x.repeat(nbatch, 1, 1)
    return x.float() 
def rot_mat_batch(u,v, normalized=False, check=True):
    if not normalized:
        u = normalize_vec_batch(u)
        v = normalize_vec_batch(v)
    r = u+v
    R = 2*torch.bmm(r,r.mT)/torch.bmm(r.mT,r) - batch_eye(len(r),r.shape[1])
    if check:
        try:
            assert torch.sum(torch.bmm(R,R.mT)-batch_eye(len(r),R.shape[1]))/len(r) < 10**-4 #sufficent for float. double needs a higher bar
        except AssertionError:
            print([u,v,R])


    return R
def ndPolar(coords):
    d = len(coords)
    out = np.zeros_like(coords)
    out[0]=np.sqrt(np.sum(coords**2))
    for q in range(d-1):
        out[q+1]=np.arctan2(coords[q],coords[q+1])
    return out

class rot_keeper(object):
    def __init__(self,d=3):
        assert(d>2)
        self.d=d
        self.e1 = np.zeros(d)
        self.e1[0]=1.0
        self.e1_sm = np.zeros(d-1)
        self.e1_sm[0]=1.0
        self.e2 = np.zeros(d)
        self.e2[1]=1.0
        self.R = np.eye(d)

    def get_rot(self,u):
        Rs = rot_mat(u,self.e1)
        e2old = self.R.T@self.e2
        Rpp = np.eye(self.d)
        Rpp[1:,1:] = rot_mat((Rs@e2old)[1:],self.e1_sm)
        Rs = Rpp @ Rs
        self.R = Rs
        return Rs

def perp_plane(pnt,rot,delta=0.1,N=10):
    N = N//2
    g=np.stack(np.meshgrid(0,np.arange(-N,N),np.arange(-N,N))) #yes, the order is really, really weird. Thanks numpy
    q = np.einsum('ij,jlkm->ilkm',rot.T,g)[:,:,0,:] #transposed because we are going from g to f, squeezing
    q = delta*q+np.expand_dims(np.expand_dims(pnt,axis=1),axis=1) #dumping the flatness, shifting to the main point
    return q
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

from matplotlib import pyplot as plt
def plt5(x):
    plt.figure(figsize=[11,11])
    plt.subplot(4,1,1)
    if len(x.shape)==2:
        plt.scatter(x[:,0],x[:,1],c='r')
        plt.ylabel('y')
        plt.subplot(4,1,2)
        plt.scatter(x[:,0],x[:,2],c='r')
        plt.ylabel('z')
        plt.subplot(4,1,3)
        plt.scatter(x[:,0],x[:,3],c='r')
        plt.ylabel('a')
        plt.subplot(4,1,4)
        plt.scatter(x[:,0],x[:,4],c='r')
    else:
        for n in range(x.shape[1]):
            plt.scatter(x[:,n,0],x[:,n,1])
        plt.ylabel('y')
        plt.subplot(4,1,2)
        for n in range(x.shape[1]):
            plt.scatter(x[:,n,0],x[:,n,2])
        plt.ylabel('z')
        plt.subplot(4,1,3)
        for n in range(x.shape[1]):
            plt.scatter(x[:,n,0],x[:,n,3])
        plt.ylabel('a')
        plt.subplot(4,1,4)
        for n in range(x.shape[1]):
            plt.scatter(x[:,n,0],x[:,n,4])
    plt.ylabel('b')
    plt.xlabel('x')
    plt.show()
def plt3(x):
    plt.figure(figsize=[11,11])
    plt.subplot(2,1,1)
    if len(x.shape)==2:
        plt.scatter(x[:,0],x[:,1],c='r')
        plt.ylabel('y')
        plt.subplot(2,1,2)
        plt.scatter(x[:,0],x[:,2],c='r')
    else:
        for n in range(x.shape[1]):
            plt.scatter(x[:,n,0],x[:,n,1])
        plt.ylabel('y')
        plt.subplot(2,1,2)
        for n in range(x.shape[1]):
            plt.scatter(x[:,n,0],x[:,n,2])
    plt.ylabel('z')
    plt.xlabel('x')
    plt.show()

def pltN(x):
    N = x.shape[-1]
    plt.figure(figsize=[11,11])
    if len(x.shape)==2:
        print(N)
        for q in range(N-1):
            plt.subplot(N-1,q+1,1)
            print(((N,q+1,1)))
            plt.scatter(x[:,0],x[:,q+1],c='r')
    else:
        for q in range(N-1):
            for n in range(x.shape[1]):
                plt.subplot(N,q+1,1)
                plt.scatter(x[:,n,0],x[:,n,q+1])
    plt.show()
