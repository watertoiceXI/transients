import numpy as np
import utils
from matplotlib import pyplot as plt
import copy
from itertools import product
from sklearn import decomposition as skd


def normaleq(X,Y):
    try:
        return np.linalg.inv(X.T@X)@(X.T@Y)
    except np.linalg.LinAlgError:
        return np.zeros([X.shape[1],Y.shape[1]])

class PCA_pyramid():
   def __init__(self,reduces=[],chipsizes=[],n_components=[]):
      self.reduces = reduces
      self.chipsizes = chipsizes
      self.n_components = n_components
      assert(len(reduces)==len(chipsizes))
      #assert(len(reduces)==len(n_components))
      self.n_stages = len(reduces)
      if len(self.n_components) == 0:
         self.n_components = [0,]*len(reduces)
      self.n_components.insert(0,1) #this makes thecode for reverse so much cleaner
   def fit_transform(self,vid):
      self.pcas = []
      level = 0
      for n, (reduce, chipsize, n_component) in enumerate(zip(self.reduces, self.chipsizes, self.n_components[1:])):
         print(f'Starting level {level}')
         #print([chipsize,n_component,reduce])
         vid,pcap = pca_reduce(vid,chipsize=chipsize,n_components=n_component,reduce=reduce)
         if not n_component:
            self.n_components[n+1] = pcap.components_.shape[0]
            print(self.n_components[n+1])
         self.pcas.append(copy.copy(pcap))
         level+=1
      return vid.squeeze()
   def reverse(self,state_vec):
      assert(len(state_vec.shape)==2)
      state_vec = np.expand_dims(state_vec,axis=[1,2])
      quant = state_vec.shape[0]
      current_size = np.array([1,1])
      for pca,n_component,chipsize in zip(self.pcas[-1::-1],self.n_components[-2::-1],self.chipsizes[-1::-1]):
         state_vec = state_vec@pca.components_
         current_size = current_size*chipsize
         rvid = np.zeros([quant,current_size[0],current_size[1],n_component])
         grid = product(range(0, current_size[0], chipsize), range(0, current_size[1], chipsize))
         for n,ij in enumerate(grid):
            i,j = ij 
            rvid[:,i:(i+chipsize),j:(j+chipsize)]=state_vec[:,i//chipsize,j//chipsize,:].reshape([-1,chipsize,chipsize,n_component])
         state_vec = copy.copy(rvid)
      return state_vec
         





def pca_reduce(vid, reduce=1, chipsize=8,n_components=8):
   #vid should be [framenum, x, y, c]
   x = vid.shape[1]
   y = vid.shape[2]
   c = chipsize
   if not vid.shape[1]%chipsize == 0 or\
      not vid.shape[2]%chipsize == 0:
      print('get a better chipsize!')
      return 0
   tc = (x//c)*(y//c)
   chips = np.zeros((vid.shape[0]*tc,c*c*vid.shape[3]))
   #chips = np.zeros((vid.shape[0]*tc//reduce,c*c*vid.shape[3]))
   #for f in range(0,vid.shape[0],reduce):
   for f in range(0,vid.shape[0]):
      grid = product(range(0, x-x%c, c), range(0, y-y%c, c))
      for n,ij in enumerate(grid):
         i,j = ij
         #chips[(f//reduce)*tc+n] = vid[f,i:(i+c),j:(j+c),:].flatten()
         chips[(f)*tc+n] = vid[f,i:(i+c),j:(j+c),:].flatten()
   if n_components:
      pca = skd.PCA(n_components=n_components)
   else:
      pca = skd.PCA(n_components='mle')
   if reduce > 1:
      pca.fit(chips[::reduce])
      chips = pca.transform(chips)
   else:
      chips = pca.fit_transform(chips)
   return chips.reshape((vid.shape[0],x//c,y//c,-1)), pca
   


def local_linear(points,l_frame=5,verify=False,rot=True,nearest=False):
    #points should be [time,instance,dim]
    rot_man = utils.rot_keeper(d=points.shape[-1])

    frame_centers = points[l_frame//2::l_frame,0,:]
    if l_frame <= 2:
        dss = -points[0:-1:l_frame,0,:]+points[1::l_frame,0,:]
    else:
        dss = -points[l_frame//2:-1:l_frame,0,:]+points[l_frame//2+1::l_frame,0,:]
    numframes = len(frame_centers)-1
    if nearest:
        frame_assignment = np.argmin(np.sum((np.expand_dims(np.expand_dims(frame_centers,1),1)-np.expand_dims(points[:-1,1:,:],0))**2,axis=3),axis=0)
    else:
        #print(np.linspace(0,points.shape[1],numframes))
        frame_assignment = np.expand_dims(np.arange(len(frame_centers)),axis=0).repeat(points.shape[1],axis=0).T
    As = []
    for q in range(numframes):
        times,instances = np.where(frame_assignment==q)
        R = rot_man.get_rot(dss[q])
        these_pnts = points[times,instances,:] - frame_centers[q]
        if rot: these_pnts = (R @ these_pnts.T).T
        these_dpnts = points[times+1,instances,:]-points[times,instances,:]
        if rot: these_dpnts = (R @ these_dpnts.T).T
        #print(len(times))
        A = normaleq(these_pnts,these_dpnts)
        As.append(copy.copy(A))
        v,e = np.linalg.eig(A)
        if verify:
            plt.figure()
            plt.scatter(np.real(v),np.imag(v))
            plt.show()
            n = 9 if len(these_pnts) > 9 else -1
            print(f"percent error {100*(these_dpnts[n,:]-these_pnts[n,:]@A)/these_dpnts[n,:]}")
            print(f"this {these_pnts[n,:]@A}")
            print(f"orig {these_dpnts[n,:]}")
            plt.figure()
            plt.scatter(these_pnts[:,1],these_pnts[:,2],c=utils.normalizer(these_dpnts[:,:3]))
            plt.show()
    return As, frame_assignment
    


if __name__ == '__main__':
    
    #testing normaleq
    samp = 100
    d = 5
    X = np.random.randn(samp,d)
    A = np.random.randn(5,5)
    Y = (X@A)
    normaleq = lambda X,Y: np.linalg.inv(X.T@X)@(X.T@Y)
    n = 2
    Ae = normaleq(X,Y)
    Y[n]-X[n]@Ae

    #test for local_linear
