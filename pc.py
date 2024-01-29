import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import decomposition as skd
from itertools import product


d = 128
xs = 300
ys = 350
vd = cv2.VideoCapture('pw_08_03_2023_03_13_02_077888.avi')
fc = int(vd.get(cv2.CAP_PROP_FRAME_COUNT))
dat = np.zeros((fc,d,d,1))
for q in range(fc):
    ret,frame = vd.read()
    dat[q,:,:,0] = frame[xs:(xs+d),ys:(ys+d),1]


def pca_reduce(vid, reduce=1, chipsize=8,n_components=8):
   #vid should be [framenum, x, y, c]
   x = vid.shape[1]
   y = vid.shape[2]
   c = chipsize
   if not vid.shape[1]%chipsize == 0 or\
      not vid.shape[2]%chipsize == 0:
      print('get a better chipsize!')
      return 0
   grid = product(range(0, x-x%c, c), range(0, y-y%c, c))
   tc = (x//c)*(y//c)
   chips = np.zeros((vid.shape[0]*tc,c*c*vid.shape[3]))
   #chips = np.zeros((vid.shape[0]*tc//reduce,c*c*vid.shape[3]))
   #for f in range(0,vid.shape[0],reduce):
   for f in range(0,vid.shape[0]):
      for n,ij in enumerate(grid):
         i,j = ij
         #chips[(f//reduce)*tc+n] = vid[f,i:(i+c),j:(j+c),:].flatten()
         chips[(f)*tc+n] = vid[f,i:(i+c),j:(j+c),:].flatten()
   pca = skd.PCA(n_components=n_components)
   if reduce > 1:
      pca.fit(chips[::reduce])
      chips = pca.transform(chips)
   else:
      chips = pca.fit_transform(chips)
   return chips


import pdb
import copy


vid=dat
reduce=10
chipsize=8
n_components=8
x = vid.shape[1]
y = vid.shape[2]
c = chipsize
if not vid.shape[1]%chipsize == 0 or\
    not vid.shape[2]%chipsize == 0:
    print('get a better chipsize!')
grid = product(range(0, x-x%c, c), range(0, y-y%c, c))
tc = (x//c)*(y//c)
chips = np.zeros((vid.shape[0]*tc,c*c*vid.shape[3]))
for f in range(vid.shape[0]):
    for n,ij in enumerate(grid):
        i,j = ij
        #chips[(f//reduce)*tc+n] = vid[f,i:(i+c),j:(j+c),:].flatten()
        chips[f*tc+n, :] = vid[f,i:(i+c),j:(j+c),:].flatten()
        #pdb.set_trace()
        #chips[f*tc+n,:] = copy.copy(vid[f,i:(i+c),j:(j+c),:].flatten())


plt.figure()
plt.imshow(chips[16*16*50+3,:].reshape([8,8]))
plt.imshow(chips[250,:].reshape([8,8]))
plt.colorbar()
plt.show()