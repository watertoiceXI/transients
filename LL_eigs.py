import numpy as np
import cv2
import matplotlib.pyplot as plt
import local_linear
import scipy.spatial as spatial
import glob
import copy
import imageio
import os 
import pickle

starts = {'25':[1187,3119,4684,6591,8931],'76':[1249,3230,4943,6587,8487],'89':[855,2393,3914,5456,6990,8508],'57':[3043]}
loc = r'data/'

def load_data(lenny=200,wid=0):
    dater = []
    for k in starts.keys():
        for start in starts[k]:
            data = np.zeros((lenny,1,480,640))
            for q in range(lenny):
                data[q]=imageio.imread(os.path.join(loc,k,f'frames{q+start:07}.png'))[-480:,:,1]
            dater.append(copy.copy(data))
    dat = np.stack([d[:200+(wid*2):2,0:,100:356,150:406] for d in dater],axis=1)
    if wid:
        sdat = np.zeros_like(dat)
        for q in range(len(dat)-wid):
            sdat[q] = np.mean(dat[q:(q+wid)],axis=0)
        dat = sdat
    dat = dat.reshape([17*(100),256,256,1])
    return dat

def pca_embed(dataname='LC1',chipsizes=[8,8,4],n_components=[12,64,0],reduces = [10,5,1]):
    a = "".join([str(i) for i in chipsizes+n_components+reduces])
    fn =  loc+dataname+a+'.pkl'
    if os.path.exists(fn):
        with open(fn,'rb') as f:
            dat = pickle.load(f)
            return dat,None
    print("loading")
    dat = load_data()
    pcap = local_linear.PCA_pyramid(reduces=reduces,chipsizes=chipsizes,n_components=n_components)
    dat = pcap.fit_transform(dat)
    with open(fn,'wb') as f:
        pickle.dump(dat,f)
    return dat,pcap


if __name__=='__main__':
    print('embed')
    dat,_ = pca_embed()
    dats = dat.reshape([100,17,-1])
    filt = np.ones(3)
    sdats = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), axis=0, arr=dats)
    As,frame_ass = local_linear.local_linear(dats,2,nearest=False)
    ne = []
    mv = []
    ee = []
    nvec = 32
    for A in As:
        e,v = np.linalg.eig(A[1:nvec+1,1:nvec+1])
        ne.append(len(np.where(np.real(e)>0)[0]))
        ee.append(e)
        mv.append(v)
        #plt.figure()
        #plt.plot(np.sort(np.real(e)))
        #plt.title(f'{len(np.where(np.real(e)>0)[0])} Positive')
        #plt.show()
    #plt.figure()
    #plt.plot(ne)
    #plt.show()

    thrs = [.1,.2,.3,.4]
    nveccs = [10,16,24,32]
    for thr in thrs:
        print(thr)
        for nvecc in nveccs:
            ve = []
            vv = []
            print(nvecc)
            for q in range(24):
                ve.append([])
                vv.append([])
                for m in range(nvecc):
                    for n in range(nvecc):
                        if mv[q][m].dot(mv[q+1][n]) > thr and np.abs(ee[q][m]) < 100:
                            #print(f"X",end='')
                            vv[-1].append(mv[q][m])
                            ve[-1].append(ee[q][m])
                            break
                    #print(f"O",end='')
                #print("\n")
                
            n = 12
            m = 1000
            sse = [np.mean(np.maximum(np.minimum(m,np.real(e)),-m)) for e in ve[:n]]
            ssee = [np.std(np.maximum(np.minimum(m,np.real(e)),-m))/np.sqrt(17) for e in ve[:n]]
            #loading LC data
            with open(r'C:\Users\water\Code\transients\data\LC_power_data.pkl','rb') as f:
                lcpow=pickle.load(f)
            
            fig,ax = plt.subplots()
            ax.errorbar(np.arange(0,n*8,8)*1/30,sse,yerr=ssee,label='Mean Eigenvalue')
            ax2 = ax.twinx()
            ax2.plot(lcpow[0][0][:50],lcpow[1][0][:50],'r',label="Normalized Current")
            ax.set_ylabel('Mean Eigenvalue')
            ax2.set_ylabel('Normalized Current')
            plt.xlabel('Time (sec)')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1+h2, l1+l2)
            plt.tight_layout()
            plt.show()