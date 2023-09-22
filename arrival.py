import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import io
import imageio
from matplotlib import cm


import systems
import torch
import torchsde
import utils



def minor():

    xrange, yrange, zrange = ((-30,30),(-40,40),(0,70))
    cent = np.expand_dims(np.array([np.mean(xrange),np.mean(yrange),np.mean(zrange)]),axis=1)
    cent = np.expand_dims(np.array([0,0,35]),axis=1)

    abc = 2*np.expand_dims(np.array([30,40,35]),axis=1)
    pnts = utils.sample_spherical(1500,ndim=3)*abc + cent
    #main run
    brownian_size = 3
    batch_size = 1
    t_size = 501
    tf = .25
    sys = systems.Lorentz_SDE(brownian_size,batch_size,noise_amp=0.0)
    y0 = torch.tensor(pnts[:,0],dtype=torch.float)# interesting loop[ 1.35641339, -0.58946927, -2.27060091]
    y0=y0.repeat(batch_size,1)
    ts = torch.linspace(0, tf, t_size)
    # Initial state y0, the SDE is solved over the interval [ts[0], ts[-1]].
    # ys will have shape (t_size, batch_size, state_size)
    ys = torchsde.sdeint(sys, y0, ts)
    yy = np.array(ys.detach())

    #noisy
    batch_size = 5000
    init_spread = 0.01
    noise_amp = 0.0
    y0 = y0.repeat(batch_size,1)
    y0 += np.sqrt(init_spread)*torch.randn_like(y0)
    sys2 = systems.Lorentz_SDE(brownian_size,batch_size,noise_amp=noise_amp)
    ys = torchsde.sdeint(sys2, y0, ts)
    yo = np.array(ys.detach())



    d = 3
    q_bounds = [0,100]#[0,t_size-1]

    dt = float(ts[1])
    initial_sigma = init_spread
    #Sigma = initial_sigma #for the arrival
    Sigma = initial_sigma*np.eye(d) #for the plane

    es = []
    vs = []
    Js = []
    dts = []
    Rss = []
    stf = []
    print(Sigma)
    print(np.cov((yo[0,:,:]-yy[0,:,:]).T))
    rotation_manager = utils.rot_keeper(d=d)

    for q in range(q_bounds[0],q_bounds[1],1):
        print(f'frame: {q}', end='\r')
        f = np.array(sys.f(0,torch.tensor(yy[q,[0],:]))[0])
        nf = np.sqrt(np.sum(f**2))
        Rs=rotation_manager.get_rot(f)
        
        J = np.array(sys.J(0,torch.tensor(yy[[q],[0],:]))[0])
        print(J)
        Jr = Rs.T@J@Rs
        Rss.append(Rs)
        Js.append(Jr.copy())
        JRR = np.eye(d)+dt*Jr
        Sigma = JRR@Sigma@JRR.T + dt*noise_amp*np.eye(d)
    print(Sigma)


    tR = Rss[-1]
    dy = yo[q_bounds[1],:,:]-yy[q_bounds[1],:,:]
    dyR = tR@dy.T
    print(np.cov(dyR))

    plt.figure()
    ax = plt.subplot(3,1,1)
    nbins = 50
    counts, _ = np.histogram(dyR[0],nbins)
    ax.hist(dyR[0],nbins)
    xl,xh = ax.get_xbound()
    xe = np.linspace(xl,xh,100)
    ye = np.max(counts)*np.exp(-xe**2/(Sigma[0,0]**2))
    plt.plot(xe,ye)
    #ye = np.max(counts)*np.exp(-xe**2/(Sigma[1,1]**2))
    #plt.plot(xe,ye)
    #ye = np.max(counts)*np.exp(-xe**2/(Sigma[2,2]**2))
    #plt.plot(xe,ye)
    plt.legend(['Data','Predicted Gaussian','Predicted Gaussian 1','Predicted Gaussian 2'])
    plt.subplot(3,1,2)
    plt.title('Spread in Perpendicular Plane')
    plt.scatter(dyR[1],dyR[2],c='r',s=0.1)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.subplot(3,1,3)
    plt.title('Spread in Perpendicular Plane')
    plt.scatter(dyR[1],dyR[0],c='r',s=0.1)
    plt.ylabel('z')
    plt.xlabel('x')
    plt.tight_layout()
    plt.show()
    #utils.plt3(yo)

if __name__ == '__main__':
    # Initialize the array with random amounts of A, B and C.
    minor()
