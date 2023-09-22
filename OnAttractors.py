import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits import mplot3d
import os
from scipy.signal import convolve2d as conv2
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import utils
import systems

name='rossler'
fi = lambda x,t: systems.rossler(t,x)
Ji = lambda x: systems.Jrossler(x)
cfun = lambda e: np.max(np.abs(e),axis=1)
a0 = np.array([-1,0,1]) #chua
a0 = np.array([-10,0,41]) #ros
#a0 = np.array([-1,1,1]) #rf
#a0 = np.array([45,45,40]) #lor
d = 3
figsize=[8,6]

transient_count = 5
dt = 0.005
T = 100

def maine(a0=a0, show=True, save=True):
    if not os.path.exists(name):
        os.mkdir(name)
    t = np.arange(0, T+transient_count, dt)
    xi = odeint(fi, a0, t)
    xi = xi[int(transient_count//dt):,:] #remove transient
    e = np.zeros((len(xi),d-1),dtype=np.complex128)
    v = np.zeros((len(xi),d-1,d-1),dtype=np.complex128)
    rotation_manager = utils.rot_keeper(d=d)

    for i,x in enumerate(xi):
        J=np.array(Ji(x))
        ds = fi(x,0)
        R=rotation_manager.get_rot(ds)
        G = (R@J)
        e[i,:] = np.linalg.det(G[1:,1:])

    c = cfun(e)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(2,2,1,projection='3d')
    p = ax.scatter3D(xi[:,0], xi[:,1], xi[:,2], c=c)
    plt.title('Perp J')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    ax = plt.subplot(2,2,2,projection='3d')
    p = ax.scatter3D(xi[:,0], xi[:,1], xi[:,2], c=np.log(c))
    plt.title('Perp J, Log')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #now the full Jacobian
    #a0 = np.array([45,45,40]) #lor
    xi = odeint(fi, a0, t)
    xi = xi[int(transient_count//dt):,:] #remove transient
    e = np.zeros((len(xi),d),dtype=np.complex128)
    v = np.zeros((len(xi),d-1,d-1),dtype=np.complex128)
    rotation_manager = utils.rot_keeper(d=d)

    for i,x in enumerate(xi):
        J=np.array(Ji(x))
        e[i,0] = np.linalg.det(J)

    c = cfun(e)
    ax = plt.subplot(2,2,3,projection='3d')
    p = ax.scatter3D(xi[:,0], xi[:,1], xi[:,2], c=c)
    plt.title('Full J')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax = plt.subplot(2,2,4,projection='3d')
    p = ax.scatter3D(xi[:,0], xi[:,1], xi[:,2], c=np.max(np.log(np.abs(e)),axis=1))
    plt.title('Full J, Log')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(name,'figs.png'))

    if show:
        plt.show()


if __name__ == '__main__':
    # Initialize the array with random amounts of A, B and C.
    maine()
