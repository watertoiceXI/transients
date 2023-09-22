import numpy as np
import torch
import utils

class dynamicalSystem(object):
    def __init__(self,params):
        self.params = params
    def f(state):
        return state
    def Df(phi,state):
        return phi
class Chua_SDE(torch.nn.Module):
    noise_type = 'additive'
    sde_type = 'ito'

    def __init__(self, brownian_size, batch_size, noise_amp=0.01, alpha = 13,beta = 28,m0 = -1.143,m1 = -0.714):
        super().__init__()
        
        self.state_size = 3
        self.batch_size = batch_size
        self.brownian_size = brownian_size
        self.mu = torch.nn.Linear(self.state_size, 
                                  self.state_size)
        self.sigma = torch.nn.Linear(self.state_size, 
                                     self.state_size * brownian_size)
        self.alpha = alpha#13#15.6
        self.beta = beta#28
        self.m0 = m0#-1.143
        self.m1 = m1#-0.714
        self.noise_amp = noise_amp

    # Drift
    def f(self, t, y):    
        h = self.m1*y[:,0]+0.5*(self.m0-self.m1)*(abs(y[:,0]+1)-abs(y[:,0]-1))
        dy = torch.empty((y.shape[0],3,))
        dy[:,0] = self.alpha*(y[:,1]-y[:,0]-h)
        dy[:,1] = y[:,0] - y[:,1] + y[:,2]
        dy[:,2] = -self.beta*y[:,1]
        return dy


    def J(self, t, y):
        h = self.m1*y[0]+0.5*(self.m0-self.m1)*(abs(y[0]+1)-abs(y[0]-1))
        dx = np.zeros((3,))
        dhdx = self.m1 + 0.5*(self.m0-self.m1)*(np.heaviside(y[0]+1,0.5) -np.heaviside(y[0]-1,0.5))
        J=[[self.alpha+dhdx,  self.alpha,     0],
           [1,                        -1,     1],
           [0,                -self.beta,     0]]
        return np.array(J)

    
    # Diffusion
    def g(self, t, y):
        return self.noise_amp*torch.ones(self.batch_size, 
                          self.state_size, 
                          self.brownian_size)
class Lorentz_SDE(torch.nn.Module):
    noise_type = 'additive'
    sde_type = 'ito'

    def __init__(self, brownian_size, batch_size, noise_amp=0.01, sigma = 10, rho = 28, beta = 8/3):
        super().__init__()
        
        self.state_size = 3
        self.batch_size = batch_size
        self.brownian_size = brownian_size
        self.mu = torch.nn.Linear(self.state_size, 
                                  self.state_size)
        self.lsigma = sigma#13#15.6
        self.beta = beta#28
        self.rho = rho#-1.143
        self.noise_amp = noise_amp

    # Drift
    def f(self, t, y):    
        dy = torch.empty((y.shape[0],3,))
        dy[:,0] = self.lsigma * (y[:,1] - y[:,0])
        dy[:,1] = y[:,0]*(self.rho - y[:,2]) - y[:,1]
        dy[:,2] = y[:,0] * y[:,1] - self.beta * y[:,2]
        return dy

    def J(self, t, y):
        J = np.empty((y.shape[0],3,3))
        J[:,0,:] = torch.tensor([-self.lsigma,    self.lsigma,            0])
        J[:,1,0] = self.rho - y[:,2]
        J[:,1,1] = -1
        J[:,1,2] = -y[:,0]
        J[:,2,0] = y[:,1]
        J[:,2,1] = y[:,0]
        J[:,2,2] = -self.beta
        return J
    
    # Diffusion
    def g(self, t, y):
        return self.noise_amp*torch.ones(self.batch_size, 
                          self.state_size, 
                          self.brownian_size)
class Lorentz_SDE_sigma(torch.nn.Module):
    noise_type = 'additive'
    sde_type = 'ito'

    def __init__(self, brownian_size, batch_size, noise_amp=0.01, sigma = 13, rho = 28, beta = 8/3):
        super().__init__()
        
        self.state_size = 3+4
        self.batch_size = batch_size
        self.brownian_size = brownian_size
        self.mu = torch.nn.Linear(self.state_size, 
                                  self.state_size)
        self.lsigma = sigma#13#15.6
        self.beta = beta
        self.rho = rho
        self.noise_amp = noise_amp
        
        self.e1 = torch.zeros((batch_size,3,),dtype=torch.float)
        self.e1[:,0]=1
        self.e1=torch.unsqueeze(self.e1,axis=2)
        self.e2 = torch.zeros((batch_size,3,),dtype=torch.float)
        self.e2[:,1]=1
        self.e2=torch.unsqueeze(self.e2,axis=2)
        self.e1_sm = torch.zeros((batch_size,2,),dtype=torch.float)
        self.e1_sm[:,0]=1
        self.e1_sm=torch.unsqueeze(self.e1_sm,axis=2)
        self.R = utils.batch_eye(batch_size,3)

    def get_rot(self,dy):
        Rs=utils.rot_mat_batch(dy,self.e1)
        e2old = torch.bmm(Rs,torch.bmm(self.R.mT,self.e2))#huh?
        Rpp = utils.batch_eye(self.batch_size,3)
        Rpp[:,1:,1:] = utils.rot_mat_batch(e2old[:,1:],self.e1_sm)
        Rs = torch.bmm(Rpp,Rs)
        self.R = Rs
        return
    # Drift
    def f(self, t, y):
        dy = torch.empty_like(y)
        dS = torch.empty((y.shape[0],3,3))

        dy[:,0] = self.lsigma * (y[:,1] - y[:,0])
        dy[:,1] = y[:,0]*(self.rho - y[:,2]) - y[:,1]
        dy[:,2] = y[:,0] * y[:,1] - self.beta * y[:,2]
        dyf = torch.unsqueeze(dy[:,:3],axis=2)
        self.get_rot(dyf)

        J = self.J(t,y)
        Sigma = y[:,3:].reshape([-1,2,2])
        J = torch.bmm(self.R.mT,torch.bmm(J,self.R))
        dS = torch.bmm(J[:,1:,1:],Sigma)#0.02*torch.bmm(J[:,1:,1:],torch.bmm(Sigma,J[:,1:,1:].mT)) #.mT is batch Transpose
        dy[:,3:] = dS.flatten()
        return dy.float()

    def J(self, t, y):
        J = torch.empty((y.shape[0],3,3),dtype=torch.float)
        J[:,0,:] = torch.tensor([-self.lsigma,    self.lsigma,            0])
        J[:,1,0] = self.rho - y[:,2]
        J[:,1,1] = -1
        J[:,1,2] = -y[:,0]
        J[:,2,0] = y[:,1]
        J[:,2,1] = y[:,0]
        J[:,2,2] = -self.beta
        return J
    # Diffusion
    def g(self, t, y):
        return self.noise_amp*torch.ones(self.batch_size, 
                          self.state_size, 
                          self.brownian_size)

def Chua(dynamicalSystem):
    def __init__(self,brownian_size, batch_size, noise_amp=0.01,alpha = 15.6,beta = 28,m0 = -1.143,m1 = -0.714):
        self.state_size = 3
        self.batch_size = batch_size 
        self.brownian_size = brownian_size
        self.alpha = alpha
        self.beta = beta
        self.m0=m0
        self.m1=m1
    def f(self,x):
        h = self.m1*x[0]+0.5*(self.m0-self.m1)*(abs(x[0]+1)-abs(x[0]-1))
        dx = np.zeros((3,))
        dx[0] = self.alpha*(x[1]-x[0]-h)
        dx[1] = x[0] - x[1] + x[2]
        dx[2] = -self.beta*x[1]
        return dx
    def Df(phi,x,alpha = 15.6,beta = 28,m0 = -1.143,m1 = -0.714):
        h = m1*x[0]+0.5*(m0-m1)*(abs(x[0]+1)-abs(x[0]-1))
        dx = np.zeros((3,))
        dhdx = m1 + 0.5*(m0-m1)*(np.heaviside(x[0]+1,0.5) -np.heaviside(x[0]-1,0.5))
        J=[[alpha+dhdx,  alpha,     0],
        [1,           -1,        1],
        [0,           -beta,     0]]
        return np.array(J)@phi


def rossler(t,x, a=0.1, b=0.1, c=14):
    dx = np.zeros((3,))
    dx[0] = -x[1] - x[2]
    dx[1] = x[0] + a*x[1]
    dx[2] = b + x[2]*(x[0]-c)
    return dx
def Jrossler(x, a=0.1, b=0.1, c=14):
    J=[[0,    -1,       -1],
       [1,     a,        0],
       [x[2],  0, (x[0]-c)]]
    return J

#https://en.wikipedia.org/wiki/Rabinovich%E2%80%93Fabrikant_equations
def rf(t,x, alpha=1.1, gamma=0.87):
    dx = np.zeros((3,))
    dx[0] = x[1]*(x[2]-1+x[0]**2) + gamma*x[0]
    dx[1] = x[0]*(3*x[2]+1-x[0]**2) + gamma*x[1]
    dx[2] = -2*x[2]*(alpha+x[0]*x[1])
    return dx
def Jrf(x, alpha=1.1, gamma=0.87):
    J=[[-x[1]*2*x[0] + gamma,    (x[2]-1+x[0]**2),      x[1]],
       [3*x[2]+1-3*x[0]**2,                 gamma,    3*x[0]],
       [-2*x[2]*x[1],                -2*x[2]*x[0],    -2*alpha+x[0]*x[1]]]
    return J

def lorentz(t,x, sigma = 10, rho = 28, beta = 8/3):
    dx = np.zeros((3,))
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = x[0]*(rho - x[2]) - x[1]
    dx[2] = x[0] * x[1] - beta * x[2]
    return dx
def Jlorentz(x, sigma = 10, rho = 28, beta = 8/3):
    J=[[-sigma,    sigma,       0],
       [rho - x[2],     -1,   -x[0]],
       [x[1],           x[0], -beta]]
    return J
def Dlorentz(phi,x, sigma = 10, rho = 28, beta = 8/3):
    J=[[-sigma,    sigma,       0],
       [rho - x[2],     -1,   -x[0]],
       [x[1],           x[0], -beta]]
    return J@phi

def dumb(t,x,a=.9):
    dx = np.zeros((3,))
    dx[0] = x[0]*a
    dx[1] = x[1]*a
    dx[2] = x[2]*a
    return dx
def Jdumb(x,a=.9):
    return np.eye(3)*a
def dumber(t,x,a=.9):
    dx = np.zeros((3,))
    dx[0] = a*x[0]**2
    dx[1] = x[1]*a
    dx[2] = x[2]*a
    return dx
def Jdumber(x,a=.9):
    return a*np.diag([x[0],1,1])
def pop_20_ode(A,t,K=0.99,x_c=0.4,y_c=2.009,R0=0.16129,C0=0.5,x_p=0.08,y_p=2.876):
    R,C,P = A
    Rdot = R*(1-R/K)-x_c*y_c*C*R/(R+R0)
    Cdot = x_c*C*(-1+y_c*R/(R+R0))-x_p*y_p*P*C/(C+C0)
    Pdot = x_p*P*(y_p*C/(C+C0)-1)
    return np.stack([Rdot,Cdot,Pdot])
def Jpop_20_ode(A,t,K=0.99,x_c=0.4,y_c=2.009,R0=0.16129,C0=0.5,x_p=0.08,y_p=2.876):
    R,C,P = A
    Rdotdot = [(1-2*R/K)-x_c*y_c*C/((R+R0)**2),
               -x_c*y_c*R/(R+R0),
               0]
    Cdotdot = [x_c*C*(y_c/((R+R0)**2)),
               x_c*(-1+y_c*R/(R+R0))-x_p*y_p*P/((C+C0)**2),
               -x_p*y_p*C/(C+C0)]
    Pdotdot = [0,
               x_p*P*(y_p/((C+C0)**2)),
               x_p*(y_p*C/(C+C0)-1)]
    return np.stack([Rdotdot,Cdotdot,Pdotdot])


#chua from http://www.chuacircuits.com/matlabsim.php
#chua from http://www.chuacircuits.com/matlabsim.php
c_alpha = 13#15.6
c_beta = 28
c_m0 = -1.143
c_m1 = -0.714

def chua_linear(x):
    if x[0]<-1:
        chua = np.array([[-c_alpha*(1+c_m1), c_alpha, 0],
                            [1       , -1     , 1],
                            [0       , -c_beta, 0]])
        chua_c = np.array([-c_alpha*(c_m1-c_m0), 0, 0])
    elif x[0]<1:
        chua = np.array([[-c_alpha*(1+(c_m0)), c_alpha, 0],
                         [1                  , -1     , 1],
                         [0                  , -c_beta, 0]])
        chua_c = np.array([0, 0, 0])
    else:
        chua = np.array([[-c_alpha*(1+c_m1), c_alpha, 0],
                        [1       , -1     , 1],
                        [0       , -c_beta, 0]])
        chua_c = np.array([c_alpha*(c_m1-c_m0), 0, 0])
    return chua, chua_c
def lp_chua(t,x):
    chua, chua_c = chua_linear(x)
    return chua@x+chua_c

def chua(t,x,alpha = 15.6,beta = 28,m0 = -1.143,m1 = -0.714):
    h = m1*x[0] + 0.5*(m0-m1) * (abs(x[0]+1)-abs(x[0]-1))
    dx = np.zeros((3,))
    dx[0] = alpha*(x[1]-x[0]-h)
    dx[1] = x[0] - x[1] + x[2]
    dx[2] = -beta*x[1]
    return dx
def Jchua(x,alpha = 15.6,beta = 28,m0 = -1.143,m1 = -0.714):
    h = m1*x[0]+0.5*(m0-m1)*(abs(x[0]+1)-abs(x[0]-1));
    dx = np.zeros((3,))
    dx[0] = alpha*(x[1]-x[0]-h)
    dx[1] = x[0] - x[1] + x[2]
    dx[2] = -beta*x[1]
    dhdx = m1 + 0.5*(m0-m1)*(np.heaviside(x[0]+1,0.5) -np.heaviside(x[0]-1,0.5))
    J=[[alpha+dhdx,alpha     ,0],
       [1,           -1,          1],
       [0,           -beta,     0]]

    return np.array(J)
def Dchua(phi,x,alpha = 15.6,beta = 28,m0 = -1.143,m1 = -0.714):
    h = m1*x[0]+0.5*(m0-m1)*(abs(x[0]+1)-abs(x[0]-1));
    dx = np.zeros((3,))
    dhdx = m1 + 0.5*(m0-m1)*(np.heaviside(x[0]+1,0.5) -np.heaviside(x[0]-1,0.5))
    J=[[alpha+dhdx,  alpha,     0],
       [1,           -1,        1],
       [0,           -beta,     0]]
    return np.array(J)@phi

