import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import animation

# Width, height of the image.
nx, ny = 150, 150
# Reaction parameters.
alpha, beta, gamma = 1, 1, 1
#alpha, beta, gamma = 1, 1, 1

def BZiter(arr):
    """Update arr[p] to arr[q] by evolving in time."""

    for p in [0,1]:
        # Count the average amount of each species in the 9 cells around each cell
        # by convolution with the 3x3 array m.
        q = (p+1) % 2
        s = np.zeros((3, ny,nx))
        m = np.ones((3,3)) / 9
        for k in range(3):
            s[k] = convolve2d(arr[p,k], m, mode='same', boundary='wrap')
        # Apply the reaction equations
        arr[q,0] = s[0] + s[0]*(alpha*s[1] - gamma*s[2])
        arr[q,1] = s[1] + s[1]*(beta*s[2] - alpha*s[0])
        arr[q,2] = s[2] + s[2]*(gamma*s[0] - beta*s[1])
        # Ensure the species concentrations are kept within [0,1].
        np.clip(arr[q], 0, 1, arr[q])
    return arr

def BZstep(arr):
    """Update arr[p] to arr[q] by evolving in time."""
    # Count the average amount of each species in the 9 cells around each cell
    # by convolution with the 3x3 array m.
    dif = np.zeros_like(arr)
    s = np.zeros((3, ny,nx))
    m = np.ones((3,3)) / 9
    for k in range(3):
        s[k] = convolve2d(arr[k], m, mode='same', boundary='wrap')
    # Apply the reaction equations
    dif[0] = s[0]*(alpha*s[1] - gamma*s[2])
    dif[1] = s[1]*(beta*s[2] - alpha*s[0])
    dif[2] = s[2]*(gamma*s[0] - beta*s[1])
    # Ensure the species concentrations are kept within [0,1].
    return dif

def run_B_Z(arr=None,t=200,nx=nx,ny=ny,show=False):
    if arr is None:
        arr = np.random.random(size=(2, 3, ny, nx))
    if show:
        # Set up the image
        fig, ax = plt.subplots()
        im = ax.imshow(arr[0,0], cmap=plt.cm.winter)
        ax.axis('off')

        def animate(i, arr):
            """Update the image for iteration i of the Matplotlib animation."""

            arr = BZiter(arr)
            im.set_array(arr[0,0])
            return [im]

        anim = animation.FuncAnimation(fig, animate, frames=t, interval=5,
                                    blit=False, fargs=(arr,))

        # To view the animation, uncomment this line
        plt.show()

        # To save the animation as an MP4 movie, uncomment this line
        #anim.save(filename='bz.mp4', fps=30)
    else:
        for dt in range(t):
            arr = BZiter(arr)
    return arr

def diff_B_Z(arr=None,t=200,nx=nx,ny=ny,show=False,percent = 0.01):
    if arr is None:
        arr = np.random.random(size=(2, 3, ny, nx))
    t_settle = 0
    for dt in range(t_settle):
        arr = BZiter(arr)
    arr1 = percent*np.random.random(size=(2, 3, ny, nx)) + (1-percent)*arr
    arr2 = percent*np.random.random(size=(2, 3, ny, nx)) + (1-percent)*arr

    if show:
        # Set up the image
        fig, ax = plt.subplots()
        im = ax.imshow(arr[0,0], cmap=plt.cm.winter)
        ax.axis('off')

        def animate(i, arr1, arr2):
            """Update the image for iteration i of the Matplotlib animation."""

            arr1 = BZiter(arr1)
            arr2 = BZiter(arr2)
            im.set_array(arr1[0,0]-arr2[0,0])
            return [im]

        anim = animation.FuncAnimation(fig, animate, frames=t, interval=5,
                                    blit=False, fargs=(arr1,arr2,))

        # To view the animation, uncomment this line
        plt.show()

        # To save the animation as an MP4 movie, uncomment this line
        #anim.save(filename='bz.mp4', fps=30)
    else:
        for dt in range(t):
            arr1 = BZiter(arr1)
            arr2 = BZiter(arr2)
    return arr1,arr2



if __name__ == '__main__':
    # Initialize the array with random amounts of A, B and C.
    diff_B_Z(show=True,t=2)
