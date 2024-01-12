import matplotlib.pyplot as plt
import numpy as np


class PolarPlotter:
    def plot(self, rArray: list, thetaArray: list, zArray: list, colormap: str = "viridis", zrange: tuple = (None, None)):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # check whether thetaArray is Chebyshev or Uniform grid
        if np.max(thetaArray) < 2:
            thetaArray *= np.pi
            
        # Create the mesh in polar coordinates and reshape Z
        R, P = np.meshgrid(rArray, thetaArray)
        Z = zArray.reshape((len(rArray), len(thetaArray))).T

        # Express the mesh in the cartesian system and reshape Z
        X, Y = R*np.cos(P), R*np.sin(P)

        # Plot the surface.
        fig.set_size_inches(7, 7)
        ax.plot_surface(X, Y, Z, cmap=colormap)
        ax.axes.set_zlim3d(bottom=zrange[0], top=zrange[1])
        plt.show()


class CartesianPlotter2D:
    def plot(self, xArray: list, yArray: list, zArray: list, colormap: str = "viridis"):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Create the mesh in polar coordinates and reshape corresponding Z.
        X, Y = np.meshgrid(xArray, yArray)
        Z = zArray.reshape((len(xArray), len(yArray))).T

        # Plot the surface.
        fig.set_size_inches(7, 7)
        ax.plot_surface(X, Y, Z, cmap=colormap)
        
        plt.show()