""" Rover path planning simulation using Sequential Convex Programming (SCP). """

from time import time
import numpy as np
import matplotlib.pyplot as plt
from SCP import SCP
from Rover import Rover 

if __name__=="__main__":
    Leo_Rover = Rover()
    initialization = {}
    initialization["valid"] = False
    scp_planner = SCP(Rover=Leo_Rover, initialization=initialization)
    t0 = time()
    scp_planner.scp()
    print(f"SCP completed in {time() - t0:.2f} s")
    xscp = np.array(scp_planner.sol["state"])
    uscp = np.array(scp_planner.sol["control"])
    np.save("xscp.npy", xscp)
    np.save("uscp.npy", uscp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xscp[:,0], xscp[:,1], xscp[:,2], label="SCP Trajectory", linewidth=3)
    ax.set_title("SCP Optimized Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("θ [rad]")
    ax.legend()
    plt.show()
