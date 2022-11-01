import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from matplotlib.colors import LightSource
from scipy.spatial import ConvexHull, HalfspaceIntersection

# Wheels are on a circle of robot radius [m]
robot_radius = 1

# Those are the angles of the wheels [rad]
wheel_angles = [np.deg2rad(60), np.deg2rad(-60), np.deg2rad(180)]

# This is the maximum linear speed of a wheel [m/s]
vmax = 1

# Slice with the plane theta=0 ?
slice = True

# Computing wheel positions
R = np.array(
    [[robot_radius * np.cos(a), robot_radius * np.sin(a)] for a in wheel_angles]
)

# Computing wheel normals (tangent to the circle)
N = np.array([[-np.sin(a), np.cos(a)] for a in wheel_angles])

# Computing inverse kinematics matrix
# w = Ms, where w are wheel speeds and s the chassis speed
M = np.array(
    [
        [N[i][0], N[i][1], N[i][1] * R[i][0] - N[i][0] * R[i][1]]
        for i in range(len(wheel_angles))
    ]
)

# Half-plane constraints
lt = np.hstack((M, -vmax * np.ones((len(M), 1))))
gt = np.hstack((-M, -vmax * np.ones((len(M), 1))))
constraints = np.vstack((lt, gt))

if slice:
    # Adding a constraint to slice the feasible zone with plane theta=0
    constraints = np.vstack((constraints, np.array([[0.0, 0.0, 1.0, 0.0]])))

# Computing the intersection of the half spaces
hs = HalfspaceIntersection(constraints, np.array([0.0, 0.0, -1e-3]))
pts = hs.intersections

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Showing the points
ax.scatter(pts.T[0], pts.T[1], pts.T[2], c="black")

# Computing the convex hull and drawing the simplices
hull = ConvexHull(pts)
faces = [pts[s] for s in hull.simplices]
f = a3.art3d.Poly3DCollection(faces, alpha=1.0)
ax.add_collection(f)
ls = LightSource(azdeg=0.0, altdeg=35.0)

normals = np.array([eq[:3] for eq in hull.equations])

f.set_facecolor(
    [
        np.array([0.5, 0.5, 1.0]) * normal
        for normal in 0.3 + 0.7 * ls.shade_normals(normals)
    ]
)

ax.set_xlabel("x_d")
ax.set_ylabel("y_d")
ax.set_zlabel("theta_d")

plt.show()
