#computation libraries
import numpy as np
import matplotlib.pyplot as plt

#benchmarking libraries
from time import time
import cProfile, io
import pstats

#3d printing libraries
import open3d as o3d

#homemade libraries
import dispersion
import orbitcreation
import conductivity

def main():

    dispersionInstance = dispersion.LSCOdispersion()
    initialpointsInstance = orbitcreation.InterpolatedCurves(200,dispersionInstance,True)

    starttime = time()
    initialpointsInstance.solveforpoints(parallelised=False)
    initialpointsInstance.extendedZoneMultiply(0)
    initialpointsInstance.createPlaneAnchors(200)
    #initialpointsInstance.plotpoints()
    endtime = time()

    print(f"Time taken to create initialcurves = {endtime - starttime}")


    #ax = plt.figure().add_subplot(projection='3d')

    #plotting extendedcurveslist
    #for curve in initialpointsInstance.extendedcurvesList:
    #    ax.scatter(curve[:,0],curve[:,1], curve[:,2], label='parametric curve',s=1)

    theta = np.deg2rad(0)
    phi = np.deg2rad(0)
    B = [45*np.sin(theta)*np.cos(phi),45*np.sin(theta)*np.sin(phi),45*np.cos(theta)]

    #plottingintersections
    #ax.scatter(intersections[:,0],intersections[:,1],intersections[:,2],c='#FF0000',s=10)

    starttime = time()
    orbitsinstance = orbitcreation.NewOrbits(dispersionInstance,initialpointsInstance)
    orbitsinstance.createOrbits(B,termination_resolution=0.005,mult_factor=10)
    orbitsinstance.createOrbitsEQS(integration_resolution=0.005)

    #creates the point cloud for 3d printing
    pointcloud = np.concatenate(orbitsinstance.orbitsEQS)
    pointcloud_scaled95 = np.transpose([pointcloud[:,0]*0.95,pointcloud[:,1]*0.95,pointcloud[:,2]])
    pointcloud_scaled97 = np.transpose([pointcloud[:,0]*0.97,pointcloud[:,1]*0.97,pointcloud[:,2]])
    pointcloud = np.concatenate((pointcloud,pointcloud_scaled95,pointcloud_scaled97))
    print(pointcloud)

    #plots the pointcloud using matplotlib
    #ax = plt.figure().add_subplot(projection='3d')

    #ax.scatter(pointcloud[:,0],pointcloud[:,1],pointcloud[:,2])
    #plt.show()

    #creates a pointcloud object (pcl) from coordinates in pointclould
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.visualization.draw_geometries([pcl])

    #creates a mesh from pointcloud object
    alpha = 0.015
    print(f"alpha={alpha:.3f}")

    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcl)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcl, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    o3d.io.write_triangle_mesh("lsco.stl",mesh,print_progress=True)

main()