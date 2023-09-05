import pyvista as pv

def points2stl(points, file_name):
    """Saves given point cloud as stl file"""
    point_cloud = pv.PolyData(points)
    mesh = point_cloud.reconstruct_surface()
    mesh.save('./10_3D-Images/PointNet/stl_files/' + file_name + '.stl')
    point_cloud.plot()
    mesh.plot()