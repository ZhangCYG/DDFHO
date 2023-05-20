'''
Sampling functions
Also - add additional training examples by adding d to start point and d to depth
'''
from . import utils_sampling as odf_utils
import numpy as np
import argparse
import datetime
import matplotlib.pyplot as plt
import trimesh
import trimesh.sample
import trimesh.proximity
from tqdm import tqdm
from . import rasterization


def mesh_normalize(verts, return_scale=False):
    '''
    Translates and rescales mesh vertices so that they are tightly bounded within the unit sphere
    '''
    translation = (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2.
    verts = verts - translation
    scale = np.max(np.linalg.norm(verts, axis=1))
    verts = verts / scale
    if return_scale:
        return verts, scale
    else:
        return verts
    

# implemented based on trimesh
def avoid_too_near_points(start_point, end_point):
    residual = np.linalg.norm(start_point - end_point, axis=1)
    random_direction = trimesh.sample.sample_surface_sphere(start_point.shape[0])
    end_point[residual < 1e-4] = start_point[residual < 1e-4] + random_direction[residual < 1e-4]
    return start_point, end_point


#  -------     data type sampling    --------
# start point: sample uniformly in the sphere
# end point: sample uniformly in the sphere
def sample_within_sphere_start_end_uniformly(sample_num):
    start_point = trimesh.sample.sample_surface_sphere(count=sample_num)
    end_point = trimesh.sample.sample_surface_sphere(count=sample_num)

    start_norm = np.expand_dims(np.random.random(size=sample_num), axis=1).repeat(3, axis=1)
    end_norm = np.expand_dims(np.random.random(size=sample_num), axis=1).repeat(3, axis=1)
    start_point = start_point * start_norm
    end_point = end_point * end_norm
    start_point, end_point = avoid_too_near_points(start_point, end_point)
    return start_point, end_point


# start point: sample uniformly in the sphere
# end point: sample randomly on the mesh
def sample_start_uniform_within_sphere_end_mesh(mesh, sample_num):
    start_point = trimesh.sample.sample_surface_sphere(count=sample_num)
    start_norm = np.expand_dims(np.random.random(size=sample_num), axis=1).repeat(3, axis=1)

    end_point, _ = trimesh.sample.sample_surface(mesh, sample_num)
    start_point *= start_norm
    start_point, end_point = avoid_too_near_points(start_point, end_point)
    return start_point, end_point


# start point: sample along the tangent space of the mesh
# end point: sample on the mesh
def sample_start_noise_end_mesh_tangent(mesh, sample_num):
    # first sample end points on the mesh
    end_point, face_index = trimesh.sample.sample_surface(mesh, sample_num)
    faces = mesh.faces[face_index, :]  # n x 3
    vertices_1 = mesh.vertices[faces[:, 0], :]
    vertices_2 = mesh.vertices[faces[:, 1], :]
    vertices_3 = mesh.vertices[faces[:, 2], :]

    vertices_center = (vertices_3 + vertices_2 + vertices_1) / 3.
    end_point_direction = (vertices_center - vertices_2)
    end_point_direction = end_point_direction / np.expand_dims((np.linalg.norm(end_point_direction, axis=1) + 1e-5), axis=1).repeat(3, axis=1)
    start_point_norm = np.expand_dims(np.random.random(size=sample_num), axis=1).repeat(3, axis=1)
    start_point = end_point - end_point_direction * start_point_norm
    start_point, end_point = avoid_too_near_points(start_point, end_point)
    return start_point, end_point


def sample_end_mesh_tangent_noise(mesh, noise, sample_num):
    # first sample end points on the mesh
    end_point, face_index = trimesh.sample.sample_surface(mesh, sample_num)
    faces = mesh.faces[face_index, :]  # n x 3
    vertices_1 = mesh.vertices[faces[:, 0], :]
    vertices_2 = mesh.vertices[faces[:, 1], :]
    vertices_3 = mesh.vertices[faces[:, 2], :]

    vertices_center = (vertices_3 + vertices_2 + vertices_1) / 3.
    end_point_direction = (vertices_center - vertices_2)
    noise_all = np.random.randn(end_point_direction.shape[0], end_point_direction.shape[1]) * noise
    end_point_direction = end_point_direction + noise_all
    end_point_direction = end_point_direction / np.expand_dims((np.linalg.norm(end_point_direction, axis=1) + 1e-5), axis=1).repeat(3, axis=1)

    start_point_norm = np.expand_dims(np.random.random(size=sample_num), axis=1).repeat(3, axis=1)
    start_point = end_point - end_point_direction * start_point_norm
    start_point, end_point = avoid_too_near_points(start_point, end_point)
    return start_point, end_point


def sample_start_on_sphere_end_mesh(mesh, sample_num):
    start_point = trimesh.sample.sample_surface_sphere(count=sample_num)

    end_point, _ = trimesh.sample.sample_surface(mesh, sample_num)
    return start_point, end_point


def sample_symmetry_point(plane, sample_num):
    plane = plane.reshape(3, 1)
    start_point = trimesh.sample.volume_rectangular(extents=(1., 1., 1.), count=sample_num)
    points_on_plane = start_point - np.dot(start_point, plane).dot(plane.transpose())

    return points_on_plane  # the direction is plane normal (plane)


def sample_signed_distance_sphere_surface(mesh, sample_num):
    tpp = trimesh.proximity.ProximityQuery(mesh)
    points = trimesh.sample.sample_surface_sphere(count=sample_num)
    signed_dis = tpp.signed_distance(points)
    return points, signed_dis


def sample_signed_distance_volume_rectangle(mesh, sample_num):
    tpp = trimesh.proximity.ProximityQuery(mesh)
    points = trimesh.sample.volume_rectangular(extents=(1., 1., 1.), count=sample_num)
    signed_dis = tpp.signed_distance(points)
    return points, signed_dis


#  -------     5D SPACE SAMPLING METHODS     -------


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demonstrate ray sampling or run data generation speed tests")
    parser.add_argument("-v", "--viz", action="store_true", help="visualize randomly sampled rays")
    parser.add_argument("-s", "--speed", action="store_true", help="show speed benchmarks for randomly generated rays")
    parser.add_argument("-d", "--depthmap", action="store_true", help="show a depth map image of the mesh")
    parser.add_argument("-c", "--coverage", action="store_true", help="show the intersected vertices of the mesh")
    parser.add_argument("--use_4d", action="store_true", help="show results for the 4D sampling strategies")
    parser.add_argument("--mesh_file", default="F:\\ivl-data\\sample_data\\stanford_bunny.obj",
                        help="Source of mesh file")
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh_file)
    faces = mesh.faces
    verts = mesh.vertices

    verts = odf_utils.mesh_normalize(verts)
    radius = 1.25
    fixed_endpoint = 700

    # threshold for how far away a face can be from the ray before it gets culled in rasterization
    near_face_threshold = rasterization.max_edge(verts, faces)
    vert_normals = odf_utils.get_vertex_normals(verts, faces)

    if not args.use_4d:
        sampling_methods = [sample_uniform_ray_space, sample_vertex_noise, sample_vertex_all_directions,
                            sample_vertex_tangential]
        method_names = ["sample_uniform_ray_space", "sample_vertex_noise", "sample_vertex_all_directions",
                        "sample_vertex_tangential"]
    else:
        sampling_methods = [sample_uniform_4D, sample_vertex_4D, sample_tangential_4D]
        method_names = ["sample_uniform_4D", "sample_vertex_4D", "sample_tangential_4D"]

    if args.viz:
        import visualization

        lines = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, [0, 2]]], axis=0)
        for i, sampling_method in enumerate(sampling_methods):
            visualizer = visualization.RayVisualizer(verts, lines)
            visualizer.set_mesh_color(np.array([1., 0., 0.]))
            print(method_names[i])
            for _ in range(100):
                # Sample a ray
                ray_start, ray_end, v = sampling_method(radius, verts=verts, vert_normals=vert_normals,
                                                        v=fixed_endpoint)
                # rotate and compute depth/occupancy
                rot_verts = rasterization.rotate_mesh(verts, ray_start, ray_end)
                occ = False
                if args.use_4d:
                    depths, intersected_faces = rasterization.ray_all_depths(faces, rot_verts,
                                                                             near_face_threshold=near_face_threshold,
                                                                             ray_start_depth=np.linalg.norm(
                                                                                 ray_end - ray_start),
                                                                             return_faces=True)
                else:
                    occ, depth, intersected_faces = rasterization.ray_occ_depth_visual(faces, rot_verts,
                                                                                       ray_start_depth=np.linalg.norm(
                                                                                           ray_end - ray_start),
                                                                                       near_face_threshold=near_face_threshold,
                                                                                       v=v)
                    depths = [depth]
                # update visualizer
                visualizer.add_sample(ray_start, ray_end, occ, depths,
                                      faces[intersected_faces] if intersected_faces.shape[0] > 0 else [])
            visualizer.show_axes()
            visualizer.display()

    if args.speed:
        n_samples = 1000
        print(f"Generating {n_samples} samples per test")
        for i, sampling_method in enumerate(sampling_methods):
            print(method_names[i])
            start = datetime.datetime.now()
            for _ in range(n_samples):
                ray_start, ray_end, v = sampling_method(radius, verts=verts, vert_normals=vert_normals, v=None)
                rot_verts = rasterization.rotate_mesh(verts, ray_start, ray_end)
                if args.use_4d:
                    depths = rasterization.ray_all_depths(faces, rot_verts, near_face_threshold=near_face_threshold,
                                                          return_faces=False)
                else:
                    occ, depth = rasterization.ray_occ_depth(faces, rot_verts,
                                                             ray_start_depth=np.linalg.norm(ray_end - ray_start),
                                                             near_face_threshold=near_face_threshold, v=v)
            end = datetime.datetime.now()
            secs = (end - start).total_seconds()
            print(f"\t{n_samples / secs :.0f} rays per second")

    if args.depthmap:
        import visualization
        from camera import Camera

        cam_center = [0., 1.0, 1.]
        direction = [0., -1.0, -1.]
        focal_length = 1.0
        sensor_size = [1.0, 1.0]
        resolution = [100, 100]

        # Shows the camera location and orientation in the scene
        direction /= np.linalg.norm(direction)
        if direction[0] == 0. and direction[2] == 0.:
            u_direction = np.array([1., 0., 0.])
            v_direction = np.array([0., 0., 1.]) * (-1. if direction[1] > 0. else 1.)
        else:
            u_direction = np.cross(direction, np.array([0., 1., 0.]))
            v_direction = np.cross(direction, u_direction)
            v_direction /= np.linalg.norm(v_direction)
            u_direction /= np.linalg.norm(u_direction)

        lines = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, [0, 2]]], axis=0)
        visualizer = visualization.RayVisualizer(verts, lines)
        visualizer.add_point([1., 0., 0.], [1., 0., 0.])
        visualizer.add_point([0., 1., 0.], [0., 1., 0.])
        visualizer.add_point([0., 0., 1.], [0., 0., 1.])
        visualizer.add_ray([cam_center, cam_center + direction / np.linalg.norm(direction) * 0.1],
                           np.array([1., 0., 0.]))
        visualizer.add_ray([cam_center, cam_center + u_direction * 0.1], np.array([0., 1., 0.]))
        visualizer.add_ray([cam_center, cam_center + v_direction * 0.1], np.array([0., 0., 1.]))
        visualizer.display()

        cam = Camera(center=cam_center, direction=direction, focal_length=focal_length, sensor_size=sensor_size,
                     sensor_resolution=resolution)
        intersection, depth = cam.mesh_depthmap(cam.rays_on_sphere(cam.generate_rays(), radius), verts, faces)
        plt.imshow(depth)
        plt.show()
        plt.imshow(intersection)
        plt.show()

    if args.coverage:
        lines = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, [0, 2]]], axis=0)
        print(f"There are {faces.shape[0]} faces in the mesh")
        print(f"Sampling 10*{faces.shape[0]} rays per method")
        for i, sampling_method in enumerate(sampling_methods):
            visualizer = visualization.RayVisualizer(verts, lines)
            print(method_names[i])
            face_counts = np.zeros(faces.shape[0]).astype(float)
            # Sample 10 rays per face (on average)
            for _ in tqdm(range(10 * faces.shape[0])):
                # Sample a ray
                ray_start, ray_end, v = sampling_method(radius, verts=verts, vert_normals=vert_normals, v=None)
                # rotate and compute depth/occupancy
                rot_verts = rasterization.rotate_mesh(verts, ray_start, ray_end)
                if args.use_4d:
                    depths, intersected_faces = rasterization.ray_all_depths(faces, rot_verts,
                                                                             near_face_threshold=near_face_threshold,
                                                                             ray_start_depth=np.linalg.norm(
                                                                                 ray_end - ray_start),
                                                                             return_faces=True)
                else:
                    occ, depth, intersected_faces = rasterization.ray_occ_depth_visual(faces, rot_verts,
                                                                                       ray_start_depth=np.linalg.norm(
                                                                                           ray_end - ray_start),
                                                                                       near_face_threshold=near_face_threshold,
                                                                                       v=None)
                face_counts[intersected_faces.astype(int)] += 1.
            upper_limit = 20.
            upper_color = np.array([0., 1., 0.])
            lower_color = np.array([1., 0., 0.])
            pick_face_color = lambda x: np.array([1., 1., 1.]) if face_counts[x] == 0. else ((
                min(face_counts[x] / upper_limit, 1.))) * upper_color + (1. - min(face_counts[x] / upper_limit,
                                                                                  1.)) * lower_color
            mesh_verts = np.vstack(verts[faces[i]] for i in range(faces.shape[0]))
            mesh_faces = np.arange(mesh_verts.shape[0]).reshape((-1, 3))
            mesh_vert_colors = np.vstack(
                [np.vstack([pick_face_color(x)[np.newaxis, :]] * 3) for x in range(faces.shape[0])])
            visualizer.add_colored_mesh(mesh_verts, mesh_faces, mesh_vert_colors)
            visualizer.display()
