import numpy as np
import trimesh
import trimesh.ray.ray_triangle as rayt
import multiprocessing
import time
from pathos.multiprocessing import ProcessingPool as Pool


def direct_ddf_generation(start_point, end_point, mesh):
    direction = end_point - start_point
    direction = direction / np.expand_dims(np.linalg.norm(direction, axis=1), axis=1).repeat(3, axis=1)
    intersection_mask = np.zeros(start_point.shape[0])
    intersection_depth = np.zeros(start_point.shape[0]) - 1.
    locations_res = np.zeros((start_point.shape[0], 3))

    ray_detector = rayt.RayMeshIntersector(mesh)
    # index_ray, the index of ray that intersects with the mesh
    # locations: coordinates of the intersections
    locations, index_ray, _ = ray_detector.intersects_location(ray_origins=start_point, ray_directions=direction, multiple_hits=False)
        
    intersection_mask[index_ray] = 1.

    depth_intersection = np.linalg.norm(locations - start_point[index_ray], axis=1)
    intersection_depth[index_ray] = depth_intersection

    locations_res[index_ray] = locations

    # print('Dir Check: ', np.cross(locations - start_point[index_ray], direction[index_ray], axis=1))
    # print('Depth Check: ', locations - start_point[index_ray] - direction[index_ray] * np.expand_dims(intersection_depth[index_ray], axis=1).repeat(3, axis=1))

    return start_point, direction, intersection_mask, intersection_depth, locations_res


def batch_direct_ddf_generation_mp(start_point_all, end_point_all, mesh):
    inner_bs = 2000
    mp_manager = multiprocessing.Manager()
    direc = mp_manager.list()
    inter_mask = mp_manager.list()
    inter_depth = mp_manager.list()
    loc = mp_manager.list()

    def mp_func(i):
        print("Batch: ", i)
        end_point = end_point_all[i * inner_bs: (i + 1) * inner_bs, :]
        start_point = start_point_all[i * inner_bs: (i + 1) * inner_bs, :]

        _, direction, intersection_mask, intersection_depth, locations_res = direct_ddf_generation(start_point, end_point, mesh)

        direc.append(direction)
        inter_mask.append(intersection_mask)
        inter_depth.append(intersection_depth)
        loc.append(locations_res)

        time.sleep(0.1)
        print("Finished Batch: ", i)

    pool = Pool(processes=8) # number of processes
    try:
        pool.map(mp_func, range(int(start_point_all.shape[0] / inner_bs)))
    except:
        pool.restart()
        pool.map(mp_func, range(int(start_point_all.shape[0] / inner_bs)))
    pool.close()
    pool.join()

    direc = np.concatenate(direc, axis=0)
    inter_mask = np.concatenate(inter_mask, axis=0)
    inter_depth = np.concatenate(inter_depth, axis=0)
    loc = np.concatenate(loc, axis=0)

    return start_point_all, direc, inter_mask, inter_depth, loc


def batch_direct_ddf_generation(start_point_all, end_point_all, mesh):
    inner_bs = 4000
    direc = []
    inter_mask = []
    inter_depth = []
    loc = []

    for i in range(int(start_point_all.shape[0] / inner_bs)):
        print("Batch: ", i)
        end_point = end_point_all[i * inner_bs: (i + 1) * inner_bs, :]
        start_point = start_point_all[i * inner_bs: (i + 1) * inner_bs, :]

        _, direction, intersection_mask, intersection_depth, locations_res = direct_ddf_generation(start_point, end_point, mesh)

        direc.append(direction)
        inter_mask.append(intersection_mask)
        inter_depth.append(intersection_depth)
        loc.append(locations_res)

    direc = np.concatenate(direc, axis=0)
    inter_mask = np.concatenate(inter_mask, axis=0)
    inter_depth = np.concatenate(inter_depth, axis=0)
    loc = np.concatenate(loc, axis=0)

    return start_point_all, direc, inter_mask, inter_depth, loc


# mesh symmetry detection based under the canonical pose
def sym_mesh_detection(points_on_plane, plane_normal, mesh):
    ray_detector = rayt.RayMeshIntersector(mesh)
    intersection_mask = np.zeros(points_on_plane.shape[0])
    intersection_depth = np.zeros_like(intersection_mask) - 1.
    plane_normal = np.expand_dims(plane_normal, axis=0).repeat(points_on_plane.shape[0], axis=0)
    _, index_ray, locations = ray_detector.intersects_id(ray_origins=points_on_plane, ray_directions=plane_normal,
                                                         return_locations=True, multiple_hits=False)

    _, index_ray_re, locations_re = ray_detector.intersects_id(ray_origins=points_on_plane,
                                                               ray_directions=-plane_normal,
                                                               return_locations=True, multiple_hits=False)
    if index_ray.shape[0] == 0:
        return False, None, None, None, None
    if index_ray.shape == index_ray_re.shape:
        if np.mean(np.linalg.norm(locations + locations_re - 2.0 * points_on_plane[index_ray, :], axis=-1)) < 1e-2:
            print('Maybe Symmetric!')
            print(np.mean(np.linalg.norm(locations + locations_re - 2.0 * points_on_plane[index_ray, :], axis=-1)))
            intersection_mask[index_ray] = 1.
            intersection_depth[index_ray] = np.linalg.norm(locations - points_on_plane[index_ray, :], axis=1)
            return True, points_on_plane, plane_normal, intersection_mask, intersection_depth
        else:
            print('Maybe not Symmetric!')
            print(np.mean(np.linalg.norm(locations + locations_re - 2.0 * points_on_plane[index_ray, :], axis=-1)))
            return False, None, None, None, None
    else:
            return False, None, None, None, None
