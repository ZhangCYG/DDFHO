import os
import pickle
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import trimesh


class ObManPre():
    def __init__(self,
                 root,
                 shapenet_root,
                 split='train',
                 ):
        self.split = split
        self.root = os.path.join(root, split)

        if shapenet_root.endswith('/'):
            shapenet_root = shapenet_root[:-1]
        self.shapenet_root = shapenet_root
        self.meta_folder = os.path.join(self.root, "meta")
        self.prefix_template = '{:08d}'
        self.rgb_folder = os.path.join(self.root, "rgb")
        self.shapenet_template = os.path.join(
            shapenet_root, "{}/{}/models/model_normalized.pkl"
        )
        self.cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                                  [0., 0., -1., 0.]]).astype(np.float32)
        self.load_dataset()
    
    def _get_image_path(self, prefix):
        image_path = os.path.join(self.rgb_folder, '{}.jpg'.format(prefix))

        return image_path
    
    def _get_obj_path(self, class_id, sample_id):
        shapenet_path = self.shapenet_template.format(class_id, sample_id)
        return shapenet_path
    
    def load_dataset(self):
        idxs = [
            int(imgname.split(".")[0])
            for imgname in sorted(os.listdir(self.meta_folder))
        ]

        prefixes = [self.prefix_template.format(idx) for idx in idxs]
        print(
            "Got {} samples for split {}, generating cache !".format(
                len(idxs), self.split
            )
        )

        image_names = []
        all_joints2d = []
        all_joints3d = []
        hand_sides = []
        hand_poses = []
        hand_pcas = []
        hand_verts3d = []
        obj_paths = []
        obj_transforms = []
        meta_infos = []
        depth_infos = []
        for idx, prefix in enumerate(tqdm(prefixes)):
            meta_path = os.path.join(
                self.meta_folder, "{}.pkl".format(prefix)
            )
            with open(meta_path, "rb") as meta_f:
                meta_info = pickle.load(meta_f)
            image_path = self._get_image_path(prefix)
            image_names.append(image_path)
            all_joints2d.append(meta_info["coords_2d"])
            all_joints3d.append(meta_info["coords_3d"])
            hand_verts3d.append(meta_info["verts_3d"])
            hand_sides.append(meta_info["side"])
            hand_poses.append(meta_info["hand_pose"])
            hand_pcas.append(meta_info["pca_pose"])
            depth_infos.append(
                {
                    "depth_min": meta_info["depth_min"],
                    "depth_max": meta_info["depth_max"],
                    "hand_depth_min": meta_info["hand_depth_min"],
                    "hand_depth_max": meta_info["hand_depth_max"],
                    "obj_depth_min": meta_info["obj_depth_min"],
                    "obj_depth_max": meta_info["obj_depth_max"],
                }
            )
            obj_path = self._get_obj_path(
                meta_info["class_id"], meta_info["sample_id"]
            )

            obj_paths.append(obj_path)
            obj_transforms.append(meta_info["affine_transform"])
            meta_info_full = {
                "obj_scale": meta_info["obj_scale"],
                "obj_class_id": meta_info["class_id"],
                "obj_sample_id": meta_info["sample_id"],
            }
            if "grasp_quality" in meta_info:
                meta_info_full["grasp_quality"] = meta_info[
                    "grasp_quality"
                ]
                meta_info_full["grasp_epsilon"] = meta_info[
                    "grasp_epsilon"
                ]
                meta_info_full["grasp_volume"] = meta_info["grasp_volume"]
            meta_infos.append(meta_info_full)

        annotations = {
            "depth_infos": depth_infos,
            "image_names": image_names,
            "joints2d": all_joints2d,
            "joints3d": all_joints3d,
            "hand_sides": hand_sides,
            "hand_poses": hand_poses,
            "hand_pcas": hand_pcas,
            "hand_verts3d": hand_verts3d,
            "obj_paths": obj_paths,
            "obj_transforms": obj_transforms,
            "meta_infos": meta_infos,
        }
        print(
            "class_nb: {}".format(
                np.unique(
                    [
                        (meta_info["obj_class_id"])
                        for meta_info in meta_infos
                    ],
                    axis=0,
                ).shape
            )
        )
        print(
            "sample_nb : {}".format(
                np.unique(
                    [
                        (
                            meta_info["obj_class_id"],
                            meta_info["obj_sample_id"],
                        )
                        for meta_info in meta_infos
                    ],
                    axis=0,
                ).shape
            )
        )

        # Set dataset attributes
        all_objects = [
            obj[:-7].split("/")[-1].split("_")[0]
            for obj in annotations["obj_paths"]
        ]
        selected_idxs = list(range(len(all_objects)))
        obj_paths = [annotations["obj_paths"][idx] for idx in selected_idxs]
        image_names = [
            annotations["image_names"][idx] for idx in selected_idxs
        ]
        joints3d = [annotations["joints3d"][idx] for idx in selected_idxs]
        joints2d = [annotations["joints2d"][idx] for idx in selected_idxs]
        hand_sides = [annotations["hand_sides"][idx] for idx in selected_idxs]
        hand_pcas = [annotations["hand_pcas"][idx] for idx in selected_idxs]
        hand_verts3d = [
            annotations["hand_verts3d"][idx] for idx in selected_idxs
        ]
        obj_transforms = [
            annotations["obj_transforms"][idx] for idx in selected_idxs
        ]
        meta_infos = [annotations["meta_infos"][idx] for idx in selected_idxs]
        if "depth_infos" in annotations:
            has_depth_info = True
            depth_infos = [
                annotations["depth_infos"][idx] for idx in selected_idxs
            ]
        else:
            has_depth_info = False
        if has_depth_info:
            self.depth_infos = depth_infos
        self.image_names = image_names
        self.joints2d = joints2d
        self.joints3d = joints3d
        self.hand_sides = hand_sides
        self.hand_pcas = hand_pcas
        self.hand_verts3d = hand_verts3d
        self.obj_paths = obj_paths
        self.obj_transforms = obj_transforms
        self.meta_infos = meta_infos
        # Initialize cache for center and scale in case objects are used
        self.center_scale_cache = {}

    def get_obj_verts_faces(self, idx):
        model_path = self.obj_paths[idx]
        model_path_obj = model_path.replace(".pkl", ".obj")
        if os.path.exists(model_path):
            with open(model_path, "rb") as obj_f:
                mesh = pickle.load(obj_f)
        elif os.path.exists(model_path_obj):
            with open(model_path_obj, "r") as m_f:
                mesh = fast_load_obj(m_f)[0]
        else:
            raise ValueError(
                "Could not find model pkl or obj file at {}".format(
                    model_path.split(".")[-2]
                )
            )

        # NOTE: no_scale

        # obj_scale = self.meta_infos[idx]["obj_scale"]
        # verts = mesh["vertices"] * obj_scale
        verts = mesh["vertices"]

        # # Apply transforms
        # obj_transform = self.obj_transforms[idx]
        # hom_verts = np.concatenate(
        #     [verts, np.ones([verts.shape[0], 1])], axis=1
        # )
        # trans_verts = obj_transform.dot(hom_verts.T).T[:, :3]
        # trans_verts = (
        #     self.cam_extr[:3, :3].dot(trans_verts.transpose()).transpose()
        # )
        trans_verts = verts
        
        return (
            np.array(trans_verts).astype(np.float32),
            np.array(mesh["faces"]).astype(np.int16),
            model_path.replace('database/ShapeNetCore.v2', '').replace('/models/model_normalized.pkl', '')
        )
    
    def get_obj_mesh(self, idx):
        model_path = self.obj_paths[idx]
        model_path_obj = model_path.replace(".pkl", ".obj")
        if os.path.exists(model_path):
            with open(model_path, "rb") as obj_f:
                mesh = pickle.load(obj_f)
        elif os.path.exists(model_path_obj):
            mesh = trimesh.load(model_path_obj, force='mesh')
        else:
            raise ValueError(
                "Could not find model pkl or obj file at {}".format(
                    model_path.split(".")[-2]
                )
            )
        return mesh, model_path.replace('database/ShapeNetCore.v2', '').replace('/models/model_normalized.pkl', '')

    def __len__(self):
        return len(self.image_names)


def fast_load_obj(file_obj, **kwargs):
    """
    Code slightly adapted from trimesh (https://github.com/mikedh/trimesh) 
    Thanks to Michael Dawson-Haggerty for this great library !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.

    vertices with the same position but different normals or uvs
    are split into multiple vertices.

    colors are discarded.

    parameters
    ----------
    file_obj : file object
                   containing a wavefront file

    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """

    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'

    meshes = []

    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current['f']) > 0:
            # get vertices as clean numpy array
            vertices = np.array(
                current['v'], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current['f'], dtype=np.int64).reshape((-1, 3))

            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (np.array(list(remap.keys())),
                            np.array(list(remap.values())))
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)

            # apply the ordering and put into kwarg dict
            loaded = {
                'vertices': vertices[vert_order],
                'faces': face_order[faces],
                'metadata': {}
            }

            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current['g']) > 0:
                face_groups = np.zeros(len(current['f']) // 3, dtype=np.int64)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups

            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ['v']}
    current = {k: [] for k in ['v', 'f', 'g']}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0]) - 1])
                current['f'].append(remap[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))

    if next_idx > 0:
        append_mesh()

    return meshes