import os

import numpy as np
import open3d as o3d
import segmentator
import torch


def get_superpoint(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces)
    return superpoint


if __name__ == "__main__":
    os.makedirs("superpoints", exist_ok=True)
    scans_trainval = os.listdir("ScanNet_archive/scans/")
    for scan in scans_trainval:
        ply_file = os.path.join("ScanNet_archive/scans", scan, f"{scan}_vh_clean_2.ply")
        spp = get_superpoint(ply_file)
        spp = spp.numpy()

        torch.save(spp, os.path.join("superpoints", f"{scan}.pth"))

    scans_test = os.listdir("ScanNet_archive/scans_test/")
    for scan in scans_test:
        ply_file = os.path.join("ScanNet_archive/scans_test", scan, f"{scan}_vh_clean_2.ply")
        spp = get_superpoint(ply_file)
        spp = spp.numpy()

        torch.save(spp, os.path.join("ScanNetV2_seg/Gapro/superpoints", f"{scan}.pth"))
