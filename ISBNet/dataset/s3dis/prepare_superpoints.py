import glob

import numpy as np
import torch


files = sorted(glob.glob("/ssd/Dataset/s3dis/gapro/learned_superpoint_graph_segmentations/*.npy"))

for file in files:
    chunks = file.split("/")[-1].split(".")
    area = chunks[0]
    room = chunks[1]

    spp = np.load(file, allow_pickle=True).item()["segments"]
    torch.save((spp), f"/ssd/Dataset/s3dis/gapro/superpoints/{area}_{room}.pth")
