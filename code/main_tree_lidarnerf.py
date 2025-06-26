# main_tree_lidarnerf.py
import torch
from torch.utils.data import DataLoader
from tree_dataset import TreePointCloudDataset
from lidarnerf.lidar_field import LiDARNeRFNetwork  # assumed location
from lidarnerf.trainer import Trainer  # assumed trainer file

# ======== CONFIGURATION ========
xyz_path = "data/tree_sample.xyz"  # path to your .xyz file
H, W = 64, 512  # range image resolution
batch_size = 1024
num_epochs = 100
lr = 1e-2

# ======== LOAD DATA ========
dataset = TreePointCloudDataset(xyz_path, H=H, W=W)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ======== INITIALIZE MODEL ========
model = LiDARNeRFNetwork(
    use_viewdirs=False,
    input_ch=3,
    input_ch_views=0,
    D=4,
    W=128
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ======== TRAINING LOOP ========
trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)
trainer.train(num_epochs=num_epochs)

# ======== SAVE MODEL ========
torch.save(model.state_dict(), "lidar_nerf_tree.pth")
print("Model saved as lidar_nerf_tree.pth")
