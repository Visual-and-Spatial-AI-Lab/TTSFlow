import subprocess
import time

chairs=[
"python", "main.py",
"--checkpoint_dir", "chairs",
"--stage", "chairs",
"--batch_size", "16",
"--num_steps", "200000",
"--val_dataset", "chairs", "sintel",
"--output_path", "chairs",
"--lr", "4e-4",
"--image_size", "384", "512",
"--padding_factor", "16",
"--upsample_factor", "8",
"--with_speed_metric",
"--val_freq", "20000",
"--save_ckpt_freq", "20000",
"--dino_path", "facebook/dinov2-small",
"--depth_model_path", "depth_anything_v2_ckpt.pth"
]

things=[
"python", "main.py",
"--checkpoint_dir", "things",
"--resume", "chairs/step_200000.pth",
"--stage", "things",
"--batch_size", "8",
"--num_steps", "800000",
"--val_dataset", "things", "sintel",
"--output_path", "things",
"--lr", "2e-4",
"--image_size", "384", "768",
"--padding_factor", "16",
"--upsample_factor", "8",
"--with_speed_metric",
"--val_freq", "20000",
"--save_ckpt_freq", "20000",
"--dino_path", "facebook/dinov2-small",
"--depth_model_path", "depth_anything_v2_ckpt.pth"
]


sintel=[
"python", "main.py",
"--checkpoint_dir", "sintel",
"--resume", "things/step_800000.pth",
"--stage", "sintel",
"--batch_size", "8",
"--num_steps", "200000",
"--val_dataset", "sintel",
"--output_path", "sintel",
"--lr", "2e-4",
"--image_size", "320", "896",
"--padding_factor", "16",
"--upsample_factor", "8",
"--with_speed_metric",
"--val_freq", "20000",
"--save_ckpt_freq", "20000",
"--dino_path", "facebook/dinov2-small",
"--depth_model_path", "depth_anything_v2_ckpt.pth"
]



kitti=[
"python", "main.py",
"--checkpoint_dir", "kitti",
"--resume", "sintel/step_200000.pth",
"--stage", "kitti",
"--batch_size", "8",
"--num_steps", "90000",
"--val_dataset", "kitti",
"--output_path", "kitti",
"--lr", "2e-4",
"--image_size", "320", "1152",
"--padding_factor", "16",
"--upsample_factor", "8",
"--with_speed_metric",
"--val_freq", "10000",
"--save_ckpt_freq", "20000",
"--dino_path", "facebook/dinov2-small",
"--depth_model_path", "depth_anything_v2_ckpt.pth"
]


for i, cmd in enumerate([chairs,things,sintel,kitti],start=1):
    print ("****** ---> ********* ",i,cmd)
    subprocess.run(cmd,shell=True,check=True)
    print("Starting Time delay ------")
    time.sleep(600)
    print("Ending Time delay")


print("All Done -------------")