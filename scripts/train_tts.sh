## Our model is trained on 2 48GB RTX 6000 GPU
## Please adjust the batch size according to your hardware
## We used DataParallel for training with both GPU's present on the same node

## Train on chairs for 200k
CHECKPOINT_DIR=checkpoints/chairs && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage chairs \
--batch_size 16 \
--num_steps 200000 \
--val_dataset chairs sintel \
--output_path chairs \
--lr 4e-4 \
--image_size 384 512 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 20000 \
--save_ckpt_freq 20000 \
--dino_path facebook/dinov2-small \
--depth_model_path depth_anything_v2_ckpt.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


## Train on Things for 800k
CHECKPOINT_DIR=checkpoints/things && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints/chairs/step_200000.pth \
--stage things \
--batch_size 8 \
--num_steps 800000 \
--val_dataset things sintel \
--output_path things \
--lr 2e-4 \
--image_size 384 768 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 20000 \
--save_ckpt_freq 20000 \
--dino_path facebook/dinov2-small \
--depth_model_path depth_anything_v2_ckpt.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log


## Train on TSKH for 200k
CHECKPOINT_DIR=checkpoints/sintel && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints/things/step_800000.pth \
--stage sintel \
--batch_size 8 \
--num_steps 200000 \
--val_dataset sintel \
--output_path sintel \
--lr 2e-4 \
--image_size 320 896 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 20000 \
--save_ckpt_freq 20000 \
--dino_path facebook/dinov2-small \
--depth_model_path depth_anything_v2_ckpt.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log

## Train on KITTI for 90k
CHECKPOINT_DIR=checkpoints/kitti && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--resume checkpoints/sintel/step_200000.pth \
--stage kitti \
--batch_size 8 \
--num_steps 90000 \
--val_dataset kitti \
--output_path kitti \
--lr 2e-4 \
--image_size 320 1152 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 10000 \
--save_ckpt_freq 20000 \
--dino_path facebook/dinov2-small \
--depth_model_path depth_anything_v2_ckpt.pth \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
