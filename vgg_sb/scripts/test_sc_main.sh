set -ex
python test.py \
--dataroot ./dataset/summer2winter_yosemite  \
--checkpoints_dir ./checkpoints \
--name sum2win_vgg \
--model sc \
--num_test 10 --phase test