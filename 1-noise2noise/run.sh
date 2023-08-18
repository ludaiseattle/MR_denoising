python train.py --train-dir /home/alyld7/data/35-1800_8_random_circle8_addelse/samp1 --train-target-dir /home/alyld7/data/35-1800_8_random_circle8_addelse/samp2 --train-size 600 --valid-dir /home/alyld7/data/1-test --valid-target-dir /home/alyld7/data/1-test --valid-size 220 --ckpt-save-path /home/alyld7/data/35-1800_8_random_circle8_addelse/0_0_5/checkpoint --ckpt-overwrite --report-interval 60 --nb-epochs 150 --batch-size 3 --loss l2 --noise-type gaussian --noise-param 50 --crop-size 64 --plot-stats --cuda > /home/alyld7/data/35-1800_8_random_circle8_addelse/0_0_5/output.log 2>&1;python train.py --train-dir /home/alyld7/data/35-1800_8_random_circle16_addelse/samp1 --train-target-dir /home/alyld7/data/35-1800_8_random_circle16_addelse/samp2 --train-size 600 --valid-dir /home/alyld7/data/1-test --valid-target-dir /home/alyld7/data/1-test --valid-size 220 --ckpt-save-path /home/alyld7/data/35-1800_8_random_circle16_addelse/0_0_5/checkpoint --ckpt-overwrite --report-interval 60 --nb-epochs 150 --batch-size 3 --loss l2 --noise-type gaussian --noise-param 50 --crop-size 64 --plot-stats --cuda > /home/alyld7/data/35-1800_8_random_circle16_addelse/0_0_5/output.log 2>&1;python train.py --train-dir /home/alyld7/data/35-1800_8_random_circle4_addelse/samp1 --train-target-dir /home/alyld7/data/35-1800_8_random_circle4_addelse/samp2 --train-size 600 --valid-dir /home/alyld7/data/1-test --valid-target-dir /home/alyld7/data/1-test --valid-size 220 --ckpt-save-path /home/alyld7/data/35-1800_8_random_circle4_addelse/0_0_5/checkpoint --ckpt-overwrite --report-interval 60 --nb-epochs 150 --batch-size 3 --loss l2 --noise-type gaussian --noise-param 50 --crop-size 64 --plot-stats --cuda > /home/alyld7/data/35-1800_8_random_circle4_addelse/0_0_5/output.log 2>&1 