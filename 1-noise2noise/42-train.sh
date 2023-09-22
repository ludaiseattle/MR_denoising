#fusion1-net6
nohup python train.py --train-dir /home/alyld7/data/42-nfusion1/samp1 --train-target-dir /home/alyld7/data/42-nfusion1/samp2 --train-size 7200 --valid-dir /home/alyld7/data/42-test_n180_batch1/samp1 --valid-target-dir /home/alyld7/data/1-test --valid-size 220 --ckpt-save-path /home/alyld7/data/42-nfusion1/net6_005/checkpoint --report-interval 60 --nb-epochs 100 --batch-size 3 --loss l2 --learning-rate 0.001 --plot-stats --cuda > /home/alyld7/data/42-nfusion1/net6_005/output.log 2>&1 &

#fusion3-net6
nohup python train.py --train-dir /home/alyld7/data/42-nfusion3/samp1 --train-target-dir /home/alyld7/data/42-nfusion3/samp2 --train-size 21600 --valid-dir /home/alyld7/data/42-test_n180_batch1/samp1 --valid-target-dir /home/alyld7/data/1-test --valid-size 220 --ckpt-save-path /home/alyld7/data/42-nfusion3/net6_005/checkpoint --report-interval 60 --nb-epochs 100 --batch-size 3 --loss l2 --learning-rate 0.001 --plot-stats --cuda > /home/alyld7/data/42-nfusion3/net6_005/output.log 2>&1 &

#fusion3-net2
nohup python train.py --train-dir /home/alyld7/data/42-nfusion3/samp1 --train-target-dir /home/alyld7/data/42-nfusion3/samp2 --train-size 21600 --valid-dir /home/alyld7/data/42-test_n180_batch1/samp1 --valid-target-dir /home/alyld7/data/1-test --valid-size 220 --ckpt-save-path /home/alyld7/data/42-nfusion3/net2_ori/checkpoint --report-interval 60 --nb-epochs 100 --batch-size 3 --loss l2 --learning-rate 0.001 --plot-stats --cuda > /home/alyld7/data/42-nfusion3/net2_ori/output.log 2>&1 &

#fusion6-net2
nohup python train.py --train-dir /home/alyld7/data/42-nfusion6/samp1 --train-target-dir /home/alyld7/data/42-nfusion6/samp2 --train-size 43200 --valid-dir /home/alyld7/data/42-test_n180_batch1/samp1 --valid-target-dir /home/alyld7/data/1-test --valid-size 220 --ckpt-save-path /home/alyld7/data/42-nfusion6/net2_ori/checkpoint --report-interval 60 --nb-epochs 100 --batch-size 3 --loss l2 --learning-rate 0.001 --plot-stats --cuda > /home/alyld7/data/42-nfusion6/net2_ori/output.log 2>&1 &

#fusion6-net9
nohup python train.py --train-dir /home/alyld7/data/42-nfusion6/samp1 --train-target-dir /home/alyld7/data/42-nfusion6/samp2 --train-size 43200 --valid-dir /home/alyld7/data/42-test_n180_batch1/samp1 --valid-target-dir /home/alyld7/data/1-test --valid-size 220 --ckpt-save-path /home/alyld7/data/42-nfusion6/net9_001/checkpoint --report-interval 60 --nb-epochs 100 --batch-size 3 --loss l2 --learning-rate 0.001 --plot-stats --cuda > /home/alyld7/data/42-nfusion6/net9_001/output.log 2>&1 &
