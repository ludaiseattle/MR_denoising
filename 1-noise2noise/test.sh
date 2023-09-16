#test
python test.py  --data /home/alyld7/data/test_out/valid --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/checkpoint/stats/n2n-epoch75-0.00067.pt --ckpt-save-path /home/alyld7/data/test_out --nb-epochs 5 --cuda

#t_n180est
nohup python test.py  --data /home/alyld7/data/test/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/test/n2n-epoch40-0.00044.pt --ckpt-save-path /home/alyld7/data/test/checkpoint --nb-epochs 3 --cuda > /home/alyld7/data/test/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/39-test_n250/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/39-n250/epoch100/checkpoint/stats/n2n-epoch40-0.00044.pt --ckpt-save-path /home/alyld7/data/39-n250/test/checkpoint --nb-epochs 100 --cuda > /home/alyld7/data/39-n250/test/output.log 2>&1 &


nohup python test.py  --data /home/alyld7/data/39-test_n300/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/39-n200/epoch100/checkpoint/stats/n2n-epoch45-0.00059.pt --ckpt-save-path /home/alyld7/data/39-n200/epoch100/45_300 --nb-epochs 100 --cuda > /home/alyld7/data/39-n200/epoch100/45_300/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/39-test_n350/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/39-n200/epoch100/checkpoint/stats/n2n-epoch45-0.00059.pt --ckpt-save-path /home/alyld7/data/39-n200/epoch100/45_350 --nb-epochs 100 --cuda > /home/alyld7/data/39-n200/epoch100/45_350/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/39-test_n200/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/39-n200/epoch100/checkpoint/stats/n2n-epoch45-0.00059.pt --ckpt-save-path /home/alyld7/data/39-n200/epoch100/45_200 --nb-epochs 100 --cuda > /home/alyld7/data/39-n200/epoch100/45_200/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/39-test_n250/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/39-n200/epoch100/checkpoint/stats/n2n-epoch45-0.00059.pt --ckpt-save-path /home/alyld7/data/39-n200/epoch100/45_250 --nb-epochs 100 --cuda > /home/alyld7/data/39-n200/epoch100/45_250/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/39-test_n400/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/39-n200/epoch100/checkpoint/stats/n2n-epoch45-0.00059.pt --ckpt-save-path /home/alyld7/data/39-n200/epoch100/45_400 --nb-epochs 100 --cuda > /home/alyld7/data/39-n200/epoch100/45_400/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/39-test_n450/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/39-n200/epoch100/checkpoint/stats/n2n-epoch45-0.00059.pt --ckpt-save-path /home/alyld7/data/39-n200/epoch100/45_450 --nb-epochs 100 --cuda > /home/alyld7/data/39-n200/epoch100/45_450/output.log 2>&1 &





nohup python test.py  --data /home/alyld7/data/41-test_n180/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n180 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n180/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n210/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n210 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n210/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n240/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n240 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n240/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n270/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n270 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n270/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n300/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n300 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n300/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n330/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n330 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n330/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n360/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n360 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n360/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n390/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n390 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n390/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n420/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n420 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n420/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n450/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion/0_3/checkpoint/stats/n2n-epoch60-0.00081.pt --ckpt-save-path /home/alyld7/data/41-nfusion/0_3/65_n450 --nb-epochs 100 --cuda > /home/alyld7/data/41-nfusion/0_3/65_n450/output.log 2>&1 &
----------------------------------------------------------------------------------
nohup python test.py  --data /home/alyld7/data/41-test_n180/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n180 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n180/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n210/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n210 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n210/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n240/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n240 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n240/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n270/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n270 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n270/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n300/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n300 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n300/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n330/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n330 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n330/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n360/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n360 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n360/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n390/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n390 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n390/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n420/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n420 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n420/output.log 2>&1 &

nohup python test.py  --data /home/alyld7/data/41-test_n450/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-n180/0_3/checkpoint/stats/n2n-epoch75-0.00169.pt --ckpt-save-path /home/alyld7/data/41-n180/0_3/75_n450 --nb-epochs 100 --cuda > /home/alyld7/data/41-n180/0_3/75_n450/output.log 2>&1 &
