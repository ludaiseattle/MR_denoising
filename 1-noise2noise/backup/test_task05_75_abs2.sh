python test.py  --data /home/alyld7/data/41-test_n270/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/checkpoint/stats/n2n-epoch75-0.00067.pt --ckpt-save-path /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/75_abs_n270 --nb-epochs 1 --cuda > /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/75_abs_n270/output.log 2>&1;python test.py  --data /home/alyld7/data/41-test_n300/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/checkpoint/stats/n2n-epoch75-0.00067.pt --ckpt-save-path /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/75_abs_n300 --nb-epochs 1 --cuda > /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/75_abs_n300/output.log 2>&1; python test.py  --data /home/alyld7/data/41-test_n330/samp1 --test-target-dir /home/alyld7/data/1-test --load-ckpt /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/checkpoint/stats/n2n-epoch75-0.00067.pt --ckpt-save-path /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/75_abs_n330 --nb-epochs 1 --cuda > /home/alyld7/data/41-nfusion2/0_3_dropout_0_5/75_abs_n330/output.log 2>&1;