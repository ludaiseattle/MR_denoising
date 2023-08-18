# whole process
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/ &
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/1-test_18000_800_0_400 > ../data/1-test_18000_800_0_400/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/37-18000_800_0_400 > ../data/37-18000_800_0_400/output.log 2>&1 &

#test
python denoising.py -f whole -i ./testin -o ./testout
python denoising.py -f whole -i ../data/test_input/ -o ../data/test_out

