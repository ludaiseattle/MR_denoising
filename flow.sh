# whole process
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/ &
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/1-test_18000_800_0_400 > ../data/1-test_18000_800_0_400/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/37-18000_800_0_400 > ../data/37-18000_800_0_400/output.log 2>&1 &

#test
python denoising.py -f whole -i ./testin -o ./testout
python denoising.py -f whole -i ../data/test_input/ -o ../data/test_out

#new
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/1-test_n250 > ../data/1-test_n250/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/38-n250 > ../data/38-n250/output.log 2>&1 &


#39-200
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/39-test_n200 > ../data/39-test_n200/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/39-n200 > ../data/39-n200/output.log 2>&1 &

#39-250
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/39-test_n250 > ../data/39-test_n250/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/39-n250 > ../data/39-n250/output.log 2>&1 &

#39-300
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/39-test_n300 > ../data/39-test_n300/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/39-n300 > ../data/39-n300/output.log 2>&1 &

#39-350
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/39-test_n350 > ../data/39-test_n350/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/39-n350 > ../data/39-n350/output.log 2>&1 &

#39-400
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/39-test_n400 > ../data/39-test_n400/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/39-n400 > ../data/39-n400/output.log 2>&1 &

#39-450
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/39-test_n450 > ../data/39-test_n450/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/39-n450 > ../data/39-n450/output.log 2>&1 &
