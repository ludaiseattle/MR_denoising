# whole process
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/ &
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/1-test_18000_800_0_400 > ../data/1-test_18000_800_0_400/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/37-18000_800_0_400 > ../data/37-18000_800_0_400/output.log 2>&1 &

#test
python denoising.py -f whole -i ./testin -o ./testout
python denoising.py -f whole -i ../data/test_input/ -o ../data/test_out -n 180

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



#40-150
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/40-test_n150 > ../data/40-test_n150/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/40-n150 > ../data/40-n150/output.log 2>&1 &

#40-100
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/40-test_n100 > ../data/40-test_n100/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/40-n100 > ../data/40-n100/output.log 2>&1 &

#40-50
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/40-test_n50 > ../data/40-test_n50/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/40-n50 > ../data/40-n50/output.log 2>&1 &
--------------------------------------------------------------------------------------------------------------------------------

#41-180
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n180 -n 180 > ../data/41-test_n180/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n180 -n 180 > ../data/41-n180/output.log 2>&1 &

#41-210
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n210 -n 210 > ../data/41-test_n210/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n210 -n 210 > ../data/41-n210/output.log 2>&1 &

#41-240
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n240 -n 240 > ../data/41-test_n240/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n240 -n 240 > ../data/41-n240/output.log 2>&1 &

#41-270
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n270 -n 270 > ../data/41-test_n270/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n270 -n 270 > ../data/41-n270/output.log 2>&1 &

#41-300
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n300 -n 300 > ../data/41-test_n300/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n300 -n 300 > ../data/41-n300/output.log 2>&1 &

#41-330
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n330 -n 330 > ../data/41-test_n330/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n330 -n 330 > ../data/41-n330/output.log 2>&1 &

#41-360
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n360 -n 360 > ../data/41-test_n360/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n360 -n 360 > ../data/41-n360/output.log 2>&1 &

#41-390
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n390 -n 390 > ../data/41-test_n390/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n390 -n 390 > ../data/41-n390/output.log 2>&1 &

#41-420
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n420 -n 420 > ../data/41-test_n420/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n420 -n 420 > ../data/41-n420/output.log 2>&1 &

#41-450
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n450 -n 450 > ../data/41-test_n450/output.log 2>&1 &
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/41-n450 -n 450 > ../data/41-n450/output.log 2>&1 &

#41-18
nohup python denoising.py -f whole -i ../data/1-test/ -o ../data/41-test_n18 -n 18 > ../data/41-test_n18/output.log 2>&1 &
