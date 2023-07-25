# whole process
nohup python denoising.py -f whole -i ../data/0-Image512/ -o ../data/ &

#test
python denoising.py -f whole -i ./testin -o ./testout
python denoising.py -f whole -i ../data/test_input/ -o ../data/test_out

