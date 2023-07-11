# whole process
python denoising.py -f whole -i ../data/0-Image512/ -o ../data/1-whole_out

#test
python denoising.py -f whole -i ./testin -o ./testout
python denoising.py -f whole -i ../data/test_input/ -o ../data/test_out

