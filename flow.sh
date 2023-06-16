# 1.denoising
python denoising.py -f fftshift -i ../data/0-Image512/ -o ../data/1-fft/

# 2.undersampling
python denoising.py -f undersampling -i ../data/1-fft/ -o ../data/2-undersampling/
