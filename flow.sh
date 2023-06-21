# 1.denoising
python denoising.py -f fftshift -i ../data/0-Image512/ -o ../data/1-fft/

# 2.undersampling
python denoising.py -f undersampling -i ../data/1-fft/ -o ../data/2-undersampling/

# 3. create self align area
python denoising.py -f alignArea -i ../data/1-fft/ -o ../data/3-selfAlignArea/

# 4. reconstruction
python denoising.py -f reconstruction -i ../data/2-undersampling/ -a ../data/3-selfAlignArea/ -o ../data/4-reconstruction/
