import os
import numpy as np

log_name = 'test_train_1015_1.log'
f = open(log_name,"r")
line = f.readline()

r_loss_arr = []
psnr_arr = []
ssim_arr = []

while line:
    lineArr = line.split(":")
    if lineArr[0] == 'r_loss':
        substr = line.split("(")[1]
        end_index = substr.find(",")
        r_loss_arr.append( float(substr[0:end_index]) )
    elif lineArr[0] == 'psnr':
        substr = line.split("(")[1]
        end_index = substr.find(",")
        psnr_arr.append( float(substr[0:end_index]) )
    elif lineArr[0] == 'ssim':
        substr = line.split("(")[1]
        end_index = substr.find(",")
        ssim_arr.append( float(substr[0:end_index]) )

    line = f.readline()

f.close()

print('r loss mean = ', np.mean(r_loss_arr))
print('psnr mean = ', np.mean(psnr_arr))
print('ssim mean = ',np.mean(ssim_arr))

'''
r loss mean =  0.021554227941176472
psnr mean =  29.78309080882353
ssim mean =  0.9743316176470588
'''

