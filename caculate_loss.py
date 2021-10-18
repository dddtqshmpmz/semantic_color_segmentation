import os
import numpy as np

log_name = 'test_train_1015_1.log'
f = open(log_name,"r")
line = f.readline()

total_loss_arr = []
r_loss_arr = []
m_loss_arr = []
d_loss_arr = []
psnr_arr = []
ssim_arr = []

while line:
    lineArr = line.split(":")
    lineArr2 = line.split(" ")
    if lineArr[0] == 'r_loss':
        substr = line.split("(")[1]
        end_index = substr.find(",")
        r_loss_arr.append( float(substr[0:end_index]) )

    elif lineArr[0] == 'm_loss':
        substr = line.split("(")[1]
        end_index = substr.find(",")
        m_loss_arr.append( float(substr[0:end_index]) )

    elif lineArr[0] == 'd_loss':
        substr = line.split("(")[1]
        end_index = substr.find(",")
        d_loss_arr.append( float(substr[0:end_index]) )

    elif lineArr[0] == 'psnr':
        substr = line.split("(")[1]
        end_index = substr.find(",")
        psnr_arr.append( float(substr[0:end_index]) )

    elif lineArr[0] == 'ssim':
        substr = line.split("(")[1]
        end_index = substr.find(",")
        ssim_arr.append( float(substr[0:end_index]) )
    
    if lineArr2[0] == 'total_loss':
        substr = line.split("(")[1]
        end_index = substr.find(",")
        total_loss_arr.append( float(substr[0:end_index]) )

    line = f.readline()

f.close()
print('total_loss mean = ',np.mean(total_loss_arr))
print('r_loss mean = ', np.mean(r_loss_arr))
print('m_loss mean = ', np.mean(m_loss_arr))
print('d_loss mean = ', np.mean(d_loss_arr))
print('psnr mean = ', np.mean(psnr_arr))
print('ssim mean = ',np.mean(ssim_arr))

'''
total_loss mean =  0.053910477941176475
r_loss mean =  0.021554595588235293
m_loss mean =  0.030494485294117645
d_loss mean =  0.0018625000000000002
psnr mean =  29.7957875
ssim mean =  0.9743294117647059
'''

