from math import factorial
import os
import numpy as np
from torch import batch_norm
from torch.nn.functional import l1_loss

def caculate_loss():
    log_name = 'test_train_1015_1.log'
    f = open(log_name,"r")
    line = f.readline()

    total_loss_arr = []
    r_loss_arr = []
    m_loss_arr = []
    d_loss_arr = []
    l_loss_arr = []
    psnr_arr = []
    ssim_arr = []
    deltaE_arr = []

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

        elif lineArr[0] == 'l_loss':
            substr = line.split("(")[1]
            end_index = substr.find(",")
            l_loss_arr.append( float(substr[0:end_index]) )

        elif lineArr[0] == 'psnr':
            substr = line.split("(")[1]
            end_index = substr.find(",")
            psnr_arr.append( float(substr[0:end_index]) )

        elif lineArr[0] == 'ssim':
            substr = line.split("(")[1]
            end_index = substr.find(",")
            ssim_arr.append( float(substr[0:end_index]) )

        elif lineArr[0] == 'delta_e':
            substr = line.split("(")[1]
            end_index = substr.find(",")
            deltaE_arr.append( float(substr[0:end_index]) )
        
        # if lineArr2[0] == 'total_loss':
        #     substr = line.split("(")[1]
        #     end_index = substr.find(",")
        #     total_loss_arr.append( float(substr[0:end_index]) )

        line = f.readline()

    f.close()
    # print('total_loss mean = ',np.mean(total_loss_arr))
    
    print('r_loss mean = ',  np.round(np.mean(r_loss_arr),7) )
    print('m_loss mean = ',   np.round(np.mean(m_loss_arr),7))
    print('d_loss mean = ', np.round(np.mean(d_loss_arr) ,7))
    print('l_loss mean = ',  np.round(np.mean(l_loss_arr),7))
    print('psnr mean = ',  np.round(np.mean(psnr_arr),7))
    print('ssim mean = ', np.round(np.mean(ssim_arr),7))
    print('delta_e mean = ', np.round(np.mean(deltaE_arr),7))


    '''
    total_loss mean =  0.053910477941176475
    r_loss mean =  0.021554595588235293
    m_loss mean =  0.030494485294117645
    d_loss mean =  0.0018625000000000002
    psnr mean =  29.7957875
    ssim mean =  0.9743294117647059
    '''

def caculate_mean_metrics_of_K_epoch():
    dir_name = '20211116'
    dir_path = os.path.join('models','multitask',dir_name,'train_process.log')
    end_epoch = 80
    k = 10
    factor = 1.0
    st_epoch = end_epoch-k+1

    f = open(dir_path,"r")
    line = f.readline()

    r_loss_arr = []
    m_loss_arr = []
    d_loss_arr = []

    while line:
        lineArr = line.split(":")
        if len(lineArr) >= 3 and lineArr[1][4:] == 'Average val reconst_loss *lambda' \
            and int(lineArr[1].split(" ")[1]) >= st_epoch and\
                 int(lineArr[1].split(" ")[1]) <= end_epoch:
            
            r_loss_arr.append(float( lineArr[2]))

        elif len(lineArr) >= 3 and lineArr[1][4:] == 'Average val mono_loss *lambda' \
            and int(lineArr[1].split(" ")[1]) >= st_epoch and\
                 int(lineArr[1].split(" ")[1]) <= end_epoch:

            m_loss_arr.append(float(lineArr[2]))

        elif len(lineArr) >= 3 and lineArr[1][4:] == 'Average val squared_mahalanobis_distance_loss *lambda' \
            and int(lineArr[1].split(" ")[1]) >= st_epoch and\
                 int(lineArr[1].split(" ")[1]) <= end_epoch:
        
            d_loss_arr.append(float(lineArr[2]))


        line = f.readline()

    f.close()
    print('r_loss mean = ', np.round(np.mean(r_loss_arr)*factor,7))
    print( 'm_loss mean = ', np.round( np.mean(m_loss_arr)*factor,7))
    print('d_loss mean = ', np.round(np.mean(d_loss_arr)*factor,7))




if __name__ == '__main__':
    caculate_loss()
    # caculate_mean_metrics_of_K_epoch()
