import os
import numpy as np
import cv2

def pic_ihc_with_mask(pic_name):
    path_pic_input = 'pics_for_paper/inputs'
    path_pic_mask = 'pics_for_paper/masks'
    path_pic_output = 'pics_for_paper/outputs'
    input = cv2.imread(os.path.join(path_pic_input, pic_name+'.jpg'))
    mask = cv2.imread(os.path.join(path_pic_mask, pic_name+'.png'))
    cv2.addWeighted(input, 0.7, mask, 0.3, 0, input)
    cv2.imwrite(os.path.join(path_pic_output, pic_name+'.png'),input)

if __name__ == '__main__':
    # pic_ihc_with_mask(pic_name='0-2D-1')
    