import cv2
def letterbox(x):
    height,width = x.shape[0:2]
    new_width = (width//32+1)*32
    new_height = (height // 32 + 1) * 32
    width_pad=new_width-width
    height_pad = new_height - height
    return  np.pad(x,((0,height_pad),(0,width_pad),(0,0)))
