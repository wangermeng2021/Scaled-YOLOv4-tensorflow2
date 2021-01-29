
import numpy as np
import requests
import cv2
def get_frame(server_ip="10.111.126.81", monitor_id='44', max_image_num=1,alg=None):

    try:
        imageurl = 'http://'+server_ip+'/zm/cgi-bin/nph-zms?mode=single&monitor='+str(monitor_id)
        # response = requests.get(imageurl, timeout=0.01).content
        response = requests.get(imageurl).content
    except:
        return None
    nparr = np.frombuffer(response, np.uint8)
    cv_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return cv_img