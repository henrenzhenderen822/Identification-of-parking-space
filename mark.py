
'''此程序用来制作掩膜（得到停车位的坐标保存为csv文件）'''

import pandas as pd
import cv2
import numpy as np


# 读取需要操作的图片
image = cv2.imread('cameras/baidu1/baidu1.jpg')
# 填写需要保存的文件名称
filename = 'cameras/baidu2/baidu2.csv'

current_pos = None
tl = None
br = None


# 鼠标事件
def get_rect(im, title='get_rect'):  # (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
                    'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                          mouse_params['current_pos'], (255, 0, 0))
        cv2.imshow(title, im_draw)
        _ = cv2.waitKey(10)

    # tl=(y1,x1), br=(y2,x2)
    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
          min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
          max(mouse_params['tl'][1], mouse_params['br'][1]))
    (y1, x1) = tl
    (y2, x2) = br

    key = cv2.waitKey(0)
    if key == 32:  # 按空格键继续，否则停止框选
        points.append([y1, x1, y2, x2])
        return get_rect(im_draw, title='get_rect')
    else:
        cv2.destroyAllWindows()

    return points


points = []
points = get_rect(image, title='get_rect')

title = ('x1', 'y1', 'x2', 'y2')
test = pd.DataFrame(columns=title, data=points)

# 将patch坐标保存为csv文件
# test.to_csv(filename)
