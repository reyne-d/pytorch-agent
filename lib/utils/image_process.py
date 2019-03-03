import cv2
import numpy as np


def warp_image_with_text(img, info_list,tag=None,is_scale=False):
    # img format -> BGR
    if info_list is None or len(info_list)==0:
        return img
    assert isinstance(info_list, list) and all([isinstance(info, str) for info in info_list]), \
        "None or list of string are expected in argument2 $info_list, got {}".format(info_list)

    assert isinstance(info_list ,list)
    h,w,c = img.shape

    # if image size is [600,600], use font_scale=1 and thickness=1, delta_h=25 is suitable
    base_img_size = 600
    base_delta_h = 36
    base_offset_h = 10
    base_offset_w = 30
    base_thickness_scale = 1

    scale = (h / base_img_size)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if is_scale:
        pad_values = 1
        color=(1,0,0)
    else:
        pad_values = 255
        color = (255, 0, 0)

    thickness = int(np.max([scale, 1])) * base_thickness_scale

    w_offset = int(scale * base_offset_w)
    h_offset = int(scale * base_offset_h)
    delta_h = int(scale * base_delta_h)
    pad_h_bottom = int(3 * h_offset + len(info_list) * base_delta_h * scale)

    if tag is not None:
        pad_h_top = int(3*h_offset + base_delta_h*scale)
        img = np.pad(img, ((pad_h_top, pad_h_bottom), (0, 0), (0, 0)),
                     mode='constant', constant_values=pad_values)

        img = cv2.putText(img, tag, (w_offset, h_offset+delta_h) ,
                          fontFace=font, fontScale=scale, color=color,
                          thickness=thickness, lineType=cv2.LINE_AA)
        h_offset += pad_h_top
    else:
        img = np.pad(img, ((0, pad_h_bottom), (0, 0), (0, 0)),
                     mode='constant', constant_values=pad_values)

    h_offset += h
    for i, text in enumerate(info_list,1):
        img = cv2.putText(img, text, (w_offset, h_offset+i*delta_h),
                          fontFace=font, fontScale=scale, color=color,
                          thickness=thickness, lineType=cv2.LINE_AA)
    return img


def format_str(k,v):
    if isinstance(v, (float, np.float32,np.float64,np.float16,np.float128)):
            st = '{} : {:.4f}'.format(k, v)
    else:
        st = '{} : {}'.format(k, v)
    return st


def add_text_in_img(imgs, print_dic, is_scale=True):
    assert isinstance(imgs, np.ndarray), 'nd.ndarray are expected ,got {}'.format(type(imgs))
    assert all(imgs.shape[0] == len(data) for data in print_dic.values()), \
        "imgs and print_dic should have same length"

    imgs = imgs.transpose((0,2,3,1))

    ret = []
    tags = print_dic.get('tag', None)
    if tags is not None:
        del print_dic['tag']
    for i in range(imgs.shape[0]):
        img = imgs[i]
        info = [format_str(k, v[i]) for k, v in print_dic.items()]
        tag = tags[i] if tags is not None else None
        warp_img = warp_image_with_text(img, info, tag, is_scale=is_scale)
        ret.append(warp_img)
    ret = np.array(ret).transpose((0,3,1,2))
    return ret


if __name__=='__main__':
    test = (np.random.rand(3,3,224,224)*255).astype(np.uint8)
    print_dic = {
        'tag': ['xxxx','yyyy','zzzz'],
        'info1': [0.555555,0.698444,0.44444],
        'info2': [443543, 2434, 41]
    }
    print(print_dic)
    print("test.shape: {}".format(test.shape))
    imgs = add_text_in_img(test, print_dic)
    print("return imgs.shape :{}".format(imgs.shape))
    import torchvision.utils as vutils
    import torch
    imgs = torch.Tensor(imgs)
    imgs = vutils.make_grid(imgs)
    print(imgs.size())
    cv2.imwrite('test_add_imgs.png', imgs.numpy().transpose(1,2,0))
