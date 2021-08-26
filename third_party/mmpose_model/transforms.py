import cv2
import numpy as np




def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0

    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def flip_back(output_flipped, target_type='GaussianHeatmap'):

    assert output_flipped.ndim == 4, \
        'output_flipped should be [batch_size, num_keypoints, height, width]'
    shape_ori = output_flipped.shape
    channels = 1
    if target_type.lower() == 'CombinedTarget'.lower():
        channels = 3
        output_flipped[:, 1::3, ...] = -output_flipped[:, 1::3, ...]
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels,
                                            shape_ori[2], shape_ori[3])
    output_flipped_back = output_flipped.copy()

    # Swap left-right parts
    flip_pairs = [[5,6],[7,8],[9,10],[11,12]]
    for left, right in flip_pairs:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    # Flip horizontally
    output_flipped_back = output_flipped_back[..., ::-1]
    output_flipped_back[:, :, :, 1:] = output_flipped_back[:, :, :, :-1]
    return output_flipped_back


##########################################################################
def trans_affine(img, center, scale, rotation, size):
    trans = get_affine_transform(center, scale, rotation, size)
    img = cv2.warpAffine(
        img,
        trans, size,
        flags=cv2.INTER_LINEAR)
    return img

def trans_reshape(img):
    img = img.astype(np.float16)
    img = img.transpose(2,0,1)
    img = img/255
    return img

def trans_normalize(img, mean, std):
    img = ((img.transpose()-np.array(mean))/std).transpose()
    return img

def trans_expand(img):
    img = np.expand_dims(img, axis=0)
    return img
######################################################################

def reformCoord(coords, bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2]) - x
    h = int(bbox[3]) - y

    assert w > 0 and h > 0

    fx = w/h
    fy = h/w

    if h > w:
        w_new = int(256 * fx)
        pad = int((256-w_new)/2)
        coords[:,0] -= pad
        coords = np.multiply(coords, [h/256, h/256, 1])
        coords = np.add(coords, [x, y, 0])

    if w > h:
        h_new = int(256 * fy)
        pad = int((256-h_new)/2)
        coords[:, 1] -= pad
        coords = np.multiply(coords, [w / 256, w / 256, 1])
        coords = np.add(coords, [x, y, 0])

    return coords


def resizeData(img, bbox):
    # img (h, w, c)
    """
    ['image_file', 'center', 'scale', 'bbox', 'rotation', 'joints_3d', 'joints_3d_visible', 'dataset', 'bbox_score', 'bbox_id', 'ann_info', 'img'])
    """
    x = int(bbox[0])
    y = int(bbox[1])
    x1 = int(bbox[2])
    y1 = int(bbox[3])
    w = x1-x
    h = y1-y
    assert w>0 and h>0

    img_clipped = img[y:y + h, x:x + w]
    try:
        if h > w:
            fx = w / h
            w_new = int(256 * fx)
            pad = int((256 - w_new) / 2)
            img_resize = cv2.resize(img_clipped, dsize=(w_new, 256))
            img_pad = np.pad(img_resize, ((0, 0), (pad, 256 - w_new - pad), (0, 0)), 'constant', constant_values=0)

        else:
            fy = h / w
            h_new = int(256 * fy)
            pad = int((256 - h_new) / 2)
            img_resize = cv2.resize(img_clipped, dsize=(256, h_new))
            img_pad = np.pad(img_resize, ((pad, 256 - h_new - pad), (0, 0), (0, 0)), 'constant', constant_values=0)

        return img_pad
    except:
        return None

def compose(img, img_metas):
    # transform
    img = trans_affine(img, img_metas[0]['center'], img_metas[0]['scale'], img_metas[0]['rotation'],
                       img_metas[0]['size'])
    img = trans_reshape(img)
    img = trans_normalize(img, mean=img_metas[0]['mean'], std=img_metas[0]['std'])
    img = trans_expand(img)
    img = img.astype(np.float32)

    img_flipped = np.flip(img, 3)

    return img, img_flipped
