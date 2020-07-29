import cv2
import pandas as pd 
import os 
import time 
import colorsys
TRACKING_OUT_COLS = ['frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
# TRACKING_OUT_COLS = ['frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'cls', 'conf', 'vis']  # GT


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b

def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)

def rectangle(image, x, y, w, h, _color, label=None, thickness=2):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.

    """
    img_h, img_w = image.shape[:-1]
    pt1 = max(int(x),0), max(int(y),0)
    pt2 = min(int(x + w), img_w-1), min(int(y + h), img_h-1)
    cv2.rectangle(image, pt1, pt2, _color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)
        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        cv2.rectangle(image, pt1, pt2, _color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness)
    return image

def vis_func(seq_name, seq_path, imgs_path, out_dir, window_shape=(640, 480)):
    df = pd.read_csv(os.path.join(seq_path, seq_name+'.txt'), header=None)
    df.columns = TRACKING_OUT_COLS
    frameids = df['frame'].unique()
    print('frameis: ', frameids.shape)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for frame in frameids:
        img_path = os.path.join(imgs_path, 'img1', '0'*(6-len(str(frame)))+str(frame)+'.jpg')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # det
        track = df[df['frame'].eq(frame)].copy()
        ped_ids, bb_lefts, bb_tops, bb_widths, bb_heights = track['ped_id'].values, track['bb_left'].values, track['bb_top'].values, track['bb_width'].values, track['bb_height'].values
        for ped_id, bb_left, bb_top, bb_width, bb_height in zip(ped_ids, bb_lefts, bb_tops, bb_widths, bb_heights):
            color = create_unique_color_uchar(ped_id)
            image = rectangle(image, bb_left, bb_top, bb_width, bb_height, color, label=str(ped_id), thickness=2)
        cv2.imwrite(os.path.join(out_dir, 'frame{}.jpg'.format(frame)), cv2.resize(image, window_shape[:2]))


if __name__ == "__main__":
    seq_name = 'MOT17-02-FRCNN'
    root = '/mnt/truenas/scratch/lqf/data/mot/MOT17/train/'
    # seq_path = root + 'res_delete4/'
    seq_path = root + 'res_delete_train/'    
    # imgs_path = root + 'MOT17-02-FRCNN/'
    imgs_path = os.path.join(root, seq_name)
    out_dir = '/mnt/truenas/scratch/lqf/code/tmp/detr2tracking/mot_tracker/vis/'
    vis_func(seq_name, seq_path, imgs_path, out_dir=out_dir+'/vis_my',  window_shape=(1920, 1080))  #  window_shape=(960, 720)  (1440, 1080)