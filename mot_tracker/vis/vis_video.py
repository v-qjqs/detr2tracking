import cv2
import pandas as pd 
import os 
import time 
import colorsys
# TRACKING_OUT_COLS = ['frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
TRACKING_OUT_COLS = ['frame', 'ped_id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'cls', 'conf', 'vis']  # for GT
from vis import create_unique_color_uchar, rectangle
DEFAULT_UPDATE_MS = 20


def enable_videowriter(output_filename, fourcc_string="MJPG", window_shape=(640, 480), _update_ms=None, fps=None):
    """ Write images to video file.
    Parameters
    ----------
    output_filename : str
        Output filename.
    fourcc_string : str
        The OpenCV FOURCC code that defines the video codec (check OpenCV
        documentation for more information).
    fps : Optional[float]
        Frames per second. If None, configured according to current
        parameters.
    """
    fourcc = cv2.VideoWriter_fourcc(*fourcc_string)
    if fps is None:
        if _update_ms is None:
            _update_ms = DEFAULT_UPDATE_MS
        fps = int(1000. / _update_ms)
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, window_shape)
    return video_writer

def vis_func(seq_name, seq_path, imgs_path, out_dir, window_shape=(640, 480)):
    df = pd.read_csv(os.path.join(seq_path, 'gt.txt'), header=None)  # GT
    df.columns = TRACKING_OUT_COLS
    frameids = df['frame'].unique()
    # print('frameis: ', frameids.shape)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    video_filename = os.path.join(out_dir, "%s.avi" % seq_name)
    video_writer = enable_videowriter(video_filename, fourcc_string="MJPG", window_shape=window_shape, _update_ms=None, fps=30)
    for frame in frameids:
        img_path = os.path.join(imgs_path, 'img1', '0'*(6-len(str(frame)))+str(frame)+'.jpg')
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # det
        track = df[df['frame'].eq(frame)].copy()
        ped_ids, bb_lefts, bb_tops, bb_widths, bb_heights = track['ped_id'].values, track['bb_left'].values, track['bb_top'].values, track['bb_width'].values, track['bb_height'].values
        for ped_id, bb_left, bb_top, bb_width, bb_height in zip(ped_ids, bb_lefts, bb_tops, bb_widths, bb_heights):
            color = create_unique_color_uchar(ped_id)
            image = rectangle(image, bb_left, bb_top, bb_width, bb_height, color, label=str(ped_id), thickness=1)  # NOTE
        video_writer.write(cv2.resize(image, window_shape[:2]))


if __name__ == "__main__":
    seq_name = 'MOT17-02-FRCNN'
    root = '/mnt/truenas/scratch/lqf/data/mot/MOT17/train/'
    # seq_path = root + 'res_delete4/'
    seq_path = root + 'res_delete_train/'    
    seq_path = '/mnt/truenas/scratch/lqf/data/jiawei/data/MOT_eval_gt/MOT17-02-GT/gt/'  # GT
    # imgs_path = root + 'MOT17-02-FRCNN/'
    imgs_path = os.path.join(root, seq_name)
    out_dir = '/mnt/truenas/scratch/lqf/code/tmp/detr2tracking/mot_tracker/vis/'
    vis_func(seq_name, seq_path, imgs_path, out_dir=out_dir+'/vis_my_video_gt', window_shape=(1920, 1080))