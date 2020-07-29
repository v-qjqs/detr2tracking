import os
import numpy as np 
import json 
import pickle as pkl


def func(gttxt_path, file_path, reid_file_path_list, filename_len=6):
    filenames=os.listdir(file_path)
    filenames=[name.split('.jpg')[0] for name in filenames]
    for name in filenames:
        assert len(name)==filename_len, '{} vs {}'.format(name, filename_len)
    reid_filenames=[]
    for reid_file_path in reid_file_path_list:
        filenames_ = os.listdir(reid_file_path)
        filenames_ = [name for name in filenames_ if 'reid2gtbbox' not in name]
        assert isinstance(filenames_, list)
        reid_filenames.extend(filenames_)
    reid_filenames = [filename for filename in reid_filenames if filename.split('f')[1][:3] == '002']
    reid_filenames = [filename.split('.jpg')[0] for filename in reid_filenames]
    assert len(reid_filenames) == len(set(reid_filenames))
    nb_ = len(reid_filenames)
    reid_filenames = [filename.replace('_c3', '') if 'c3' in filename else filename.replace('_c2', '') for filename in reid_filenames]
    assert len(reid_filenames)==len(set(reid_filenames))==nb_

    with open(gttxt_path, 'r') as f:
        lines = f.readlines()
    reid2gtbbox=dict()
    for line in lines:
        frameid,idd,bbox_l,bbox_t,bbox_w,bbox_h,score,classid,vis=line.split(',')
        assert int(classid) > 0
        if int(score)==0 or int(classid)>2 or float(vis)<0.2:
            continue
        assert isinstance(frameid, str) and isinstance(idd, str)
        filename='0'*(filename_len-len(frameid))+frameid
        assert filename in filenames, '{} vs {}'.format(filename, filenames)
        frameid='0'*(4-len(frameid))+frameid
        name='0'*(4-len(idd))+str(idd)+'_f002'+frameid
        assert name in reid_filenames, '{} vs {}'.format(name, reid_filenames[0])
        reid2gtbbox[name]=tuple([(int(bbox_l), int(bbox_t), int(bbox_w), int(bbox_h)), int(classid), float(vis)])
    assert len(set(reid_filenames).difference(set(reid2gtbbox.keys()))) == 0
    assert len(set(reid2gtbbox.keys()).difference(set(reid_filenames))) == 0
    with open(reid_file_path+'reid2gtbbox.pkl', 'wb') as fid:
        pkl.dump(reid2gtbbox, fid)


# 11
def func2(gttxt_path, file_path, reid_file_path_list, maxidd_before, filename_len=6):
    filenames=os.listdir(file_path)
    filenames=[name.split('.jpg')[0] for name in filenames]
    print('len dataset: ', len(filenames))
    for name in filenames:
        assert len(name)==filename_len, '{} vs {}'.format(name, filename_len)
    reid_filenames=[]
    for reid_file_path in reid_file_path_list:
        filenames_ = os.listdir(reid_file_path)
        filenames_ = [name for name in filenames_ if 'reid2gtbbox' not in name]
        filenames_ = [name for name in filenames_ if 'reid2gtbbox' not in name]
        assert isinstance(filenames_, list)
        reid_filenames.extend(filenames_)
    reid_filenames = [filename for filename in reid_filenames if filename.split('f')[1][:3] == '001']
    print('len reid_filenames: ', len(reid_filenames))
    reid_filenames = [filename.split('.jpg')[0] for filename in reid_filenames]
    assert len(reid_filenames) == len(set(reid_filenames))
    nb_ = len(reid_filenames)
    reid_filenames = [filename.replace('_c3', '') if 'c3' in filename else filename.replace('_c2', '') for filename in reid_filenames]
    assert len(reid_filenames)==len(set(reid_filenames))==nb_

    with open(gttxt_path, 'r') as f:
        lines = f.readlines()
    reid2gtbbox=dict()
    for line in lines:
        frameid,idd,bbox_l,bbox_t,bbox_w,bbox_h,score,classid,vis=line.split(',')
        assert int(classid) > 0
        if int(score)==0 or int(classid)>2 or float(vis)<0.2:
            continue
        assert isinstance(frameid, str) and isinstance(idd, str)
        assert int(idd)>0
        idd = str(int(idd) + maxidd_before)
        filename='0'*(filename_len-len(frameid))+frameid
        assert filename in filenames, '{} vs {}'.format(filename, filenames)
        frameid='0'*(4-len(frameid))+frameid
        name='0'*(4-len(idd))+str(idd)+'_f001'+frameid
        assert name in reid_filenames, '{} vs {}'.format(name, reid_filenames[0])
        reid2gtbbox[name]=tuple([(str(bbox_l), int(bbox_t), int(bbox_w), int(bbox_h)), int(classid), float(vis)])
    assert len(set(reid_filenames).difference(set(reid2gtbbox.keys()))) == 0
    assert len(set(reid2gtbbox.keys()).difference(set(reid_filenames))) == 0
    with open(reid_file_path+'reid2gtbbox_part2.pkl', 'wb') as fid:
        pkl.dump(reid2gtbbox, fid)


if __name__ == "__main__":
    root='/mnt/truenas/scratch/lqf/data/'
    gttxt_path='mot/MOT17/train/MOT17-04-FRCNN/gt/gt.txt'
    file_path='/mot/MOT17/train/MOT17-04-FRCNN/img1/'
    reid_file_path_list=[root+'/reid/dukemtmc-reid/DukeMTMC-reID/query/', root+'/reid/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/']
    # func(root+gttxt_path, root+file_path, reid_file_path_list)

    gttxt_path='mot/MOT17/train/MOT17-11-FRCNN/gt/gt.txt'
    file_path='/mot/MOT17/train/MOT17-11-FRCNN/img1/'
    with open(reid_file_path_list[1]+'reid2gtbbox.pkl', 'rb') as fid:
        info_dict=pkl.load(fid)
        filenames=info_dict.keys()
        maxidd_before = max([int(filename.split('_f')[0]) for filename in filenames])
        print('maxidd_before: ', maxidd_before)  # NOTE 141
    func2(root+gttxt_path, root+file_path, reid_file_path_list, maxidd_before)