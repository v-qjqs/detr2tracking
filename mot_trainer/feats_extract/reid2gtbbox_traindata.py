import os
import numpy as np 
import json 
import pickle as pkl 


def func(datapath_list, reid_file_path):
    idd_start=0
    reid2gtbbox=dict()
    reidnames=os.listdir(reid_file_path)
    reidnames=[name.split('.jpg')[0] for name in reidnames]
    idd_cps=[]
    idx_dict={}
    for dataid,datapath in enumerate(datapath_list):
        gttxt_path=datapath+'/gt/gt.txt'
        img_path=datapath+ '/img1/'
        imgfilenames=os.listdir(img_path)
        imgfilenames=[name.split('.jpg')[0] for name in imgfilenames]
        imgfilenames=[name[2:] for name in imgfilenames]
        
        with open(gttxt_path, 'r') as f:
            lines = f.readlines()
            assert len(lines)>0
        for ii,line in enumerate(lines):
            if dataid<5:
                frameid,idd_,bbox_l,bbox_t,bbox_w,bbox_h,score,classid,vis=line.split(',')
                assert int(classid) > 0 and int(idd_)>0
                if int(score)==0 or int(classid)>2 or float(vis)<0.2:
                    if ii==len(lines)-1 and len(idd_cps)>0:
                        idd_start=max(idd_cps)
                        idx_dict[dataid]=idd_start
                    continue
            else:
                frameid,idd_,bbox_l,bbox_t,bbox_w,bbox_h,score,_,_,_=line.split(',')
                assert score in [str(0), str(1)]
                assert int(idd_)>0
                if int(score)==0:
                    if ii==len(lines)-1 and len(idd_cps)>0:
                        idd_start =max(idd_cps)
                        idx_dict[dataid]=idd_start
                    continue
            frameid='0'*(4-len(frameid))+frameid
            idd_cp = idd_start+int(idd_)
            idd=str(idd_cp)
            idd = '0'*(4-len(idd))+idd
            idd_cps.append(int(idd))
            assert frameid in imgfilenames
            dataidname='0'*(3-len(str(dataid+1)))+str(dataid+1)
            
            reidname=idd+'_c2_f{}{}'.format(dataidname,frameid)
            if dataid==8:
                print('ii={} finished.++++++'.format(ii))
            assert reidname in reidnames, '{} vs {} {}'.format(reidname, reidnames[-1], idd)
            assert reidname not in reid2gtbbox
            if dataid<5:
                reid2gtbbox[reidname]=tuple([(bbox_l,bbox_t,bbox_w,bbox_h),score,classid,vis])
            else:
                reid2gtbbox[reidname]=tuple([(bbox_l,bbox_t,bbox_w,bbox_h),score])
                        
            if ii==len(lines)-1:
                name1=[name for name in reid2gtbbox.keys() if 'f{}'.format(dataidname) in name]
                name2=[name for name in reidnames if 'f{}'.format(dataidname) in name]
                assert len(name1)==len(name2)
                assert set(name1)==set(name2)
                assert len(set(name1).difference(set(name2)))==0
                assert len((set(name2).difference(set(name1))))==0
                idd_start=max(idd_cps)
                idx_dict[dataid]=idd_start
            
        print('dataid={} finished. idx_dict={}'.format(dataid+1, idx_dict))
            
    reid2gtbbox_names=reid2gtbbox.keys()
    assert len(set(reid2gtbbox_names).difference(set(reidnames)))==0
    assert len(set(reidnames).difference(set(reid2gtbbox_names)))==0


if __name__ == "__main__":
    root='/mnt/truenas/scratch/lqf/data/'
    reid_file_path=root+'/reid/dukemtmc-reid/DukeMTMC-reID/bounding_box_train/'
    mot17_root=root+'/mot/MOT17/train/'
    mot15_root=root+'/mot/2DMOT2015/train/'
    mot17_data=['MOT17-02-FRCNN','MOT17-05-FRCNN','MOT17-09-FRCNN','MOT17-10-FRCNN','MOT17-13-FRCNN']
    mot15_data=['KITTI-17','ETH-Sunnyday','ETH-Bahnhof','PETS09-S2L1','TUD-Stadtmitte']
    mot17_datapath_list=[mot17_root+dataname for dataname in mot17_data]
    mot15_datapath_list=[mot15_root+dataname for dataname in mot15_data]
    print(mot17_datapath_list+mot15_datapath_list)
    func(mot17_datapath_list+mot15_datapath_list, reid_file_path)