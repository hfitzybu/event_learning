import numpy as np
import h5py
import cv2
import os
import torch
def loa():



    print('sdss')

    path='h5/INI_Caltech256_10fps_20160424.hdf5'
    f = h5py.File(path, "r")  # mode = {'w', 'r', 'a'}

    # Print the keys of groups and datasets under '/'.
    print(f.filename, ":")
    print([key for key in f.keys()], "\n")

    # ===================================================
    # Read dataset 'dset' under '/'.
    d = f["001.ak47"]
    print(d.name, ":")
    print(type(f))
    print(type(d))

    print([key for key in d.keys()], "\n")


    j=d['001_0001']
    print(type(j))
    print([key for key in j.keys()], "\n")
    w=j['pol']
    print(type(w))

    print(w)

    # Print the data of 'dset'.
    # print(d[:])

    print(d.attrs.keys())
    # Print the attributes of dataset 'dset'.
    for key in d.attrs.keys():
        print(key, ":", d.attrs[key])

    print()



    return

def deh5():

    path='h5/INI_Caltech256_10fps_20160424.hdf5'

    # file
    f = h5py.File(path, "r")  # mode = {'w', 'r', 'a'}

    # caltech 256格式

    #

    print(f.filename, ":")
    print([key for key in f.keys()], "\n")

    ak47 = f["001.ak47"]
    print(ak47.name, ":")
    print("type(f)",type(f))

    print("type(ak47)",type(ak47))


    # print([key for key in ak47.keys()], "\n")

    set=ak47['001_0001']
    print("type(set)",type(set))
    print([key for key in set.keys()], "\n")
    pol=set['pol']
    timestamps=set['timestamps']
    x_pos=set['x_pos']
    y_pos=set['y_pos']

    print("type(pol)",type(pol))
    # print(pol)
    pol = np.array(pol[:])
    timestamps = np.array(timestamps[:])
    x_pos = np.array(x_pos[:])
    y_pos = np.array(y_pos[:])

    # print(pol.shape)
    # print(timestamps.shape)
    # print(x_pos.shape)
    # print(y_pos.shape)
    #
    # print(pol)
    # print(timestamps)
    # print(x_pos)
    # print(y_pos)
    return


def numpy2video(array, fps, H, W, save_path='test.mp4'):


    frame=array.shape[0]
    print(frame)

    fourcc=cv2.VideoWriter_fourcc(*'mp42')

    frame=array.shape[0]
    video=cv2.VideoWriter(save_path,fourcc,fps,(H,W))

    for i in range(frame):
        video.write(array[i])
    video.release()

    return

def save_np():


    return

def caltech():

    path='h5/INI_Caltech256_10fps_20160424.hdf5'

    # file
    f = h5py.File(path, "r")  # mode = {'w', 'r', 'a'}

    # caltech 256格式

    #

    print(f.filename, ":")
    print([key for key in f.keys()], "\n")

    ak47 = f["001.ak47"]
    print(ak47.name, ":")
    print("type(f)",type(f))

    print("type(ak47)",type(ak47))


    # print([key for key in ak47.keys()], "\n")

    set=ak47['001_0001']
    print("type(set)",type(set))
    print([key for key in set.keys()], "\n")
    pol=set['pol']
    timestamps=set['timestamps']
    x_pos=set['x_pos']
    y_pos=set['y_pos']

    print("type(pol)",type(pol))
    pol = np.array(pol[:])
    timestamps = np.array(timestamps[:])
    x_pos = np.array(x_pos[:])
    y_pos = np.array(y_pos[:])

    print(pol.shape)
    print(timestamps)

    step_num=timestamps.shape[0]
    time_max=timestamps[-1]
    frame=240
    fps=10

    step_per_frame=int(time_max/frame)

    # v=np.zeros((frame,256,256,1),dtype=np.float)
    # v=v+0.5

    v=np.zeros((frame,256,256),dtype=np.uint8)
    v=v+128

    # for i in range(frame):
    #
    #     for step in range(step_per_frame):


    for step in range(step_num):

        frame_index=int(float(timestamps[step])/float(time_max))

        if pol[step]:
            v[frame_index,x_pos[step],y_pos[step]]=255
        else:
            v[frame_index,x_pos[step],y_pos[step]]=0

    # print(v[])

    # v=int(v*255)
    print(np.max(v))
    print(np.min(v))

    numpy2video(v,fps,256,256)




    # cv2.imshow('dsds',v[0])
    # cv2.waitKey(0)
    return

def caltech2():

    path='h5/INI_Caltech256_10fps_20160424.hdf5'

    # file
    f = h5py.File(path, "r")  # mode = {'w', 'r', 'a'}

    # caltech 256格式

    #

    print(f.filename, ":")
    print([key for key in f.keys()], "\n")

    ak47 = f["001.ak47"]
    print(ak47.name, ":")
    print("type(f)",type(f))

    print("type(ak47)",type(ak47))


    # print([key for key in ak47.keys()], "\n")

    set=ak47['001_0001']
    print("type(set)",type(set))
    print([key for key in set.keys()], "\n")
    pol=set['pol']
    timestamps=set['timestamps']
    x_pos=set['x_pos']
    y_pos=set['y_pos']

    print("type(pol)",type(pol))
    pol = np.array(pol[:])
    timestamps = np.array(timestamps[:])
    x_pos = np.array(x_pos[:])
    y_pos = np.array(y_pos[:])

    print(pol.shape)
    print(timestamps)

    step_num=timestamps.shape[0]
    print('step num',step_num)
    time_max=timestamps[-1]
    frame=240
    fps=24

    step_per_frame=time_max/frame

    # v=np.zeros((frame,256,256,1),dtype=np.float)
    # v=v+0.5

    v=np.zeros((frame,256,256),dtype=np.uint8)
    v=v+128

    # for i in range(frame):
    #
    #     for step in range(step_per_frame):

    for step in range(step_num):
        # print('----')
        frame_index=int(float(timestamps[step]-1)/float(step_per_frame))
        # print(step)
        # print(frame_index)
        # print(timestamps[step]-1)
        # print(step_per_frame)



        if pol[step]:
            v[frame_index,x_pos[step],y_pos[step]]=255
        else:
            v[frame_index,x_pos[step],y_pos[step]]=0

    # print(v[])

    # v=int(v*255)

    # xx=np.zeros_like(v)
    # xx=np.zeros((100,250,250),dtype=np.uint8)
    xx=np.zeros((100,256,256),dtype=np.uint8)

    print('v',v.shape)
    print('xx',xx.shape)

    # xx=np.zeros((100,256,256),dtype=np.uint8)

    # numpy2video(xx,fps,256,256)
    numpy2video(v,fps,256,256)


    # cv2.imshow('dsds',v[0]/255)
    # cv2.waitKey(0)
    return

def caltech_decode_all():

    path='h5/INI_Caltech256_10fps_20160424.hdf5'
    f = h5py.File(path, "r")  # mode = {'w', 'r', 'a'}
    print(f.filename, ":")
    print([key for key in f.keys()], "\n")

    # train set (class--index,frame,H,W)

    for key in f.keys():
        # class label

        item=f[key]
        # print(item.name)
        for index in item.keys():
            set=item[index]
            # print(set.name)
            pol = set['pol']
            timestamps = set['timestamps']
            x_pos = set['x_pos']
            y_pos = set['y_pos']

            print("type(pol)", type(pol))
            pol = np.array(pol[:])
            timestamps = np.array(timestamps[:])
            x_pos = np.array(x_pos[:])
            y_pos = np.array(y_pos[:])

            print(pol.shape)
            print(timestamps)

            step_num = timestamps.shape[0]
            print('step num', step_num)
            time_max = timestamps[-1]
            frame = 240
            fps = 24
            step_per_frame = time_max / frame


            # 三段 0 128 255
            v = np.zeros((frame, 256, 256), dtype=np.uint8)
            v = v + 128

            # 三段 -1 0 1
            nparrary = np.zeros((frame, 256, 256), dtype=np.int)
            # v = v + 128
            for step in range(step_num):
                frame_index = int(float(timestamps[step] - 1) / float(step_per_frame))

                if pol[step]:
                    v[frame_index, x_pos[step], y_pos[step]] = 255
                    nparrary[frame_index, x_pos[step], y_pos[step]] = 1

                else:
                    nparrary[frame_index, x_pos[step], y_pos[step]] = -1


            # folder = key+'/'+index+'.mp4'
            root='dataset/caltech/np/'
            folder=root+key
            print(folder)
            if os.path.isdir(folder):
                print('True')
            else:
                os.mkdir(folder)

            np_path=root+set.name+'.npy'

            np.save(file=np_path,arr=nparrary)
            # numpy2video(v,fps,256,256,save_path=save_path)

    return


def swin_t():

    from swin_transformer import SwinTransformer
    st=SwinTransformer()

    return

def test_scnn():

    # test input path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x=np.load('dataset/caltech/np/001.ak47/001_0001.npy')

    # batch size time_step/channel H and W
    x=x.reshape(1,240,256,256)
    # x=x[:,0,:,:].reshape(1,256,256,1)
    # x=x.reshape([1,240,256,256,1])
    x=torch.from_numpy(x).to(device)
    print(x.shape)
    from scnn import SCNN

    # cnn_layer(in_planes(channels), out_planes(channels), kernel_size, stride, padding)
    # cfg_cnn = [(1, 16, 3, 2, 1),
    #            (16, 32, 3, 2, 1),
    #            (32, 64, 3, 2, 1),
    #            (64, 64, 3, 2, 1), ]

    cfg_cnn = [(1, 16, 3, 2, 1),
               (16, 32, 3, 2, 1),
               (32, 64, 3, 2, 1),]
    scnn = SCNN(cfg_cnn=cfg_cnn)


    import torch.nn as nn

    target=torch.zeros([1,10]).to(device)
    target[0,0]=1
    # target[1,9]=1

    print('target', target)
    y = scnn.forward(input=x, time_window=24)
    print(y.shape)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(scnn.parameters(), lr=1e-2, momentum=0.9)
    # for i in range(100):
    #     y = scnn.forward(input=x, time_window=24)
    #
    #     loss = criterion(y, target)
    #     loss.backward(retain_graph=True)
    #     print('loss',loss)
    #     optimizer.step()
    return

def spiking_conv():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x=np.load('dataset/caltech/np/001.ak47/001_0001.npy')

    # batch size time_step/channel H and W
    x=x.reshape(1,240,256,256)
    # x=x[:,0,:,:].reshape(1,256,256,1)
    # x=x.reshape([1,240,256,256,1])
    x=torch.from_numpy(x).to(device)
    print(x.shape)
    from spiking_conv import SCNN

    # cnn_layer(in_planes(channels), out_planes(channels), kernel_size, stride, padding)

    cfg_cnn = [(1, 16, 3, 2, 1),
               (16, 32, 3, 2, 1),
               (32, 64, 3, 2, 1),]
    scnn = SCNN(cfg_cnn=cfg_cnn)


    import torch.nn as nn

    target=torch.zeros([1,10]).to(device)
    target[0,0]=1
    # target[1,9]=1

    print('target', target)
    y = scnn.forward(input=x, time_window=24)
    print(y.shape)
    return

def testvit():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x=np.load('dataset/caltech/np/001.ak47/001_0001.npy')
    print(x.shape)
    # from vit_pytorch import ViT
    from vit import ViT
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        channels=64
    )

    img = torch.randn(1, 64, 256, 256)

    preds = v(img)  # (1, 1000)
    print(preds.shape)


    return

def testStrans():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x=np.load('dataset/caltech/np/001.ak47/001_0001.npy')
    print(x.shape)
    x=x.reshape(1,240,256,256)

    x=torch.from_numpy(x).to(device)
    print('x shape' , x.shape)

    cfg_cnn = [(1, 16, 3, 2, 1),
               (16, 32, 3, 2, 1),
               (32, 64, 3, 2, 1),]
    # from vit_pytorch import ViT
    from event_hard_attention import strans

    st=strans(cfg_cnn=cfg_cnn)

    y=st(x)


    # batch size time_step/channel H and W


    return


def testvis():

    from visevent import main
    main()

    return

def test_vis():
    from visevent_load import read
    read()

    return