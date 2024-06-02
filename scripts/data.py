"""
Load full version and deduplicated version of CIFAR-10 data.
"""
import numpy as np
import torch as th

from improved_diffusion.image_datasets import load_data
data_all = load_data(
    data_dir="/root/autodl-tmp/improved-diffusion/cifar_train",
    batch_size=50000,
    image_size=32,
    class_cond=False,
    deterministic=True
)

dat, _ = next(data_all)  
th.save(dat, 'dat.pt')

def get_l2(i, dat):
    img = dat[i]
    imgs = img.repeat(len(dat),1,1,1)
    l2_tensor = th.sqrt(th.mean((imgs - dat) ** 2, (1,2,3))) / 2
    l2_list = l2_tensor.tolist()
    del l2_list[i]
    return l2_list

def NN(l2_list):
    idx = np.argmin(l2_list)
    return idx

def mL2(l2_list, n, alpha):
    l2_list = sorted(l2_list)
    nearest = l2_list[0]
    neighgours = l2_list[:n]
    return nearest/(alpha * sum(neighgours) / n)

def L2(real, fake):
    return th.sqrt(sum((((real-fake)**2).reshape(3*32*32)/(3*32*32)))) / 2

def sL2(real, fake):
    return th.sum((real-fake)**2)


def inspect_l2_dist(img, dat):  
    l2_list = []
    for i in range(len(dat)):
        l2_list.append(L2(img, dat[i]))
    return l2_list

ml2_list = []
i = 0
dat_new = dat
while i < len(dat_new):
    l2_list = get_l2(i, dat_new)
    ml2 = mL2(l2_list, 50, 0.5)
    if ml2 < 1:
        dat_new = th.cat((dat_new[0:i,:,:,:], dat_new[i+1:,:,:,:]))
    else:
        ml2_list.append(ml2)
        i = i + 1
th.save(dat_new, 'dat_new_100.pt')













     




