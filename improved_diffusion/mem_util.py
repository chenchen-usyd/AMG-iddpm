import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


def L2(a, b):
       return torch.sqrt(sum((((a-b)**2).reshape(3*32*32)/(3*32*32)))) / 2
    
def imshow(img, fig_name):
    img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    plt.imshow(img.permute(1,2,0))
    plt.savefig(fig_name)

#imshow(dat[0], "dat[0].png")
    
def inspect_l2_dist(img, dat):  
    l2_list = []
    for i in range(len(dat)):
        l2_list.append(L2(img, dat[i]))
    return l2_list

#l2_list = inspect_l2_dist(dat[0], dat)
#l2_list_1 = inspect_l2_dist(dat[1], dat)
     
def vis_l2_dist(l2_list, fig_name):
    fig1, ax1 = plt.subplots()
    ax1.set_yscale('log')
    ax1.hist(l2_list, bins=80)
    plt.savefig(fig_name)

# vis_l2_dist(l2_list[1:], "vis_l2_dist.png")

def mL2(l2_list, n, alpha):
    l2_list = sorted(l2_list)
    nearest = l2_list[0]
    neighgours = l2_list[:n]
    return nearest/(alpha * np.mean(neighgours))
    
# mL2(l2_list[1:], 50, 0.5)
# mL2(sorted(l2_list_1)[1:], 50, 0.5)


