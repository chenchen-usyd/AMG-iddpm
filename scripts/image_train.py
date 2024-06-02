"""
Train a diffusion model on images.
"""

import argparse

from improved_diffusion import dist_util, logger, mem_util
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev()) # set model to device - gpu
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    data_all = load_data(
        data_dir=args.data_dir,
        batch_size=50000,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True
    )

    dat, cond = next(data_all)

    import torch
    def L2(a, b):
        return torch.sqrt(sum((((a-b)**2).reshape(3*32*32)/(3*32*32)))) / 2
    
    import matplotlib.pyplot as plt
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

    import numpy as np
    def mL2(l2_list, n, alpha):
        l2_list = sorted(l2_list)
        nearest = l2_list[0]
        neighgours = l2_list[:n]
        return nearest/(alpha * np.mean(neighgours))
    
    # mL2(l2_list[1:], 50, 0.5)
    # mL2(sorted(l2_list_1)[1:], 50, 0.5)

    from torch.autograd import Variable
    def cond_fn(real, fake, classifier_scale):
        real = Variable(real)
        fake = Variable(fake, requires_grad = True)
        neg_l2 = -L2(real, fake)
        neg_l2.backward()
        grad = fake.grad
        return grad * classifier_scale   









    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
