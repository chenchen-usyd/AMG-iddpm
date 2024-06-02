"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

############################# load training images #############################
# If use all 50,000 training images without deduplications
#from improved_diffusion.image_datasets import load_data
#data_all = load_data(
#    data_dir="/root/autodl-tmp/improved-diffusion/cifar_train",
#    batch_size=50000,
#    image_size=32,
#    class_cond=False,
#    deterministic=True
#)

#dat, _ = next(data_all)  
#dat = dat.to("cuda:0")

# If use deduplicated training images for more precise nL2 computation (remove ones that has mL2 < 1.00)
dat = th.load("dat_new_100.pt")
dat = dat.to("cuda:0")

################################ helper functions ################################
# L2 distance
def L2(real, fake):
    return th.sqrt(sum((((real-fake)**2).reshape(3*32*32)/(3*32*32)))) / 2

# modified L2 distance
def mL2(l2_list, n, alpha):
    l2_list = sorted(l2_list)
    nearest = l2_list[0]
    neighgours = l2_list[:n]
    return nearest/(alpha * sum(neighgours) / n)

def asy(x, a, b, c):
    return a - (a-b) * np.exp(-c*x)

# get gradient for mL2 guidance
def cond_fn(real, fake, classifier_scale, l2_list):
    """
    get the scaled gradient for updating the fake images that pushes it away from the real.
    - use ml2 of the fake with its nearest real as the loss function.
    - thus for any fake img, once we have the time-corresponding reals, we can increase their distance.

    [time] ~0.0088 per call, only called when t<2000 and mL2<1.9 (average ~800 calls, about 6s/sample).
    """   
    with th.enable_grad():        
        # prepare fake image
        fake_in = fake.detach().requires_grad_(True)
        
        # compute mL2
        denom = 0.5 * sum(sorted(l2_list)[:50]) / 50
        mse_loss = th.nn.MSELoss().cuda()
        mse = mse_loss(fake_in, real)
        mse = th.reshape(mse, [1])
        l2 = th.sqrt(mse) / 2
        ml2 = l2 / denom

        # backprop and report gradient
        ml2.backward()
        grad = fake_in.grad

        return grad * classifier_scale  

####################################### main ######################################
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load pretrained diffusion model and set eval mode
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # start sampling loop
    logger.log("sampling...")
    all_images = [] # store guided images
    all_labels = [] # store guided labels
    all_images_abl = [] # store abl images
    all_labels_abl = [] # store abl labels
    ml2_a_all = [] # store abl ml2
    ml2_g_all = [] # store guided ml2

    while len(all_images) * args.batch_size < args.num_samples:
        import time
        tic = time.time()
        
        model_kwargs = {}
        
        # conditional generation
        if args.class_cond:
            #classes = th.randint(
            #    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            #)
            bscc = len(all_images) * args.batch_size
            classes = th.tensor([bscc % 10], device='cuda:0')
            model_kwargs["y"] = classes
        
        # determine sample function
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        # use sample function for sampling (with guidance)
        sample, sample_abl, ml2_g_progress, ml2_a_progress, cache_plot_g, cache_plot_a = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
        )

        ####################################### ablated sample ######################################
        # get the NN info and compute L2 and mL2
        neighbour_abl, l2list_abl, idx_abl = diffusion.nearest(sample_abl[0], dat, th.tensor([0], device="cuda"))
        l2_abl = L2(neighbour_abl, sample_abl)
        ml2_abl = mL2(l2list_abl, n=50, alpha=0.5)

        # processing for visualiation
        sample_abl = ((sample_abl + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample_abl = sample_abl.permute(0, 2, 3, 1)
        sample_abl = sample_abl.contiguous()

        gathered_samples_abl = [th.zeros_like(sample_abl) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples_abl, sample_abl)  
        all_images_abl.extend([sample_abl.cpu().numpy() for sample_abl in gathered_samples_abl])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels_abl.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images_abl) * args.batch_size} ablated samples")

        ml2_a_all.append(ml2_a_progress)
        
        # plot intermediate sampling steps
        for i in np.arange(len(cache_plot_a)):
            if i % 10 == 0:
                plt.imsave(f"sample_{len(all_images_abl) * args.batch_size}_ablated_{i}.png",cache_plot_a[len(cache_plot_a)-1-i][0].cpu().numpy().squeeze())

        plt.imsave(f"sample_{len(all_images_abl) * args.batch_size}_abl.png",sample_abl[0].cpu().numpy().squeeze())
        plt.imsave(f"sample_{len(all_images_abl) * args.batch_size}_abl_nbour.png",((neighbour_abl + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1,2,0).cpu().numpy().squeeze())
        logger.log(f"For the ablated image, the {idx_abl}th image is the nearest, with L2 of {l2_abl}, mL2 of {ml2_abl}.")

        ####################################### guided sample ######################################
        # get the NN info and compute L2 and mL2
        neighbour, l2list, idx = diffusion.nearest(sample[0], dat, th.tensor([0], device="cuda"))
        l2 = L2(neighbour, sample)
        ml2 = mL2(l2list, n=50, alpha=0.5)
        l2_compare = L2(neighbour_abl, sample)
        ml2_compare = l2_compare / (0.5 * np.mean(sorted(l2list)[:50]))

        # processing for visualiation
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        # store ml2 results for all samples, and visualise ml2 trend for each sample
        ml2_g_all.append(ml2_g_progress)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12,6))
        arr = np.arange(0,251)
        guidance_schedule = asy(arr,1.5,1.5,0.025)
        arr2 = np.arange(0, 250, 1)[::-1]
        arr3 = np.arange(0, len(ml2_g_progress), 1)[::-1]
        ax.plot(arr2, -np.array(ml2_a_progress), label=r'w/o $G_{\mathrm{sim}}$', marker='o', markersize=3, linewidth=2)
        ax.plot(arr3, -np.array(ml2_g_progress), label=r'w $G_{\mathrm{sim}}$', marker='s', markersize=3, linewidth=2)
        ax.plot(arr, -guidance_schedule, linestyle='dashed', label='threshold')  # Uncomment if needed
        ax.set_xlabel('Sampling Step')
        ax.set_ylabel('Similarity Score (negative nL2)')
        ax.grid(True)
        ax.legend(loc='best')  
        fig.savefig(f"sample_{len(all_images) * args.batch_size}_ml2_g.png")

        # plot intermediate sampling steps
        for i in np.arange(len(cache_plot_g)):
            if i % 10 == 0:
                plt.imsave(f"sample_{len(all_images) * args.batch_size}_guided_{i}.png",cache_plot_g[len(cache_plot_g)-1-i][0].cpu().numpy().squeeze())

        plt.imsave(f"sample_{len(all_images) * args.batch_size}_guided.png",sample[0].cpu().numpy().squeeze())
        plt.imsave(f"sample_{len(all_images) * args.batch_size}_diff.png", (sample[0].cpu().numpy().squeeze() - sample_abl[0].cpu().numpy().squeeze()))
        plt.imsave(f"sample_{len(all_images) * args.batch_size}_guided_nbour.png",((neighbour + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1,2,0).cpu().numpy().squeeze())
        logger.log(f"For the image, the {idx}th image is the nearest, with L2 of {l2}, mL2 of {ml2}.")
        logger.log(f"l2_compare: {l2_compare}; ml2_compare: {ml2_compare}.")

        toc = time.time()
        print("Overall: {}".format(toc-tic))

    th.save(ml2_a_all, 'ml2_a_all.pt')
    th.save(ml2_g_all, 'ml2_g_all.pt')

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    arr_abl = np.concatenate(all_images_abl, axis=0)
    arr_abl = arr_abl[: args.num_samples]

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
        label_arr_abl = np.concatenate(all_labels_abl, axis=0)
        label_arr_abl = label_arr_abl[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        
        shape_str_abl = "x".join([str(x) for x in arr_abl.shape])
        out_path_abl = os.path.join(logger.get_dir(), f"samples_{shape_str_abl}_abl.npz")
        logger.log(f"saving to {out_path_abl}")
        if args.class_cond:
            np.savez(out_path_abl, arr_abl, label_arr_abl)
        else:
            np.savez(out_path_abl, arr_abl)

    dist.barrier()
    logger.log("sampling complete")

    # print quantitative results
    ml2_a_0 = -np.array(ml2_a_all)[:, -1] 
    ml2_g_0 = -np.array(ml2_g_all)[:, -1] 

    logger.log("For ablated results:")
    logger.log("{} out of {} generations have negative nL2 > -1.4".format(np.sum(ml2_a_0 > -1.4).item(), len(ml2_a_0)))
    logger.log("{} out of {} generations have negative nL2 > -1.6".format(np.sum(ml2_a_0 > -1.6).item(), len(ml2_a_0)))
    logger.log("nL2 Bottom 5%: {:.4f}".format(np.quantile(ml2_a_0, 0.95).item()))
    logger.log("nL2 Bottom 1: {:.4f}".format(np.max(ml2_a_0).item()))

    logger.log("For guided results:")
    logger.log("{} out of {} generations have negative nL2 > -1.4".format(np.sum(ml2_g_0 > -1.4).item(), len(ml2_g_0)))
    logger.log("{} out of {} generations have negative nL2 > -1.6".format(np.sum(ml2_g_0 > -1.6).item(), len(ml2_g_0)))
    logger.log("nL2 Bottom 5%: {:.4f}".format(np.quantile(ml2_g_0, 0.95).item()))
    logger.log("nL2 Bottom 1: {:.4f}".format(np.max(ml2_g_0).item()))
    logger.log("memorization evaluation complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=1, 
        use_ddim=False,
        model_path="",
        classifier_scale=1000
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()