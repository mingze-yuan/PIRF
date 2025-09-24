"""Main finetuning loop implementing DRaFT algorithm."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from reward.pde import DarcyResidual, BurgersResidual, KolmogorovResidual, HelmholtzResidual, PoissonResidual, Darcy64Residual

LIMIT_N_PLOT_SAMPLES = 4
NUM_EVAL_SAMPLES = 1000

#----------------------------------------------------------------------------

def _get_finetune_reward_fn(reward_fn, device):
    # reward function
    if reward_fn == 'darcy':
        from functools import partial
        reward_fn = DarcyResidual(postprocess_input=True, discretize_a=True, domain_length=1, pixels_per_dim=128, device=device)
        postprocess_fn = partial(reward_fn.postprocess_darcy, discretize_a=True)
        vmax_norm = 1e2
    elif reward_fn == 'darcy64':
        reward_fn = Darcy64Residual(postprocess_input=True, domain_length=1, pixels_per_dim=64, device=device)
        postprocess_fn = reward_fn.postprocess_darcy64
        vmax_norm = 1e2
    elif reward_fn == 'burgers':
        reward_fn = BurgersResidual(postprocess_input=True, domain_length=1, pixels_per_dim=128, device=device)
        postprocess_fn = reward_fn.postprocess_burgers
        vmax_norm = 1e1
    elif reward_fn == 'kolmogorov':
        reward_fn = KolmogorovResidual(postprocess_input=True, re=1000, dt=1/32)
        postprocess_fn = reward_fn.postprocess_kolmogorov
        vmax_norm = 1e1
    elif reward_fn == 'helmholtz':
        reward_fn = HelmholtzResidual(postprocess_input=True, domain_length=1, pixels_per_dim=128, k=1)
        postprocess_fn = reward_fn.postprocess_helmholtz
        vmax_norm = 1e1
    elif reward_fn == 'poisson':
        reward_fn = PoissonResidual(postprocess_input=True, domain_length=1, pixels_per_dim=128)
        postprocess_fn = reward_fn.postprocess_poisson
        vmax_norm = 1e1
    else:
        raise ValueError(f"Invalid reward function: {reward_fn}")
    
    return reward_fn, postprocess_fn, vmax_norm

def training_loop_pgdiffusion(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs        = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs     = None,     # Options for augmentation pipeline, None = disable.
    seed               = 0,        # Global random seed.
    batch_size         = 512,      # Total batch size for one training iteration.
    batch_gpu          = None,     # Limit batch size per GPU, None = no limit.
    total_kimg         = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg  = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio   = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg     = 10000,    # Learning rate ramp-up duration.
    loss_scaling       = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick      = 50,       # Interval of progress prints.
    snapshot_ticks     = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks   = 500,      # How often to dump training state, None = disable.
    resume_pkl         = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump  = None,     # Start from the given training state, None = reset training state.
    resume_kimg        = 0,        # Start from the given training progress.
    cudnn_benchmark    = True,     # Enable torch.backends.cudnn.benchmark?
    device            = torch.device('cuda'),
    reward_fn         = None,     # Reward function r(x0),
    num_steps         = 80,     # Number of sampling steps
    sigma_min         = 0.002,    # Minimum sigma value
    sigma_max         = 80,       # Maximum sigma value
    rho               = 7,        # Karras schedule parameter
    cfg_scale         = 1.0,      # Config scale for conditional generation
):
    
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    for name, param in net.named_parameters():
        print(f"{name}: {param.shape}, {param.requires_grad}")


    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    assert augment_kwargs is None, "Augmentation is not supported in finetuning PIDM"
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume from existing pickle.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
        
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    
    reward_fn, postprocess_fn, vmax_norm = _get_finetune_reward_fn(reward_fn, device)
    while True:
        optimizer.zero_grad(set_to_none=True)
        ddp.train()
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) # normalize to -1..+1
                labels = labels.to(device)
                loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe, residual_fn=reward_fn)
                training_stats.report('Loss/diffusion_loss', loss)
                diffusion_loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                diffusion_loss.backward()
                

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        # Optionally step optimizer per round if needed
        optimizer.step()
        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            
        # Progress tracking and saving.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
            
        # evaluate in the loop
        ############################ Sample the data ############################
        # sigma_min = max(sigma_min, net.sigma_min)
        # sigma_max = min(sigma_max, net.sigma_max)
        # # step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        # # sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        # # sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
        
        # # Save samples per tick
        # plot_seed = seed + 1  # avoid same seed for training and evaluation
        # with torch.random.fork_rng(devices=list(range(dist.get_world_size()))):
        #     x_next = _generate_cond_pgdiffusion(ddp, net, batch_gpu, device, sigma_min, sigma_max, rho, num_steps, plot_seed, cfg_scale=cfg_scale, reward_fn=reward_fn)
        # x_final = postprocess_fn(x_next)
        # residual = reward_fn(x_next)
        
        # print(f"residual L2: {(residual ** 2).mean().item():.6f}")  # b, 1, h, w
        # print(f"residual L1: {residual.abs().mean().item():.6f}")
        # # training_stats.report('Eval/residual_L2', residual ** 2)
        # # training_stats.report('Eval/residual_L1', torch.abs(residual))
        
        # # print(run_dir)
        # if dist.get_rank() == 0:
        #     _visualize_samples_with_residual(x_final, residual, run_dir, cur_tick, plot_seed, vmax_norm=vmax_norm)

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))
        # print('A')
        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory
        # print('B')
        # Save training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0:   
        #     # when saving the state, we evaluate the model
        #     if run_dir is not None:
        #         run_dir_tmp = os.path.join(run_dir, 'eval', f'training-state-{cur_nimg//1000:06d}')
        #         os.makedirs(run_dir_tmp, exist_ok=True)
        #     num_eval_seeds = NUM_EVAL_SAMPLES // batch_gpu + 1
        #     res = dict(residual_l2=[], residual_l1=[])
        #     with torch.random.fork_rng(devices=list(range(dist.get_world_size()))):
        #         for eval_seed in tqdm.tqdm(range(num_eval_seeds), desc='Evaluating model'):
        #             x_next = _generate_cond_pgdiffusion(ddp, net, batch_gpu, device, sigma_min, sigma_max, rho, num_steps, eval_seed + dist.get_rank() * num_eval_seeds, cfg_scale=cfg_scale, reward_fn=reward_fn)
        #             x_final = postprocess_fn(x_next)
        #             residual = reward_fn(x_next)
        #             residual_l2 = (residual ** 2).mean().item()
        #             residual_l1 = residual.abs().mean().item()
        #             res['residual_l2'].append(residual_l2)
        #             res['residual_l1'].append(residual_l1)
        #             if run_dir is not None:
        #                 _visualize_samples_with_residual(x_final, residual, run_dir_tmp, cur_tick, eval_seed, vmax_norm=vmax_norm)
                    
        #                 summary = {
        #                     'model_path': os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'),
        #                     'num_steps': num_steps,
        #                     'cfg_scale': cfg_scale,
        #                     'batch_size': batch_gpu,
        #                     'num_seeds': num_eval_seeds,
        #                     'residual_l2': np.mean(res['residual_l2']).item(),
        #                     'residual_l1': np.mean(res['residual_l1']).item()
        #                 }
        #                 with open(os.path.join(run_dir, 'eval', f'summary-training-state-{cur_nimg//1000:06d}-rank{dist.get_rank()}.json'), 'w') as f:
        #                     json.dump(summary, f, indent=2)
        # Update logs.
        # print('A')
        training_stats.default_collector.update()
        # print('B')
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)
        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')



def finetune_loop_pidm(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs        = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs     = None,     # Options for augmentation pipeline, None = disable.
    seed               = 0,        # Global random seed.
    batch_size         = 512,      # Total batch size for one training iteration.
    batch_gpu          = None,     # Limit batch size per GPU, None = no limit.
    total_kimg         = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg  = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio   = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg     = 10000,    # Learning rate ramp-up duration.
    loss_scaling       = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick      = 50,       # Interval of progress prints.
    snapshot_ticks     = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks   = 500,      # How often to dump training state, None = disable.
    resume_pkl         = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump  = None,     # Start from the given training state, None = reset training state.
    resume_kimg        = 0,        # Start from the given training progress.
    cudnn_benchmark    = True,     # Enable torch.backends.cudnn.benchmark?
    device            = torch.device('cuda'),
    reward_fn         = None,     # Reward function r(x0),
    reward_loss_scaling = 1e-3,  # scaling factor for reward loss
    num_steps         = 80,     # Number of sampling steps
    sigma_min         = 0.002,    # Minimum sigma value
    sigma_max         = 80,       # Maximum sigma value
    rho               = 7,        # Karras schedule parameter
    mode='mean',
):
    
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    assert mode in ['mean', 'sample']

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    assert augment_kwargs is None, "Augmentation is not supported in finetuning PIDM"
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume from existing pickle.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
        
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    
    reward_fn, postprocess_fn, vmax_norm = _get_finetune_reward_fn(reward_fn, device)
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_final_step = sigma_t_steps[-1]
    while True:
        optimizer.zero_grad(set_to_none=True)
        ddp.train()
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) # normalize to -1..+1
                labels = labels.to(device)
                loss, D_yn, x_cur, sigma, _ = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe, return_intermediate=True)
                training_stats.report('Loss/diffusion_loss', loss)
                diffusion_loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                if mode == 'mean':
                    residual = reward_fn(D_yn)
                elif mode == 'sample':
                    d_cur = (x_cur - D_yn) / sigma
                    # head to final step directly based on PIDM
                    x_next = x_cur + (sigma_t_final_step - sigma) * d_cur  
                    pidm_sample = ddp(x_next, sigma_t_final_step, labels, augment_labels=None)
                    residual = reward_fn(pidm_sample)
                    
                reward_loss = (residual ** 2) / (2 * sigma)
                training_stats.report('Loss/reward_loss', reward_loss)
                reward_loss = reward_loss.sum().mul(reward_loss_scaling / batch_gpu_total)
                total_loss = diffusion_loss + reward_loss 
                total_loss.backward()
                

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        # Optionally step optimizer per round if needed
        optimizer.step()
        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            
        # Progress tracking and saving.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
            
        # evaluate in the loop
        ############################ Sample the data ############################
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
        
        # Save samples per tick
        plot_seed = seed + 1  # avoid same seed for training and evaluation
        with torch.random.fork_rng(devices=list(range(dist.get_world_size()))):
            x_next = _generate_uncond(ddp, net, batch_gpu, device, sigma_min, sigma_max, rho, num_steps, plot_seed)
        x_final = postprocess_fn(x_next)
        residual = reward_fn(x_next)
        
        print(f"residual L2: {(residual ** 2).mean().item():.6f}")  # b, 1, h, w
        print(f"residual L1: {residual.abs().mean().item():.6f}")
        training_stats.report('Eval/residual_L2', residual ** 2)
        training_stats.report('Eval/residual_L1', torch.abs(residual))
        
        # print(run_dir)
        if dist.get_rank() == 0:
            _visualize_samples_with_residual(x_final, residual, run_dir, cur_tick, plot_seed, vmax_norm=vmax_norm)

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0):   
            # when saving the state, we evaluate the model
            if run_dir is not None:
                run_dir_tmp = os.path.join(run_dir, 'eval', f'training-state-{cur_nimg//1000:06d}')
                os.makedirs(run_dir_tmp, exist_ok=True)
            num_eval_seeds = NUM_EVAL_SAMPLES // batch_gpu + 1
            res = dict(residual_l2=[], residual_l1=[])
            with torch.random.fork_rng(devices=list(range(dist.get_world_size()))):
                for eval_seed in tqdm.tqdm(range(num_eval_seeds), desc='Evaluating model'):
                    x_next = _generate_uncond(ddp, net, batch_gpu, device, sigma_min, sigma_max, rho, num_steps, eval_seed + dist.get_rank() * num_eval_seeds)
                    x_final = postprocess_fn(x_next)
                    residual = reward_fn(x_next)
                    residual_l2 = (residual ** 2).mean().item()
                    residual_l1 = residual.abs().mean().item()
                    res['residual_l2'].append(residual_l2)
                    res['residual_l1'].append(residual_l1)
                    if run_dir is not None:
                        _visualize_samples_with_residual(x_final, residual, run_dir_tmp, cur_tick, eval_seed, vmax_norm=vmax_norm)
                    
                        summary = {
                            'model_path': os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'),
                            'num_steps': num_steps,
                            'batch_size': batch_gpu,
                            'num_seeds': num_eval_seeds,
                            'residual_l2': np.mean(res['residual_l2']).item(),
                            'residual_l1': np.mean(res['residual_l1']).item()
                        }
                        with open(os.path.join(run_dir, 'eval', f'summary-training-state-{cur_nimg//1000:06d}-rank{dist.get_rank()}.json'), 'w') as f:
                            json.dump(summary, f, indent=2)
        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)
        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------

def freeze_model_except_128x128(model):
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze parameters in the 128x128_* layers of model.model.dec
    for name, module in model.model.dec.items():
        if name.startswith('128x128_'):
            for param in module.parameters():
                param.requires_grad = True
                
def freeze_model_only_128x128(model):
    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # freeze parameters in the 128x128_* layers of model.model.dec
    for name, module in model.model.dec.items():
        if not name.startswith('128x128_'):
            for param in module.parameters():
                param.requires_grad = True
                
def finetune_loop_pirf(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs        = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs     = None,     # Options for augmentation pipeline, None = disable.
    seed               = 0,        # Global random seed.
    batch_size         = 512,      # Total batch size for one training iteration.
    batch_gpu          = None,     # Limit batch size per GPU, None = no limit.
    total_kimg         = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg  = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio   = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg     = 10000,    # Learning rate ramp-up duration.
    loss_scaling       = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick      = 50,       # Interval of progress prints.
    snapshot_ticks     = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks   = 500,      # How often to dump training state, None = disable.
    resume_pkl         = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump  = None,     # Start from the given training state, None = reset training state.
    resume_kimg        = 0,        # Start from the given training progress.
    cudnn_benchmark    = True,     # Enable torch.backends.cudnn.benchmark?
    device            = torch.device('cuda'),
    draft_method      = 'DRaFT-K',  # One of ['ReFT', 'DRaFT', 'DRaFT-K', 'DRaFT-LV']
    k                 = 1,      # Number of steps for DRaFT-K
    reward_fn         = None,     # Reward function r(x₀, c)
    num_steps         = 80,     # Number of sampling steps
    sigma_min         = 0.002,    # Minimum sigma value
    sigma_max         = 80,       # Maximum sigma value
    rho               = 7,        # Karras schedule parameter
    num_inner_steps   = 10,       # Number of inner optimization steps
    freeze_early      = False,     # Freeze all parameters except 128x128_* layers
    random_schedule   = False,     # Whether to use random schedule
    freeze_late       = False,     # Freeze only 128x128_* layers
    weight_regularization = 0.0, # Whether to use weight regularization
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    # torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
    assert augment_kwargs is None, "Augmentation is not supported in finetune"
    assert resume_state_dump is None, "We don't need to resume from a training state dump"

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    # dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    # dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if freeze_early:
        freeze_model_except_128x128(net)
    if freeze_late:
        freeze_model_only_128x128(net)
    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False,)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    pretrain_model = copy.deepcopy(ema)

    # Resume from existing pickle.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    
    reward_fn, postprocess_fn, vmax_norm = _get_finetune_reward_fn(reward_fn, device)
    
    # Sampling parameters
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    # step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    # sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])
    
    num_steps_lst = [20, 40, 80] if random_schedule else [num_steps]
    while True:
        optimizer.zero_grad(set_to_none=True)
        ddp.train()
        num_steps = num_steps_lst[np.random.randint(0, len(num_steps_lst))]
        step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
        sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                latents = torch.randn([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device) * sigma_t_steps[0]
                # print(latents.dtype, net.use_fp16)
                # Determine truncation time based on method
                if draft_method == 'ReFT':
                    t_truncate = torch.randint(1, num_steps, (1,)).item()
                elif draft_method == 'DRaFT':
                    t_truncate = num_steps
                elif draft_method == 'DRaFT-K':
                    t_truncate = k
                else:  # DRaFT-LV
                    t_truncate = 1

                x_next = latents
                for i, (sigma_t_cur, sigma_t_next) in enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:])):
                    sigma_t = net.round_sigma(sigma_t_cur)
                    if i < num_steps - t_truncate:
                        # Before truncation point: no gradients needed
                        x_cur = x_next
                        ddp.eval()
                        with torch.no_grad():
                            x_0_pred = ddp(x_cur, sigma_t)
                            d_cur = (x_cur - x_0_pred) / sigma_t
                            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
                            
                            if i < num_steps - 1:
                                x_0_next = ddp(x_next, sigma_t_next)
                                d_prime = (x_next - x_0_next) / sigma_t_next
                                x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
                    
                    else:
                        ddp.train()
                        x_cur = x_next
                        x_cur.requires_grad_(True)
                        
                        x_0_pred = ddp(x_cur, sigma_t)                
                        d_cur = (x_cur - x_0_pred) / sigma_t
                        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
                        
                        if i < num_steps - 1:
                            x_0_next = ddp(x_next, sigma_t_next)
                            d_prime = (x_next - x_0_next) / sigma_t_next
                            x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)

                reward_loss = reward_fn(x_next).square().mean() * loss_scaling
                
                if draft_method == 'DRaFT-LV':
                    # For DRaFT-LV: accumulate gradients over multiple samples at t=1
                    ddp.train()
                    for _ in range(num_inner_steps):
                        noise = torch.randn_like(x_next)
                        sigma_t_1 = net.round_sigma(sigma_t_steps[-2]) # simulate last step
                        x_1 = x_next + noise * sigma_t_1
                        x_0_sample = ddp(x_1, sigma_t_1)                        
                        reward_sample = reward_fn(x_0_sample)  # Negative because we want to maximize reward
                        reward_loss += reward_sample.square().mean() * loss_scaling
                    reward_loss /= (num_inner_steps + 1)
                
                training_stats.report('Loss/reward_loss', reward_loss)
                print(f"reward_loss: {reward_loss.item():.6f}")
                
                if weight_regularization:
                    # Add L2 regularization between current and pretrained weights
                    reg_loss = 0
                    for (name, param), (_, param_pretrained) in zip(ddp.named_parameters(), pretrain_model.named_parameters()):
                        if param.requires_grad:
                            diff = param - param_pretrained
                            reg_loss += diff.norm(2)
                    reg_loss = reg_loss * weight_regularization  # Scale factor for regularization
                    training_stats.report('Loss/reg_loss', reg_loss)
                    print(f"reg_loss: {reg_loss.item():.6f}")
                    reward_loss += reg_loss
                
                reward_loss.backward()
                
                
        
        # print(f"max memory allocated: {torch.cuda.max_memory_allocated(device) / 2**30}")
        # print(f"max memory reserved: {torch.cuda.max_memory_reserved(device) / 2**30}")

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        # Optionally step optimizer per round if needed
        optimizer.step()
        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        # print(f"ema_beta: {ema_beta}")
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            
        # Progress tracking and saving.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        
        # Save samples per tick
        plot_seed = seed + 1  # avoid same seed for training and evaluation
        with torch.random.fork_rng(devices=list(range(dist.get_world_size()))):
            x_next = _generate_uncond(ddp, net, batch_gpu, device, sigma_min, sigma_max, rho, num_steps, plot_seed)
        x_final = postprocess_fn(x_next)
        residual = reward_fn(x_next)
        
    
        print(f"residual L2: {(residual ** 2).mean().item():.6f}")  # b, 1, h, w
        print(f"residual L1: {residual.abs().mean().item():.6f}")
    
        training_stats.report('Eval/residual_L2', residual ** 2)
        training_stats.report('Eval/residual_L1', residual.abs())
        
        # print(run_dir)
        if dist.get_rank() == 0:
            _visualize_samples_with_residual(x_final, residual, run_dir, cur_tick, plot_seed, vmax_norm=vmax_norm, num_steps=num_steps)

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
        
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0):   
            # when saving the state, we evaluate the model
            if run_dir is not None:
                run_dir_tmp = os.path.join(run_dir, 'eval', f'training-state-{cur_nimg//1000:06d}')
                os.makedirs(run_dir_tmp, exist_ok=True)
            num_eval_seeds = NUM_EVAL_SAMPLES // batch_gpu + 1
            res = dict(residual_l2=[], residual_l1=[])
            with torch.random.fork_rng(devices=list(range(dist.get_world_size()))):
                for eval_seed in tqdm.tqdm(range(num_eval_seeds), desc='Evaluating model'):
                    x_next = _generate_uncond(ddp, net, batch_gpu, device, sigma_min, sigma_max, rho, num_steps, eval_seed + dist.get_rank() * num_eval_seeds)
                    x_final = postprocess_fn(x_next)
                    residual = reward_fn(x_next)
                    residual_l2 = (residual ** 2).mean().item()
                    residual_l1 = residual.abs().mean().item()
                    res['residual_l2'].append(residual_l2)
                    res['residual_l1'].append(residual_l1)
                    if run_dir is not None:
                        _visualize_samples_with_residual(x_final, residual, run_dir_tmp, cur_tick, eval_seed, vmax_norm=vmax_norm, num_steps=num_steps)
                    
                        summary = {
                            'model_path': os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'),
                            'num_steps': num_steps,
                            'batch_size': batch_gpu,
                            'num_seeds': num_eval_seeds,
                            'residual_l2': np.mean(res['residual_l2']).item(),
                            'residual_l1': np.mean(res['residual_l1']).item()
                        }
                        with open(os.path.join(run_dir, 'eval', f'summary-training-state-{cur_nimg//1000:06d}-rank{dist.get_rank()}.json'), 'w') as f:
                            json.dump(summary, f, indent=2)

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)
        
        # torch.distributed.barrier()
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    dist.print0()
    dist.print0('Exiting...')    
    
# def finetune_loop_pirf(
#     run_dir             = '.',      # Output directory.
#     dataset_kwargs      = {},       # Options for training set.
#     data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
#     network_kwargs      = {},       # Options for model and preconditioning.
#     loss_kwargs        = {},       # Options for loss function.
#     optimizer_kwargs    = {},       # Options for optimizer.
#     augment_kwargs     = None,     # Options for augmentation pipeline, None = disable.
#     seed               = 0,        # Global random seed.
#     batch_size         = 512,      # Total batch size for one training iteration.
#     batch_gpu          = None,     # Limit batch size per GPU, None = no limit.
#     total_kimg         = 200000,   # Training duration, measured in thousands of training images.
#     ema_halflife_kimg  = 500,      # Half-life of the exponential moving average (EMA) of model weights.
#     ema_rampup_ratio   = 0.05,     # EMA ramp-up coefficient, None = no rampup.
#     lr_rampup_kimg     = 10000,    # Learning rate ramp-up duration.
#     loss_scaling       = 1,        # Loss scaling factor for reducing FP16 under/overflows.
#     kimg_per_tick      = 50,       # Interval of progress prints.
#     snapshot_ticks     = 50,       # How often to save network snapshots, None = disable.
#     state_dump_ticks   = 500,      # How often to dump training state, None = disable.
#     resume_pkl         = None,     # Start from the given network snapshot, None = random initialization.
#     resume_state_dump  = None,     # Start from the given training state, None = reset training state.
#     resume_kimg        = 0,        # Start from the given training progress.
#     cudnn_benchmark    = True,     # Enable torch.backends.cudnn.benchmark?
#     device            = torch.device('cuda'),
#     schedule      = 'tail',  # One of ['ReFT', 'DRaFT', 'DRaFT-K', 'DRaFT-LV']
#     k                 = 2,      # Number of steps for DRaFT-K
#     reward_fn         = None,     # Reward function r(x₀, c)
#     num_steps         = 20,     # Number of sampling steps
#     sigma_min         = 0.002,    # Minimum sigma value
#     sigma_max         = 80,       # Maximum sigma value
#     rho               = 7,        # Karras schedule parameter
#     # num_inner_steps   = 10,       # Number of inner optimization steps
# ):
#     # Initialize.
#     start_time = time.time()
#     np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
#     torch.manual_seed(np.random.randint(1 << 31))
#     # torch.manual_seed(seed)
#     torch.backends.cudnn.benchmark = cudnn_benchmark
#     torch.backends.cudnn.allow_tf32 = False
#     torch.backends.cuda.matmul.allow_tf32 = False
#     torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
#     assert augment_kwargs is None, "Augmentation is not supported in finetune"
#     assert resume_state_dump is None, "We don't need to resume from a training state dump"

#     # Select batch size per GPU.
#     batch_gpu_total = batch_size // dist.get_world_size()
#     if batch_gpu is None or batch_gpu > batch_gpu_total:
#         batch_gpu = batch_gpu_total
#     num_accumulation_rounds = batch_gpu_total // batch_gpu
#     assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

#     # Load dataset.
#     dist.print0('Loading dataset...')
#     dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
#     dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
#     dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

#     # Construct network.
#     dist.print0('Constructing network...')
#     interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
#     net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
#     net.train().requires_grad_(True).to(device)

#     # Setup optimizer.
#     dist.print0('Setting up optimizer...')
#     loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
#     optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
#     augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
#     ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False,)
#     ema = copy.deepcopy(net).eval().requires_grad_(False)

#     # Resume from existing pickle.
#     if resume_pkl is not None:
#         dist.print0(f'Loading network weights from "{resume_pkl}"...')
#         if dist.get_rank() != 0:
#             torch.distributed.barrier() # rank 0 goes first
#         with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
#             data = pickle.load(f)
#         if dist.get_rank() == 0:
#             torch.distributed.barrier() # other ranks follow
#         misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
#         misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
#         del data # conserve memory
    
#     if resume_state_dump:
#         dist.print0(f'Loading training state from "{resume_state_dump}"...')
#         data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
#         misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
#         optimizer.load_state_dict(data['optimizer_state'])
#         del data # conserve memory

#     # Train.
#     dist.print0(f'Training for {total_kimg} kimg...')
#     dist.print0()
#     cur_nimg = resume_kimg * 1000
#     cur_tick = 0
#     tick_start_nimg = cur_nimg
#     tick_start_time = time.time()
#     maintenance_time = tick_start_time - start_time
#     dist.update_progress(cur_nimg // 1000, total_kimg)
#     stats_jsonl = None
    
#     reward_fn, postprocess_fn, vmax_norm = _get_finetune_reward_fn(reward_fn, device)
    
#     # Sampling parameters
#     sigma_min = max(sigma_min, net.sigma_min)
#     sigma_max = min(sigma_max, net.sigma_max)
#     step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
#     sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
#     sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])])
    
    
    
#     while True:
#         optimizer.zero_grad(set_to_none=True)
#         ddp.train()
        
#         for round_idx in range(num_accumulation_rounds):
#             with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
#                 latents = torch.randn([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device) * sigma_t_steps[0]
#                 if schedule == 'tail':
#                     t_train = list(range(k))  # last k steps
#                 elif schedule == 'head':
#                     t_train = list(range(num_steps-k, num_steps))  # first k steps
#                 elif schedule == 'middle':
#                     t_train = list(range(num_steps // 2 - k // 2, num_steps // 2 + k // 2))
#                 elif schedule == 'paced':
#                     pace = num_steps // k
#                     t_train = list(range(0, num_steps, pace))
#                 elif schedule == 'random':
#                     t_train = torch.randint(0, num_steps, (k,))
#                 else:
#                     raise ValueError(f"Invalid schedule: {schedule}")

#                 # Iterative sampling process
#                 x_next = latents
#                 for i, (sigma_t_cur, sigma_t_next) in enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:])):
#                     x_cur = x_next
#                     ddp.train()
#                     sigma_t = net.round_sigma(sigma_t_cur)
                    
#                     x_input = x_cur.detach()
#                     x_input.requires_grad = x_cur.requires_grad
#                     x_0_pred = ddp(x_input, sigma_t)
                    
                    
#                     if num_steps - i not in t_train:
#                         x_0_pred = x_0_pred.detach()
                            
#                     d_cur = (x_cur - x_0_pred) / sigma_t
#                     x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
                    
                        
#                     # if i < num_steps - 1:
#                     #     x_0_next = ddp(x_next.detach(), sigma_t_next)
#                     #     d_prime = (x_next - x_0_next) / sigma_t_next
#                     #     x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
                    

#                 reward_loss = reward_fn(x_next).square().mean() * loss_scaling
                
#                 # if draft_method == 'DRaFT-LV':
#                 #     # For DRaFT-LV: accumulate gradients over multiple samples at t=1
#                 #     ddp.train()
#                 #     for _ in range(num_inner_steps):
#                 #         noise = torch.randn_like(x_next)
#                 #         sigma_t_1 = net.round_sigma(sigma_t_steps[-2]) # simulate last step
#                 #         x_1 = x_next + noise * sigma_t_1
#                 #         x_0_sample = ddp(x_1, sigma_t_1)                        
#                 #         reward_sample = reward_fn(x_0_sample)  # Negative because we want to maximize reward
#                 #         reward_loss += reward_sample.square().mean() * loss_scaling
#                 #     reward_loss /= (num_inner_steps + 1)
                
#                 reward_loss.backward()
                
#                 training_stats.report('Loss/reward_loss', reward_loss)
#                 print(f"reward_loss: {reward_loss.item():.6f}")
        

#         # Update weights.
#         for g in optimizer.param_groups:
#             g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
#         for param in net.parameters():
#             if param.grad is not None:
#                 torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

#         # Optionally step optimizer per round if needed
#         optimizer.step()
#         # Update EMA.
#         ema_halflife_nimg = ema_halflife_kimg * 1000
#         if ema_rampup_ratio is not None:
#             ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
#         ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
#         # print(f"ema_beta: {ema_beta}")
#         for p_ema, p_net in zip(ema.parameters(), net.parameters()):
#             p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
            
#         # Progress tracking and saving.
#         cur_nimg += batch_size
#         done = (cur_nimg >= total_kimg * 1000)
#         if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
#             continue
        
#         # Save samples per tick
#         plot_seed = seed + 1  # avoid same seed for training and evaluation
#         with torch.random.fork_rng(devices=list(range(dist.get_world_size()))):
#             x_next = _generate_uncond(ddp, net, batch_gpu, device, sigma_min, sigma_max, rho, num_steps, plot_seed)
#         x_final = postprocess_fn(x_next)
#         residual = reward_fn(x_next)
        
    
#         print(f"residual L2: {(residual ** 2).mean().item():.6f}")  # b, 1, h, w
#         print(f"residual L1: {residual.abs().mean().item():.6f}")
    
#         training_stats.report('Eval/residual_L2', residual ** 2)
#         training_stats.report('Eval/residual_L1', residual.abs())
        
#         # print(run_dir)
#         if dist.get_rank() == 0:
#             _visualize_samples_with_residual(x_final, residual, run_dir, cur_tick, plot_seed, vmax_norm=vmax_norm, num_steps=num_steps)

#         # Print status line, accumulating the same information in training_stats.
#         tick_end_time = time.time()
#         fields = []
#         fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
#         fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
#         fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
#         fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
#         fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
#         fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
#         fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
#         fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
#         fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
#         torch.cuda.reset_peak_memory_stats()
#         dist.print0(' '.join(fields))

#         # Save network snapshot.
#         if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
#             data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
#             for key, value in data.items():
#                 if isinstance(value, torch.nn.Module):
#                     value = copy.deepcopy(value).eval().requires_grad_(False)
#                     misc.check_ddp_consistency(value)
#                     data[key] = value.cpu()
#                 del value # conserve memory
#             if dist.get_rank() == 0:
#                 with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
#                     pickle.dump(data, f)
#             del data # conserve memory

#         # Save training state.
#         if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
#             torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
        
#         if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0):   
#             # when saving the state, we evaluate the model
#             if run_dir is not None:
#                 run_dir_tmp = os.path.join(run_dir, 'eval', f'training-state-{cur_nimg//1000:06d}')
#                 os.makedirs(run_dir_tmp, exist_ok=True)
#             num_eval_seeds = NUM_EVAL_SAMPLES // batch_gpu + 1
#             res = dict(residual_l2=[], residual_l1=[])
#             with torch.random.fork_rng(devices=list(range(dist.get_world_size()))):
#                 for eval_seed in tqdm.tqdm(range(num_eval_seeds), desc='Evaluating model'):
#                     x_next = _generate_uncond(ddp, net, batch_gpu, device, sigma_min, sigma_max, rho, num_steps, eval_seed + dist.get_rank() * num_eval_seeds)
#                     x_final = postprocess_fn(x_next)
#                     residual = reward_fn(x_next)
#                     residual_l2 = (residual ** 2).mean().item()
#                     residual_l1 = residual.abs().mean().item()
#                     res['residual_l2'].append(residual_l2)
#                     res['residual_l1'].append(residual_l1)
#                     if run_dir is not None:
#                         _visualize_samples_with_residual(x_final, residual, run_dir_tmp, cur_tick, eval_seed, vmax_norm=vmax_norm, num_steps=num_steps)
                    
#                         summary = {
#                             'model_path': os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'),
#                             'num_steps': num_steps,
#                             'batch_size': batch_gpu,
#                             'num_seeds': num_eval_seeds,
#                             'residual_l2': np.mean(res['residual_l2']).item(),
#                             'residual_l1': np.mean(res['residual_l1']).item()
#                         }
#                         with open(os.path.join(run_dir, 'eval', f'summary-training-state-{cur_nimg//1000:06d}-rank{dist.get_rank()}.json'), 'w') as f:
#                             json.dump(summary, f, indent=2)

#         # Update logs.
#         training_stats.default_collector.update()
#         if dist.get_rank() == 0:
#             if stats_jsonl is None:
#                 stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
#             stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
#             stats_jsonl.flush()
#         dist.update_progress(cur_nimg // 1000, total_kimg)
        
#         # torch.distributed.barrier()
#         cur_tick += 1
#         tick_start_nimg = cur_nimg
#         tick_start_time = time.time()
#         maintenance_time = tick_start_time - tick_end_time
#         if done:
#             break

#     dist.print0()
#     dist.print0('Exiting...')    
    


def _visualize_samples_with_residual(x_final, residual, run_dir, cur_tick, seed, limit_n_plot_samples=LIMIT_N_PLOT_SAMPLES, vmax_norm=1e2, num_steps=80):
    # print(x_final.shape, residual.shape, run_dir, cur_tick, seed)
    num_samples = min(limit_n_plot_samples, x_final.shape[0])
    num_channels = x_final.shape[1]
    _, axes = plt.subplots(num_samples, num_channels + 1, figsize=(5*(num_channels + 1), 5*num_samples))

    for i in range(num_samples):
        for j in range(num_channels):
            # Plot first channel of x_final
            im0 = axes[i,j].imshow(x_final[i,j].detach().cpu(), cmap='jet')
            axes[i,j].set_title(f'Sample {i+1} - Channel {j+1}')
            plt.colorbar(im0, ax=axes[i,j], fraction=0.046, pad=0.04)
        
        # Plot residual
        im2 = axes[i,num_channels].imshow(residual[i,0].abs().detach().cpu(), norm=colors.LogNorm(vmin=1e-3, vmax=vmax_norm), cmap='jet')
        axes[i,num_channels].set_title(f'Sample {i+1} - Residual L1')
        plt.colorbar(im2, ax=axes[i,num_channels], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.join(run_dir, 'samples'), exist_ok=True)
    rank = dist.get_rank()
    plt.savefig(os.path.join(run_dir, 'samples', f'samples_seed{seed:03d}_tick{cur_tick:06d}_steps{num_steps}_rank{rank}.png'))
    plt.close()

def _generate_uncond(ddp, net, batch_size=16, device='cuda', sigma_min=0.002, sigma_max=80, rho=7, num_steps=80, seed=0):
    # for multi-GPU version
    torch.manual_seed(seed)
    ddp.eval()
    # print(f'Generating {batch_size} samples...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
        
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
    
    x_next = latents.to(torch.float32) * sigma_t_steps[0]
    with torch.no_grad():
        for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'): # 0, ..., N-1
            x_cur = x_next
            # x_cur.requires_grad = True
            sigma_t = net.round_sigma(sigma_t_cur)
            
            # Euler step
            x_N = ddp(x_cur, sigma_t, class_labels=class_labels)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
            
            # 2nd order correction
            if i < num_steps - 1:
                x_N = ddp(x_next, sigma_t_next, class_labels=class_labels)
                d_prime = (x_next - x_N) / sigma_t_next
                x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
                
    return x_next

def _generate_cond_pgdiffusion(ddp, net, batch_size=16, device='cuda', sigma_min=0.002, sigma_max=80, rho=7, num_steps=80, seed=0, cfg_scale=1.0, reward_fn=None):
    # for multi-GPU version
    assert reward_fn is not None
    torch.manual_seed(seed)
    ddp.eval()
    # print(f'Generating {batch_size} samples...')
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
        
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma_t_steps = torch.cat([net.round_sigma(sigma_t_steps), torch.zeros_like(sigma_t_steps[:1])]) # t_N = 0
    
    x_next = latents.to(torch.float32) * sigma_t_steps[0]
    # with torch.no_grad():
    for i, (sigma_t_cur, sigma_t_next) in tqdm.tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        residual_cur = torch.norm(reward_fn(x_cur), p=2)
        grad_dx = torch.autograd.grad(residual_cur, x_cur)[0]
        with torch.no_grad():
            sigma_t = net.round_sigma(sigma_t_cur)
            
            # Euler step
            if cfg_scale == 1.0:
                x_N = ddp(x_cur, sigma_t, class_labels=class_labels, dx=grad_dx)
            else:
                x_N = ddp(x_cur, sigma_t, class_labels=class_labels, dx=grad_dx) * cfg_scale + ddp(x_cur, sigma_t, class_labels=class_labels) * (1 - cfg_scale)
            d_cur = (x_cur - x_N) / sigma_t
            x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
            
            # 2nd order correction
            # if i < num_steps - 1:
            #     residual_next = torch.norm(reward_fn(x_next), p=2)
            #     grad_dx = torch.autograd.grad(residual_next, x_next)[0]
            #     if cfg_scale == 1.0:
            #         x_N = ddp(x_next, sigma_t, class_labels=class_labels, dx=grad_dx)
            #     else:
            #         x_N = ddp(x_next, sigma_t, class_labels=class_labels, dx=grad_dx) * cfg_scale + ddp(x_next, sigma_t, class_labels=class_labels) * (1 - cfg_scale)
            #     d_prime = (x_next - x_N) / sigma_t_next
            #     x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
                
    return x_next



#----------------------------------------------------------------------------
