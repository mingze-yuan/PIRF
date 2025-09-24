import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
import json
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from reward.pde import DarcyResidual, Darcy64Residual, BurgersResidual, HelmholtzResidual, PoissonResidual, KolmogorovResidual

dict_base_model_pth = {
    'darcy': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/workspace_pgdiffusion/darcy/00001-darcy-uncond-ddpmpp-edm-gpus1-batch60-fp16/network-snapshot-001002.pkl',
    'burgers': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/workspace_pgdiffusion/burgers/00001-burgers-uncond-ddpmpp-edm-gpus1-batch60-fp16/network-snapshot-006000.pkl',
    'helmholtz': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/workspace_pgdiffusion/helmholtz/00001-helmholtz-uncond-ddpmpp-edm-gpus1-batch60-fp16/network-snapshot-006000.pkl',
    'poisson': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/workspace_pgdiffusion/poisson/00001-poisson-uncond-ddpmpp-edm-gpus1-batch60-fp16/network-snapshot-006000.pkl',
    'kolmogorov': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/workspace_pgdiffusion/kolmogorov/00001-kolmogorov-uncond-ddpmpp-edm-gpus1-batch20-fp16/network-snapshot-006000.pkl',
}

def generate_with_pgdiffusion(net, guided_residual_fn, batch_size=16, device='cuda', sigma_min=0.002, sigma_max=80, rho=7, num_steps=80, seed=0, cfg_scale=1.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
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
    for i, (sigma_t_cur, sigma_t_next) in tqdm(list(enumerate(zip(sigma_t_steps[:-1], sigma_t_steps[1:]))), unit='step'): # 0, ..., N-1
        x_cur = x_next.detach().clone()
        x_cur.requires_grad = True
        residual_cur = torch.norm(guided_residual_fn(x_cur), p=2)
        grad_dx = torch.autograd.grad(residual_cur, x_cur)[0]
        sigma_t = net.round_sigma(sigma_t_cur)
        # Euler step
        if i < 0.3 * num_steps:
            x_N = net(x_cur, sigma_t, class_labels=class_labels)
        elif cfg_scale == 1.0:
            x_N = net(x_cur, sigma_t, class_labels=class_labels, dx=grad_dx)
        else:
            x_N = net(x_cur, sigma_t, class_labels=class_labels, dx=grad_dx) * cfg_scale + net(x_cur, sigma_t, class_labels=class_labels) * (1 - cfg_scale)
        d_cur = (x_cur - x_N) / sigma_t
        x_next = x_cur + (sigma_t_next - sigma_t) * d_cur
        
        # 2nd order correction
        # if i < num_steps - 1:
        #     x_N = net(x_next, sigma_t_next, class_labels=class_labels)
        #     d_prime = (x_next - x_N) / sigma_t_next
        #     x_next = x_cur + (sigma_t_next - sigma_t) * (0.5 * d_cur + 0.5 * d_prime)
        
    return x_next

def get_required_tools(data_name: str, device: torch.device):
    batch_size = 8
    if data_name == 'darcy':
        eval_residual_fn = DarcyResidual(postprocess_input=True, discretize_a=True, domain_length=1, pixels_per_dim=128, device=device)
        guided_residual_fn = DarcyResidual(postprocess_input=True, discretize_a=False, domain_length=1, pixels_per_dim=128, device=device)
    elif data_name == 'darcy64':
        eval_residual_fn = Darcy64Residual(postprocess_input=True, domain_length=1, pixels_per_dim=64, device=device)
        guided_residual_fn = eval_residual_fn
    elif data_name == 'burgers':
        eval_residual_fn = BurgersResidual(postprocess_input=True, domain_length=1, pixels_per_dim=128, device=device)
        guided_residual_fn = eval_residual_fn
    elif data_name == 'helmholtz':
        eval_residual_fn = HelmholtzResidual(postprocess_input=True, domain_length=1, pixels_per_dim=128, k=1, device=device)
        guided_residual_fn = eval_residual_fn
    elif data_name == 'poisson':
        eval_residual_fn = PoissonResidual(postprocess_input=True, domain_length=1, pixels_per_dim=128, device=device)
        guided_residual_fn = eval_residual_fn
    elif data_name == 'kolmogorov':
        eval_residual_fn = KolmogorovResidual(postprocess_input=True)
        guided_residual_fn = eval_residual_fn
        batch_size = 4
    else:
        raise ValueError(f'Invalid reward function: {data_name}')
    
    return eval_residual_fn, guided_residual_fn, batch_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='darcy')
    args = parser.parse_args()
    
    output_base_dir = '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/results/baselines_updated/pgdiffusion'
    output_dir = os.path.join(output_base_dir, args.data_name)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    eval_residual_fn, guided_residual_fn, batch_size = get_required_tools(args.data_name, device)
    total_samples = 1600
    # num_steps = 20
    num_steps_lst = [20, 40, 80]
    cfg_scale_lst = [1.0]
    # batch_size = 8
    
    # zeta_pde_lst = [0, 1e-5, 1e-4, 1e-3]
    model_path = dict_base_model_pth[args.data_name]
    model_name = os.path.basename(model_path).split('.')[0]
    seeds = list(range(total_samples // batch_size))
    with open(model_path, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)
    net.eval()
    for num_steps in num_steps_lst:
        for cfg_scale in cfg_scale_lst:
            save_dir = os.path.join(output_dir, f"{model_name}-iter{num_steps}")
            os.makedirs(save_dir, exist_ok=True)
            
            lst_res_mse, lst_res_mae = [], []

            for seed in tqdm(seeds):
                x = generate_with_pgdiffusion(
                    net=net, 
                    guided_residual_fn=guided_residual_fn, 
                    batch_size=batch_size, 
                    device=device, 
                    seed=seed, 
                    num_steps=num_steps,
                    cfg_scale=cfg_scale)
                
                L_pde = eval_residual_fn(x.to(torch.float32))
                
                residual_mse = (L_pde ** 2).mean()
                residual_mae = L_pde.abs().mean()
                lst_res_mse.append(residual_mse.item())
                lst_res_mae.append(residual_mae.item())
                    
                vis_dir = os.path.join(save_dir, 'visualize')
                os.makedirs(vis_dir, exist_ok=True)
                for i in range(batch_size):
                    num_channels = x.shape[1]
                    plt.figure(figsize=(5*(num_channels+1), 5))
                    
                    # Plot each channel of x
                    for j in range(num_channels):
                        plt.subplot(1, num_channels+1, j+1)
                        im = plt.imshow(x[i,j].detach().cpu(), cmap='jet')
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        plt.title(f'Channel {j+1}')
                    
                    # Plot residual (always 1 channel)
                    plt.subplot(1, num_channels+1, num_channels+1)
                    im_res = plt.imshow(L_pde[i,0].abs().detach().cpu(), norm=colors.LogNorm(vmin=1e-3, vmax=1e2), cmap='jet')
                    plt.colorbar(im_res, fraction=0.046, pad=0.04)
                    plt.title('PDE Residual (abs)')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'seed_{seed:03d}_sample_{i:03d}.png'), dpi=300)
                    plt.close()
                    print(f'Saved to {os.path.join(vis_dir, f"seed_{seed:03d}_sample_{i:03d}.png")}')


                res = {
                    'model_path': model_path,
                    'num_steps': num_steps,
                    'batch_size': batch_size,
                    'num_seeds': len(seeds),
                    'cfg_scale': cfg_scale,
                    'residual_l2': np.mean(lst_res_mse).item(),
                    'residual_l1': np.mean(lst_res_mae).item()
                }
                # print(res)
                metrics_file = os.path.join(output_dir, f'{model_name}-iter{num_steps}-cfg{cfg_scale}-metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump(res, f, indent=2)
                    
            
        