import torch
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from reward.pde import DarcyResidual, Darcy64Residual, BurgersResidual, HelmholtzResidual, PoissonResidual, KolmogorovResidual

training_data_dir_lst = {
    'darcy': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/preprocessed_data/darcy',
    'darcy64': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/preprocessed_data/darcy64',
    'burgers': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/preprocessed_data/burgers',
    'helmholtz': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/preprocessed_data/helmholtz',
    'poisson': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/preprocessed_data/poisson',
    'kolmogorov': '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/preprocessed_data/kolmogorov',
}

def get_reward_fn(data_name: str, device: torch.device):
    if data_name == 'darcy':
        return DarcyResidual(postprocess_input=True, discretize_a=False, domain_length=1, pixels_per_dim=128, device=device)
    elif data_name == 'darcy64':
        return Darcy64Residual(postprocess_input=True, domain_length=1, pixels_per_dim=64, device=device)
    elif data_name == 'burgers':
        return BurgersResidual(postprocess_input=True, domain_length=1, pixels_per_dim=128, device=device)
    elif data_name == 'helmholtz':
        return HelmholtzResidual(postprocess_input=True, domain_length=1, pixels_per_dim=128, k=1, device=device)
    elif data_name == 'poisson':
        return PoissonResidual(postprocess_input=True, domain_length=1, pixels_per_dim=128, device=device)
    elif data_name == 'kolmogorov':
        return KolmogorovResidual(postprocess_input=True)
    else:
        raise ValueError(f'Invalid reward function: {data_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='darcy')
    args = parser.parse_args()
    
    output_dir = '/n/netscratch/nali_lab_seas/Everyone/mingze/physics_workspace/preprocessed_data/training_data_statistics'
    output_dir = os.path.join(output_dir, args.data_name)
    os.makedirs(output_dir, exist_ok=True)

    dict_residual = {'file': [],'residual_l1': [], 'residual_l2': []}
    
    data_dir = training_data_dir_lst[args.data_name]
    lst_files = sorted(os.listdir(data_dir))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    reward_fn = get_reward_fn(args.data_name, device)

    for idx, file in tqdm(enumerate(lst_files)):
        data_np = np.load(os.path.join(data_dir, file))
        data = torch.from_numpy(data_np).to(device).to(torch.float32)
        # Nx, Ny, Nc = data.shape
        data = data.permute(2, 0, 1).unsqueeze(0)
        # data = data.reshape(1, Nc, Nx, Ny)
        residual = reward_fn(data)
        dict_residual['file'].append(file)
        dict_residual['residual_l1'].append(residual.abs().mean().item())
        dict_residual['residual_l2'].append(residual.square().mean().item())
        
        if idx % 200 == 0:
            df = pd.DataFrame(dict_residual)
            df.to_excel(os.path.join(output_dir, f'{args.data_name}_residual.xlsx'), index=False)
            
            summary_df = {
                'residual_l1_mean': np.mean(dict_residual['residual_l1']).item(), 
                'residual_l1_std': np.std(dict_residual['residual_l1']).item(),
                'residual_l1_median': np.median(dict_residual['residual_l1']).item(),
                'residual_l1_min': np.min(dict_residual['residual_l1']),
                'residual_l1_max': np.max(dict_residual['residual_l1']),
                'residual_l2_mean': np.mean(dict_residual['residual_l2']).item(),
                'residual_l2_std': np.std(dict_residual['residual_l2']).item(),
                'residual_l2_median': np.median(dict_residual['residual_l2']).item(),
                'residual_l2_min': np.min(dict_residual['residual_l2']),
                'residual_l2_max': np.max(dict_residual['residual_l2'])
            }
            with open(os.path.join(output_dir, f'{args.data_name}_residual_statistics.json'), 'w') as f:
                json.dump(summary_df, f, indent=2)
            
    
    print(f'Residual statistics saved to {output_dir}')