from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


def get_fraction_oracles(
    oracles,
    test_correlation,
    generate_figure=True,
    oracle_label="Oracles",
    test_label="Test correlations",
    fig_name="oracle_fig.png",
):
    
    """
    Given oracles and test_correlations (both for each neuron), this method
    computes the fraction of oracle performance
    """

    # we will fit a linear function without offset
    def f(x, a):
        return a * x

    if np.any(np.isnan(test_correlation)):
        print(
            "{}% NaNs in test_correlations, NaNs will be set to Zero.".format(
                np.isnan(test_correlation).mean() * 100
            )
        )
    test_correlation[np.isnan(test_correlation)] = 0

    if np.any(np.isinf(test_correlation)):
        print(
            "{}% infinity numbers in test_correlations, infinity will be set to Zero.".format(
                np.isinf(test_correlation).mean() * 100
            )
        )
    test_correlation[np.isinf(test_correlation)] = 0

    if np.any(np.isnan(oracles)):
        print(
            "{}% NaNs in oracles, NaNs will be set to Zero.".format(
                np.isnan(oracles).mean() * 100
            )
        )
    oracles[np.isnan(oracles)] = 0

    if np.any(np.isinf(oracles)):
        print(
            "{}% infinity numbers in oracles, infinity will be set to Zero.".format(
                np.isinf(oracles).mean() * 100
            )
        )
    oracles[np.isinf(oracles)] = 0

    slope, _ = curve_fit(f, oracles, test_correlation)
    format_float = "{:.1f}".format(slope[0] * 100)

    if generate_figure:
        plt.scatter(oracles, test_correlation, s=1, color="orange")
        x = np.linspace(0, 1, 100)
        plt.plot(x, f(x, slope), "r-", label=format_float + "% oracle")
        plt.plot(x, x, "k--", linewidth=1, label="100.0% oracle")
        plt.axvline(x=0, c="gray", linewidth=0.5)
        plt.axhline(y=0, c="gray", linewidth=0.5)
        plt.xlabel(oracle_label)
        plt.ylabel(test_label)
        plt.legend(loc="upper left")
        plt.savefig(fig_name)
        plt.clf()

    return slope

def get_config(model="EM"):
    if model == "EM":
        return {
            'batch_size': 10,
            'bias_init': 2.5,
            'bottleneck_kernel': 15,
            'brain_crop': None,
            'compute_oracle_fraction': False,
            'conservative_oracle': True,
            'core_gamma_hidden': 0.28463619129195233,
            'core_gamma_input': 0.00307424496692959,
            'core_hidden_channels': 3,
            'core_hidden_kern': 3,
            'core_input_kern': 3,
            'core_layers': 1,
            'counter_clockwise_rotation': True,
            'dataset_artifact_name': 'Antolik_dataset:latest',
            'default_ori_shift': 90,
            'depth_separable': True,
            'do_not_sample': True,
            'em_bias': True,
            'exact_init': True,
            'f_init': 0.63,
            'factor': 5.5,
            'fixed_sigma': False,
            'freeze_orientations': False,
            'freeze_positions': False,
            'generate_oracle_figure': False,
            'ground_truth_orientations_file_path': 'data/antolik/oris_reparametrized.pickle',
            'ground_truth_positions_file_path': 'data/antolik/positions_reparametrized.pickle',
            'init_mu_range': 0.3,
            'init_sigma_range': 0.1,
            'init_to_ground_truth_orientations': True,
            'init_to_ground_truth_positions': True,
            'input_regularizer': 'LaplaceL2norm',
            'jackknife_oracle': True,
            'lr': 0.001,
            'max_epochs': 1,
            'max_time': 1,
            'model_needs_dataloader': False,
            'multivariate': True,
            'needs_ground_truth': True,
            'nonlinearity': 'softplus',
            'normalize': True,
            'num_bins': 100,
            'num_rotations': 4,
            'observed_val_metric': 'val/corr',
            'orientation_shift': 87.4,
            'patience': 5,
            'positions_minus_x': False,
            'positions_minus_y': True,
            'positions_swap_axes': False,
            'readout_bias': False,
            'readout_gamma': 0.17,
            'reg_group_sparsity': 0.1,
            'reg_readout_spatial_smoothness': 0.0027,
            'reg_spatial_sparsity': 0.45,
            'rot_eq_batch_norm': True,
            'sample': False,
            'scale_init': 0.3,
            'seed': 3017,
            'sigma_x_init': 0.56,
            'sigma_y_init': 0.67,
            'smooth_reg_weight': 0.0014451681045518333,
            'smoothness_reg_order': 3,
            'stack': -1,
            'stimulus_crop': None,
            'stride': 1,
            'test': True,
            'test_average_batch': False,
            'test_data_dir': '/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized_small/ten_trials.pickle',
            'train_data_dir': '/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized_small/one_trials.pickle',
            'train_on_test': False,
            'train_on_val': False,
            'upsampling': 2,
            'use_avg_reg': True,
            'val_size': 5000,
            'vmax': 100,
        }

    elif model == "BM":
        return {'batch_size': 10,
            'bottleneck_kernel': 5,
            'brain_crop': None,
            'compute_oracle_fraction': False,
            'conservative_oracle': True,
            'core_gamma_hidden': 0.008931320307500908,
            'core_gamma_input': 0.2384005754453638,
            'core_hidden_channels': 5,
            'core_hidden_kern': 5,
            'core_input_kern': 5,
            'core_layers': 4,
            'counter_clockwise_rotation': False,
            'dataset_artifact_name': 'Antolik_dataset:latest',
            'default_ori_shift': True,
            'depth_separable': True,
            'do_not_sample': True,
            'factor': 5.5,
            'fixed_sigma': False,
            'freeze_orientations': False,
            'freeze_positions': False,
            'generate_oracle_figure': False,
            'ground_truth_orientations_file_path': 'data/antolik/oris_reparametrized.pickle',
            'ground_truth_positions_file_path': 'data/antolik/positions_reparametrized.pickle',
            'init_mu_range': 0.3,
            'init_sigma_range': 0.1,
            'init_to_ground_truth_orientations': True,
            'init_to_ground_truth_positions': True,
            'input_regularizer': 'LaplaceL2norm',
            'jackknife_oracle': True,
            'lr': 0.0005,
            'max_epochs': 10,
            'max_time': 5,
            'model_needs_dataloader': True,
            'needs_ground_truth': False,
            'nonlinearity': 'softplus',
            'normalize': True,
            'num_rotations': 64,
            'observed_val_metric': 'val/corr',
            'orientation_shift': 87.4,
            'patience': 7,
            'positions_minus_x': False, # this says that dataset has to put minus in front of x when loading the ground truth
            'positions_minus_y': True, # this says that dataset has to put minus in front of y when loading the ground truth
            'positions_swap_axes': False,
            'readout_bias': False,
            'readout_gamma': 0.17,
            'reg_group_sparsity': 0.1,
            'reg_readout_spatial_smoothness': 0.0027,
            'reg_spatial_sparsity': 0.45,
            'rot_eq_batch_norm': True,
            'sample': False,
            'seed': 42,
            'stack': -1,
            'stimulus_crop': None,
            'stride': 1,
            'test': True,
            'test_average_batch': False,
            'test_data_dir': '/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized_small/ten_trials.pickle',
            'train_data_dir': '/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized_small/one_trials.pickle',
            'train_on_test': False,
            'train_on_val': False,
            'upsampling': 1,
            'use_avg_reg': True,
            'val_size': 5000
        }
    elif model == 'encoder':
        return{
            'batch_size': 100,
            'bottleneck_kernel': 15,
            'brain_crop': None,
            'compute_oracle_fraction': False,
            'conservative_oracle': True,
            'core_gamma_hidden': 0.5920399721780876,
            'core_gamma_input': 0.12141432993956784,
            'core_hidden_channels': 56,
            'core_hidden_kern': 9,
            'core_input_kern': 15,
            'core_layers': 5,
            'counter_clockwise_rotation': False,
            'dataset_artifact_name': 'Antolik_dataset:latest',
            'default_ori_shift': True,
            'depth_separable': True,
            'do_not_sample': False,
            'factor': 5.5,
            'fixed_sigma': False,
            'freeze_orientations': False,
            'freeze_positions': False,
            'generate_oracle_figure': False,
            'ground_truth_orientations_file_path': 'data/antolik/oris_reparametrized.pickle',
            'ground_truth_positions_file_path': 'data/antolik/positions_reparametrized.pickle',
            'init_mu_range': 0.5,
            'init_sigma_range': 0.001,
            'init_to_ground_truth_orientations': False,
            'init_to_ground_truth_positions': True,
            'input_channels': 48,
            'input_regularizer': 'LaplaceL2norm',
            'jackknife_oracle': True,
            'lr': 0.01,
            'max_epochs': 180,
            'max_time': 1,
            'model_needs_dataloader': True,
            'needs_ground_truth': False,
            'nonlinearity': 'softplus',
            'normalize': True,
            'num_rotations': 4,
            'observed_val_metric': 'val/corr',
            'orientation_shift': 87.4,
            'patience': 5,
            'positions_minus_x': False,
            'positions_minus_y': True,
            'positions_swap_axes': False,
            'readout_bias': True,
            'readout_gamma': 0.0018839045211725997,
            'reg_group_sparsity': 0.1,
            'reg_readout_spatial_smoothness': 0.0027,
            'reg_spatial_sparsity': 0.45,
            'rot_eq_batch_norm': True,
            'sample': False,
            'seed': 42,
            'stack': -3,
            'stimulus_crop': None,
            'stride': 1,
            'test': True,
            'test_average_batch': False,
            'test_data_dir': '/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized_small/ten_trials.pickle',
            'train_data_dir': '/storage/brno2/home/mpicek/reCNN_visual_prosthesis/data/antolik_reparametrized_small/one_trials.pickle',
            'train_on_test': False,
            'train_on_val': False,
            'upsampling': 2,
            'use_avg_reg': True,
            'val_size': 5000
        }

    raise ValueError("Unknown model: {}".format(model))




















