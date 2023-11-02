from predict_neural_responses.models import *
from utils import get_fraction_oracles
import predict_neural_responses.dnn_blocks.dnn_blocks as bl

from torch import nn
import torch
from collections import OrderedDict

from neuralpredictors.layers.hermite import (
    HermiteConv2D,
)
from neuralpredictors.measures.np_functions import corr as corr_from_neuralpredictors
from neuralpredictors.measures.np_functions import (
    oracle_corr_jackknife,
    oracle_corr_conservative,
)
from neuralpredictors.layers.cores.conv2d import RotationEquivariant2dCore

import logging

from readout import Gaussian3dCyclic, Gaussian3dCyclicNoScale
from core import RotationEquivariant2dCoreBottleneck
import matplotlib.pyplot as plt
from experiments.utils import reconstruct_orientation_maps, get_neuron_estimates


logger = logging.getLogger(__name__)


class ExtendedEncodingModel(encoding_model):
    """Parent class for system identification enconding models, keeps track of useful metrics

    In config:
        - test_average_batch: whether to average responses in batches when computing
            the test set correlation (used in repeated trials to cancel the neural variability)
        - compute_oracle_fraction: whether to compute oracle fraction or not
        - conservative_oracle: whether to compute conservative oracle or not
        - jackknife_oracle: whether to compute jackknife oracle or not
        - generate_oracle_figure: whether to generate a figure of fitted line
            describing the oracle fraction or not
    """

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.test_average_batch = config["test_average_batch"]
        self.compute_oracle_fraction = config["compute_oracle_fraction"]
        self.conservative_oracle = config["conservative_oracle"]
        self.jackknife_oracle = config["jackknife_oracle"]
        self.generate_oracle_figure = config["generate_oracle_figure"]
        self.loss = PoissonLoss(avg=True)
        # self.loss = torch.nn.MSELoss()
        self.corr = Corr()
        self.save_hyperparameters()

    def regularization(self):
        return 0

    def training_step(self, batch, batch_idx):
        """Defines what to do at each training step.
        Gets the batch, passes it through the network, updates weights,
        computes loss and regularization, logs important metrics and
        returns regularized loss.

        Args:
            batch (tuple): tuple of (imgs, responses)
            batch_idx (int): Index of the batch

        Returns:
            float: Regularized loss
        """
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        reg_term = self.regularization()
        regularized_loss = loss + reg_term
        self.log("train/unregularized_loss", loss)
        self.log("train/regularization", reg_term)
        self.log("train/regularized_loss", regularized_loss)
        return regularized_loss

    def validation_step(self, batch, batch_idx):
        """Defines what to do at each validation step.
        We just get prediction and return them with target. Later in self.validation_epoch_end,
        we compute the correlation on the whole validation set (and not on separate
        batches with final averaging)

        Args:
            batch (tuple): tuple of (imgs, responses)
            batch_idx (int): Index of the batch

        Returns:
            tuple: (prediction of responses, true responses)
        """        

        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        self.log("val/loss", loss)

        return prediction, resp

    def test_step(self, batch, batch_idx):
        """
        - If self.test_average_batch == True, then we average the responses of
            the batch (because it is the same image shown multiple times to cancel
            out the noise)

        - We just get prediction and return them with target. Later in validation_epoch_end,
            we compute the correlation on the whole validation set (and not on separate
            batches with final averaging).

        Args:
            batch (tuple): tuple of (imgs, responses). The images might be all the
                same in case of the oracle dataset for evaluation of the averaged trial correlation.
            batch_idx (int): Index of the batch

        Returns:
            tuple: (prediction of responses, true responses of each trial (might be averaged), true responses of each trial (never averaged))
        """

        img, resp = batch
        responses_no_mean = resp

        if self.test_average_batch:
            # I take only one image as all images are the same (it is a repeated trial)
            # .unsqueeze(0) adds one dimension at the beginning (because I need
            # to create a batch of size 1)
            img = img[0].unsqueeze(0)
            resp = resp.mean(0).unsqueeze(0)

        prediction = self.forward(img)
        return prediction, resp, responses_no_mean

    def configure_optimizers(self):
        """Configures the optimizer for the training of the model (Adam).

        Returns:
            torch.optimizer: torch optimizer class
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return opt

    def test_epoch_end(self, test_outs):
        """We compute a correlation on the WHOLE test set. Predictions with target
        responses are in test_outs (= what each self.test_step() returned)

        Args:
            test_outs (list): What each self.test_step() returned
        """
        pred = []
        resp = []
        batch_of_responses = []
        num_of_repeats = None
        for (p, r, r_batches) in test_outs:

            # The number of repeats in the repeated trials have to be the same.
            # We will use the first batch as an indicator, how many trials should
            # be in every batch. If some batch does not have the same number of
            # repeats, we discard the batch
            if num_of_repeats == None:
                num_of_repeats = r_batches.shape[0]

            pred.append(p.detach().cpu().numpy())
            resp.append(r.detach().cpu().numpy())

            if (
                r_batches.shape[0] == num_of_repeats and self.compute_oracle_fraction
            ):  # does not have the appropriate number of repeats
                batch_of_responses.append(r_batches.detach().cpu().numpy())

        predictions = np.concatenate(pred)
        responses = np.concatenate(resp)
        correlation = corr_from_neuralpredictors(predictions, responses, axis=0)

        batches_of_responses = None
        if self.compute_oracle_fraction:
            batches_of_responses = np.stack(batch_of_responses)

        if self.test_average_batch:
            self.log("test/repeated_trials/corr", np.mean(correlation))
        else:
            self.log("test/corr", np.mean(correlation))

        if self.compute_oracle_fraction:
            if self.jackknife_oracle:
                oracles = oracle_corr_jackknife(batches_of_responses)
                fraction_of_oracles = get_fraction_oracles(
                    oracles,
                    correlation,
                    generate_figure=self.generate_oracle_figure,
                    oracle_label="Oracles jackknife",
                    fig_name="oracle_jackknife.png",
                )
                self.log("test/fraction_oracle_jackknife", fraction_of_oracles[0])

            if self.conservative_oracle:
                oracles = oracle_corr_conservative(batches_of_responses)
                fraction_of_oracles = get_fraction_oracles(
                    oracles,
                    correlation,
                    generate_figure=self.generate_oracle_figure,
                    oracle_label="Oracles conservative",
                    fig_name="oracle_conservative.png",
                )
                self.log("test/fraction_oracle_conservative", fraction_of_oracles[0])

    def validation_epoch_end(self, val_outs):
        """We compute the correlation on the whole set. Predictions with target
        responses are in val_outs (= what each val_step() returned)

        Args:
            val_outs (list): What each self.validation_step() returned
        """        
        pred = []
        resp = []
        for (p, r) in val_outs:
            pred.append(p.detach().cpu().numpy())
            resp.append(r.detach().cpu().numpy())

        predictions = np.concatenate(pred)
        responses = np.concatenate(resp)
        correlation = corr_from_neuralpredictors(predictions, responses, axis=0)
        self.log("val/corr", np.mean(correlation))


class reCNN_bottleneck_CyclicGauss3d_individual_neuron_scaling(ExtendedEncodingModel):
    """
        Like the main model of this repository, but with individual neuron scaling in the readout.
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self.nonlinearity = self.config["nonlinearity"]

        self.hidden_padding = None
        assert self.config["stack"] == -1

        self.core = RotationEquivariant2dCoreBottleneck(
            num_rotations=self.config["num_rotations"],
            stride=self.config["stride"],
            upsampling=self.config["upsampling"],
            rot_eq_batch_norm=self.config["rot_eq_batch_norm"],
            input_regularizer=self.config["input_regularizer"],
            input_channels=self.config["input_channels"],
            hidden_channels=self.config["core_hidden_channels"],
            input_kern=self.config["core_input_kern"],
            hidden_kern=self.config["core_hidden_kern"],
            layers=self.config["core_layers"],
            gamma_input=config["core_gamma_input"],
            gamma_hidden=config["core_gamma_hidden"],
            stack=config["stack"],
            depth_separable=config["depth_separable"],
            use_avg_reg=config["use_avg_reg"],
            bottleneck_kernel=config["bottleneck_kernel"],
        )

        self.readout = Gaussian3dCyclic(
            in_shape=(
                self.config["num_rotations"],
                self.config["input_size_x"],
                self.config["input_size_y"],
            ),
            outdims=self.config["num_neurons"],
            bias=self.config["readout_bias"],
            mean_activity=self.config["mean_activity"],
            feature_reg_weight=self.config["readout_gamma"],
            init_sigma_range=self.config["init_sigma_range"],
            init_mu_range=self.config["init_mu_range"],
            fixed_sigma=self.config["fixed_sigma"],
            do_not_sample=self.config["do_not_sample"],
        )

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]

    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x

    def __str__(self):
        return "reCNN_bottleneck_CyclicGauss3d_individual_neuron_scaling"

    def add_bottleneck(self):

        layer = OrderedDict()

        if self.hidden_padding is None:
            self.hidden_padding = self.bottleneck_kernel // 2

        layer["hermite_conv"] = HermiteConv2D(
            input_features=self.config["hidden_channels"]
            * self.config["num_rotations"],
            output_features=1,
            num_rotations=self.config["num_rotations"],
            upsampling=self.config["upsampling"],
            filter_size=self.config["bottleneck_kernel"],
            stride=self.config["stride"],
            padding=self.hidden_padding,
            first_layer=False,
        )
        super().add_bn_layer(layer)
        super().add_activation(layer)
        super().features.add_module("bottleneck", nn.Sequential(layer))

    def regularization(self):

        readout_l1_reg = self.readout.regularizer(reduction="mean")
        self.log("reg/readout_l1_reg", readout_l1_reg)

        readout_reg = readout_l1_reg

        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term
    
    def visualize_orientation_map(self, ground_truth_positions_file_path, ground_truth_orientations_file_path, save=False, img_path="img/", suffix="_truth", neuron_dot_size=5):
        
        fig, ax = plt.subplots()
        x, y, o = get_neuron_estimates(self, 5.5)
        reconstruct_orientation_maps(x, y, o, fig, ax, save, 12, 2.4, 2.4, img_path, suffix, neuron_dot_size)


class reCNN_bottleneck_CyclicGauss3d_no_scaling(ExtendedEncodingModel):
    """
        The main model of this repository.
        This model consists of:
            - a core with reCNN architecture with a bottleneck in the last layer
              to return only one scalar value for each position and orientation
              (meaning that the number of channels in the last layer is limited
              to 1)
            - a readout which is a Gaussian 3d readout but modified in a way
              that ensures that the third dimension (= orientation dimension)
              is periodic
    """

    def __init__(self, **config):
        """As this network can be initialized to the ground truth positions and orientations,
        we need a reference to the dataloader from which this ground truth will be provided.

        """
        super().__init__(**config)
        self.config = config
        self.nonlinearity = self.config["nonlinearity"]

        self.hidden_padding = None
        assert self.config["stack"] == -1

        self.core = RotationEquivariant2dCoreBottleneck(
            num_rotations=self.config["num_rotations"],
            stride=self.config["stride"],
            upsampling=self.config["upsampling"],
            rot_eq_batch_norm=self.config["rot_eq_batch_norm"],
            input_regularizer=self.config["input_regularizer"],
            input_channels=self.config["input_channels"],
            hidden_channels=self.config["core_hidden_channels"],
            input_kern=self.config["core_input_kern"],
            hidden_kern=self.config["core_hidden_kern"],
            layers=self.config["core_layers"],
            gamma_input=config["core_gamma_input"],
            gamma_hidden=config["core_gamma_hidden"],
            stack=config["stack"],
            depth_separable=config["depth_separable"],
            use_avg_reg=config["use_avg_reg"],
            bottleneck_kernel=config["bottleneck_kernel"],
        )

        self.readout = Gaussian3dCyclicNoScale(
            in_shape=(
                self.config["num_rotations"],
                self.config["input_size_x"],
                self.config["input_size_y"],
            ),
            outdims=self.config["num_neurons"],
            bias=self.config["readout_bias"],
            mean_activity=self.config["mean_activity"],
            feature_reg_weight=self.config["readout_gamma"],
            init_sigma_range=self.config["init_sigma_range"],
            init_mu_range=self.config["init_mu_range"],
            fixed_sigma=self.config["fixed_sigma"],
            ground_truth_positions_file_path=config["ground_truth_positions_file_path"],
            ground_truth_orientations_file_path=config["ground_truth_orientations_file_path"],
            init_to_ground_truth_positions=config["init_to_ground_truth_positions"],
            init_to_ground_truth_orientations=config["init_to_ground_truth_orientations"],
            freeze_positions=config["freeze_positions"],
            freeze_orientations=config["freeze_orientations"],
            orientation_shift=config["orientation_shift"], #in degrees
            factor = config["factor"],
            filtered_neurons = config["filtered_neurons"],
            positions_minus_x = config["positions_minus_x"],
            positions_minus_y = config["positions_minus_y"],
            do_not_sample = config["do_not_sample"],
        )

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]
    
    def init_neurons(self, dataloader=None):
        self.readout.init_neurons(dataloader)

    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x

    def __str__(self):
        return "reCNN_bottleneck_CyclicGauss3d"

    def add_bottleneck(self):

        layer = OrderedDict()

        if self.hidden_padding is None:
            self.hidden_padding = self.bottleneck_kernel // 2

        layer["hermite_conv"] = HermiteConv2D(
            input_features=self.config["hidden_channels"]
            * self.config["num_rotations"],
            output_features=1,
            num_rotations=self.config["num_rotations"],
            upsampling=self.config["upsampling"],
            filter_size=self.config["bottleneck_kernel"],
            stride=self.config["stride"],
            padding=self.hidden_padding,
            first_layer=False,
        )
        super().add_bn_layer(layer)
        super().add_activation(layer)
        super().features.add_module("bottleneck", nn.Sequential(layer))

    def regularization(self):

        readout_l1_reg = self.readout.regularizer(reduction="mean")
        self.log("reg/readout_l1_reg", readout_l1_reg)

        readout_reg = readout_l1_reg

        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term
    
    def visualize_orientation_map(self, ground_truth_positions_file_path, ground_truth_orientations_file_path, save=False, img_path="img/", suffix="_truth", neuron_dot_size=5, factor=5.5, shift=0, swap_y_axis=False, neuron_id="all"):
        """shift is in degrees"""

        shift = (shift * np.pi) / 180 # from degrees to radians
        
        fig, ax = plt.subplots()
        x, y, o = get_neuron_estimates(self, factor)
        o = [i*np.pi for i in o]
        o = [(i + shift)%np.pi for i in o]
        # x, y, o = self.get_ground_truth(ground_truth_positions_file_path, ground_truth_orientations_file_path)

        if neuron_id != "all":
            x = x[neuron_id]
            y = y[neuron_id]
            o = o[neuron_id]
        
        print(x)
        print(y)
        print(o)
        
        if swap_y_axis:
            reconstruct_orientation_maps(x, -y, o, fig, ax, save, 12, 5.5, 5.5, img_path, suffix, neuron_dot_size)
        else:
            reconstruct_orientation_maps(x, y, o, fig, ax, save, 12, 5.5, 5.5, img_path, suffix, neuron_dot_size)


# needs_data_loader=True
class Lurz_Control_Model(ExtendedEncodingModel):
    """Lurz's model used as a control model"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self.nonlinearity = self.config["nonlinearity"]
        

        self.core = cores.SE2dCore(
            stride=self.config["stride"],
            input_regularizer=self.config["input_regularizer"],
            input_channels=self.config["input_channels"],
            hidden_channels=self.config["core_hidden_channels"],
            input_kern=self.config["core_input_kern"],
            hidden_kern=self.config["core_hidden_kern"],
            layers=self.config["core_layers"],
            gamma_input=config["core_gamma_input"],
            gamma_hidden=config["core_gamma_hidden"],
            stack=config["stack"],
            depth_separable=config["depth_separable"],
            use_avg_reg=config["use_avg_reg"]
        )

        
        self.readout = readouts.FullGaussian2d(
            in_shape=(
                self.config["core_hidden_channels"] * abs(self.config["stack"]),
                self.config["input_size_x"],
                self.config["input_size_y"],
            ),
            outdims=self.config["num_neurons"],
            bias=self.config["readout_bias"],
            mean_activity=self.config["mean_activity"],
            feature_reg_weight=self.config["readout_gamma"],
            init_sigma=self.config["init_sigma_range"],
        )

        self.init_to_ground_truth_positions = config["init_to_ground_truth_positions"]
        self.ground_truth_positions_file_path = config["ground_truth_positions_file_path"]
        self.ground_truth_orientations_file_path = config["ground_truth_orientations_file_path"]
        self.positions_minus_x = config["positions_minus_x"]
        self.positions_minus_y = config["positions_minus_y"]
        self.do_not_sample = config["do_not_sample"]
        self.positions_swap_axes = config["positions_swap_axes"]
        
        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]


    def init_neurons(self, dataloader=None):

        if self.init_to_ground_truth_positions == True:
            print("initializing to ground truth")
            pos_x, pos_y, _ = dataloader.get_ground_truth(ground_truth_positions_file_path=self.ground_truth_positions_file_path, ground_truth_orientations_file_path=self.ground_truth_orientations_file_path, in_degrees=True, positions_minus_y=self.positions_minus_y, positions_minus_x=self.positions_minus_x, positions_swap_axes=self.positions_swap_axes)
            pos_x = torch.from_numpy(pos_x)
            pos_y = torch.from_numpy(pos_y)
            # works also when the stimulus is cropped (self.get_stimulus_visual_angle()
            # returns the visual angle corrected after the stimulus crop)
            self.readout._mu.data[0,:,0,0] = pos_x / (dataloader.get_stimulus_visual_angle() / 2)        
            self.readout._mu.data[0,:,0,1] = pos_y / (dataloader.get_stimulus_visual_angle() / 2)

    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x
    
    def __str__(self):
        return "StackedCore_FullGaussian2d"

    def reg_readout_group_sparsity(self):
        nw = self.readout.features.reshape(self.config["num_neurons"], -1)
        reg_term = self.config["reg_group_sparsity"] * torch.sum(
            torch.sqrt(torch.sum(torch.pow(nw, 2), dim=-1)), 0
        )
        return reg_term

    def regularization(self):

        readout_l1_reg = self.readout.regularizer(reduction="mean")
        self.log("reg/readout_l1_reg", readout_l1_reg)

        readout_reg = readout_l1_reg
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term
