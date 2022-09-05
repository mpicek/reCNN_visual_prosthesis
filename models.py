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

from readout import Gaussian3dCyclic
from core import RotationEquivariant2dCoreBottleneck

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
        self.corr = Corr()
        self.save_hyperparameters()

    def regularization(self):
        return 0

    def training_step(self, batch, batch_idx):
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
        """
        - We just get prediction and return them with target. Later in validation_epoch_end,
            we compute the correlation on the whole validation set (and not on separate
            batches with final averaging)
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
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return opt

    def test_epoch_end(self, test_outs):
        """
        We compute the correlation on the whole set. Predictions with target
        responses are in test_outs (= what each test_step() returned)
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
        """
        We compute the correlation on the whole set. Predictions with target
        responses are in val_outs (= what each val_step() returned)
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


class reCNN_FullFactorized(ExtendedEncodingModel):
    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self.loss = PoissonLoss(avg=True)
        self.corr = Corr()
        self.nonlinearity = self.config["nonlinearity"]

        self.core = cores.RotationEquivariant2dCore(
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
        )

        self.readout = readouts.FullFactorized2d(
            in_shape=(
                self.config["core_hidden_channels"]
                * self.config["num_rotations"]
                * abs(self.config["stack"]),
                self.config[
                    "input_size_x"
                ],  # ocividne se to padduje, takze to neztraci rozmery
                self.config["input_size_y"],
            ),
            outdims=self.config["num_neurons"],
            bias=self.config["readout_bias"],
            mean_activity=self.config["mean_activity"],
            spatial_and_feature_reg_weight=self.config["readout_gamma"],
        )

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]

    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x

    def __str__(self):
        return "reCNN_FullFactorized2d"

    def reg_readout_spatial_smoothness(self):
        nw = self.readout.normalized_spatial()
        reg_term = torch.sqrt(
            torch.sum(
                torch.pow(
                    nn.functional.conv2d(
                        nw.reshape(
                            self.config["num_neurons"],
                            1,
                            self.config["input_size_x"],
                            self.config["input_size_y"],
                        ),
                        self.laplace,
                        padding="same",
                    ),
                    2,
                )
            )
        )
        reg_term = self.config["reg_readout_spatial_smoothness"] * reg_term
        return reg_term

    def reg_readout_group_sparsity(self):
        nw = self.readout.normalized_spatial().reshape(self.config["num_neurons"], -1)
        reg_term = self.config["reg_group_sparsity"] * torch.sum(
            torch.sqrt(torch.sum(torch.pow(nw, 2), dim=-1)), 0
        )
        return reg_term

    def reg_readout_spatial_sparsity(self):
        nw = self.readout.normalized_spatial()
        reg_term = self.config["reg_spatial_sparsity"] * torch.abs(nw).mean()
        self.log("reg/readout_spatial_sparsity", reg_term)
        return reg_term

    def regularization(self):

        readout_l1_reg = self.readout.regularizer(reduction="mean")
        self.log("reg/readout_l1_reg", readout_l1_reg)
        spatial_smoothness = self.reg_readout_spatial_smoothness()
        group_sparsity = self.reg_readout_group_sparsity()
        spatial_sparsity = self.reg_readout_spatial_sparsity()
        self.log("reg/readout_spatial_smoothness", spatial_smoothness)
        self.log("reg/readout_group_sparsity", group_sparsity)
        self.log("reg/readout_spatial_sparsity", spatial_sparsity)

        readout_reg = (
            readout_l1_reg + spatial_smoothness + group_sparsity + spatial_sparsity
        )
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term


class reCNN_Gauss2D(ExtendedEncodingModel):
    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self.loss = PoissonLoss(avg=True)
        self.corr = Corr()
        self.nonlinearity = self.config["nonlinearity"]

        self.core = RotationEquivariant2dCore(
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
        )

        self.readout = readouts.FullGaussian2d(
            in_shape=(
                self.config["core_hidden_channels"]
                * self.config["num_rotations"]
                * abs(self.config["stack"]),
                self.config["input_size_x"],
                self.config["input_size_y"],
            ),
            outdims=self.config["num_neurons"],
            bias=self.config["readout_bias"],
            mean_activity=self.config["mean_activity"],
            feature_reg_weight=self.config["readout_gamma"],
        )

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]

    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x

    def __str__(self):
        return "reCNN_FullGaussian2d"

    def regularization(self):

        readout_l1_reg = self.readout.regularizer(reduction="mean")
        self.log("reg/readout_l1_reg", readout_l1_reg)
        readout_reg = readout_l1_reg
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term


class reCNN_bottleneck_Gauss2d(ExtendedEncodingModel):
    """
    DNN network composed of reCNN core, bottleneck at the end,
    and also a FullGaussian2d readout
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

        self.readout = readouts.FullGaussian2d(
            in_shape=(
                self.config["num_rotations"],
                self.config["input_size_x"],
                self.config["input_size_y"],
            ),
            outdims=self.config["num_neurons"],
            bias=self.config["readout_bias"],
            mean_activity=self.config["mean_activity"],
            feature_reg_weight=self.config["readout_gamma"],
        )

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]

    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x

    def __str__(self):
        return "reCNN_bottleneck_Gauss2d"

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


class reCNN_bottleneck_NoReadout(ExtendedEncodingModel):
    """
    Lurz's model with RotEq core and with bottleneck
    No Readout is present here. It is used to test that the features are
        really rotation equivariant (in bottleneck_test.py)
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

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]

    def forward(self, x):
        x = self.core(x)
        return x

    def __str__(self):
        return "reCNN_bottleneck_NoReadout"

    def regularization(self):

        readout_reg = 0
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        return reg_term


class LurzReimplementation(ExtendedEncodingModel):
    """Lurz's model"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self.loss = PoissonLoss(avg=True)
        self.corr = Corr()
        self.nonlinearity = self.config["nonlinearity"]

        self.core = cores.SE2dCore(
            stride=self.config["stride"],
            input_regularizer=self.config["input_regularizer"],
            input_channels=self.config["input_channels"],
            hidden_channels=self.config["core_hidden_channels"],
            input_kern=self.config["core_input_kern"],
            hidden_kern=self.config["core_hidden_kern"],
            layers=self.config["core_layers"],
            gamma_input=config["core_gamma_input"],  # 0
            gamma_hidden=config["core_gamma_hidden"],  # 0
            stack=config["stack"],
            depth_separable=config["depth_separable"],
            use_avg_reg=config["use_avg_reg"],
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
        )

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]

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


class reCNN_bottleneck_CyclicGauss3d(ExtendedEncodingModel):
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
        )

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]

    def forward(self, x):
        self.log("train/sigma1", self.readout.sigma[0, 0, 0, 0, 1])
        self.log("train/sigma1b", self.readout.sigma[0, 0, 1, 0, 1])
        self.log("train/sigma2", self.readout.sigma[0, 0, 0, 0, 1])
        self.log("train/sigma2b", self.readout.sigma[0, 0, 1, 0, 1])
        self.log("train/sigma3", self.readout.sigma[0, 0, 0, 0, 1])
        self.log("train/sigma3b", self.readout.sigma[0, 0, 1, 0, 1])
        # print(self.readout.sigma.shape)
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


class Lurz_Baseline(ExtendedEncodingModel):
    """Lurz's model used as a baseline model"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        # self.loss = PoissonLoss(avg=True)
        # self.corr = Corr()
        self.nonlinearity = self.config["nonlinearity"]
        

        self.core = cores.SE2dCore(
            stride=self.config["stride"],
            input_regularizer=self.config["input_regularizer"],
            input_channels=self.config["input_channels"],
            hidden_channels=self.config["core_hidden_channels"],
            input_kern=self.config["core_input_kern"],
            hidden_kern=self.config["core_hidden_kern"],
            layers=self.config["core_layers"],
            gamma_input=config["core_gamma_input"], # 0
            gamma_hidden=config["core_gamma_hidden"], # 0
            stack=config["stack"],
            depth_separable=config["depth_separable"],
            use_avg_reg=config["use_avg_reg"]
        )

        
        self.readout = readouts.FullGaussian2d(
            in_shape=( #TODO: stack???
                #TODO: ten shape si potvrdit
                self.config["core_hidden_channels"] * abs(self.config["stack"]),
                self.config["input_size_x"], # ocividne se to padduje, takze to neztraci rozmery
                self.config["input_size_y"],
            ),
            outdims=self.config["num_neurons"],
            bias=self.config["readout_bias"],
            mean_activity=self.config["mean_activity"],
            feature_reg_weight=self.config["readout_gamma"],
        )

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()[config["nonlinearity"]]

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