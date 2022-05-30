from predict_neural_responses.models import *
import pytorch_lightning as pl
import predict_neural_responses.dnn_blocks.dnn_blocks as bl

from collections import OrderedDict, Iterable
from functools import partial

from torch import nn
import torch

from neuralpredictors.layers.cores.base import Core
from neuralpredictors.layers.cores.conv2d import Stacked2dCore
from neuralpredictors import regularizers
from neuralpredictors.layers.conv import DepthSeparableConv2d
from neuralpredictors.layers.squeeze_excitation import SqueezeExcitationBlock
from neuralpredictors.layers.attention import AttentionConv
from neuralpredictors.layers.affine import Bias2DLayer, Scale2DLayer
from neuralpredictors.layers.activations import AdaptiveELU
from neuralpredictors.layers.hermite import (
    HermiteConv2D,
    RotationEquivariantBatchNorm2D,
    RotationEquivariantBias2DLayer,
    RotationEquivariantScale2DLayer,
)
from neuralpredictors.measures.np_functions import corr as corr_from_neuralpredictors
from neuralpredictors.layers.cores.conv2d import RotationEquivariant2dCore

import logging

logger = logging.getLogger(__name__)


class encoding_model_fixed(encoding_model):
    """Parent class for system identification enconding models, keeps track of useful metrics"""

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.test_average_batch = config["test_average_batch"]
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

        if self.test_average_batch:
            # I take only one image as all images are the same (it is a repeated trial)
            # .unsqueeze(0) adds one dimension at the beginning (because I need
            # to create a batch of size 1)
            img = img[0].unsqueeze(0)
            resp = resp.mean(0).unsqueeze(0)

        prediction = self.forward(img)
        return prediction, resp

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
        for (p, r) in test_outs:
            pred.append(p.detach().cpu().numpy())
            resp.append(r.detach().cpu().numpy())
        
        predictions = np.concatenate(pred)
        responses = np.concatenate(resp)
        correlation = corr_from_neuralpredictors(predictions, responses, axis=0)
        self.log("test/corr", np.mean(correlation))
    
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


class Picek(encoding_model_fixed):
    """My model with FullFactorized2d readout"""

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
            use_avg_reg=config["use_avg_reg"]
        )

        self.readout = readouts.FullFactorized2d(
            in_shape=( #TODO: stack???
                #TODO: ten shape si potvrdit
                self.config["core_hidden_channels"] * self.config["num_rotations"] * abs(self.config["stack"]),
                self.config["input_size_x"], # ocividne se to padduje, takze to neztraci rozmery
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
        return "RotEq_FullFactorized2d"
    
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
        # is this already enforced somehow?
        nw = self.readout.normalized_spatial()
        reg_term = self.config["reg_spatial_sparsity"] * torch.abs(nw).mean()
        self.log("reg/readout_spatial_sparsity", reg_term)
        return reg_term

    def regularization(self):
        # readout
        readout_l1_reg = self.readout.regularizer(reduction="mean")
        self.log("reg/readout_l1_reg", readout_l1_reg)
        readout_reg = 0
        for k in self.config.keys():
            if k.startswith("reg_readout"):
                kreg = getattr(self, k)()
                readout_reg = readout_reg + kreg
        # core
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_l1_reg)
        return reg_term

class PicekGauss(encoding_model_fixed):
    """My model with Gaussian readout"""

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
            use_avg_reg=config["use_avg_reg"]
        )

        self.readout = readouts.FullGaussian2d(
            in_shape=( #TODO: stack???
                #TODO: ten shape si potvrdit
                self.config["core_hidden_channels"] * self.config["num_rotations"] * abs(self.config["stack"]),
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
        return "RotEq_FullGaussian2d"
    
    def reg_readout_spatial_smoothness(self):
        nw = self.readout.features
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
        nw = self.readout.features.reshape(self.config["num_neurons"], -1)
        reg_term = self.config["reg_group_sparsity"] * torch.sum(
            torch.sqrt(torch.sum(torch.pow(nw, 2), dim=-1)), 0
        )
        return reg_term

    def reg_readout_spatial_sparsity(self):
        # is this already enforced somehow?
        nw = self.readout.features
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

        readout_reg = readout_l1_reg + spatial_smoothness + group_sparsity + spatial_sparsity
        # for k in self.config.keys():
        #     # print(k)
        #     if k.startswith("reg_readout"):
        #         kreg = getattr(self, k)()
        #         readout_reg = readout_reg + kreg
        # core
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term

class LOL(encoding_model_fixed):
    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
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
            use_avg_reg=config["use_avg_reg"]
        )

        self.readout = readouts.FullFactorized2d(
            in_shape=( #TODO: stack???
                #TODO: ten shape si potvrdit
                self.config["core_hidden_channels"] * self.config["num_rotations"] * abs(self.config["stack"]),
                self.config["input_size_x"], # ocividne se to padduje, takze to neztraci rozmery
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
        # x = self.nonlin(self.readout(x))
        return x
    
    def __str__(self):
        return "Stacked_FullFactorized2d"
    
    def reg_readout_spatial_smoothness(self):
        nw = self.readout.normalized_spatial
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
        nw = self.readout.normalized_spatial.reshape(self.config["num_neurons"], -1)
        reg_term = self.config["reg_group_sparsity"] * torch.sum(
            torch.sqrt(torch.sum(torch.pow(nw, 2), dim=-1)), 0
        )
        return reg_term

    def reg_readout_spatial_sparsity(self):
        # is this already enforced somehow?
        nw = self.readout.normalized_spatial
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

        readout_reg = readout_l1_reg + spatial_smoothness + group_sparsity + spatial_sparsity
        # for k in self.config.keys():
        #     # print(k)
        #     if k.startswith("reg_readout"):
        #         kreg = getattr(self, k)()
        #         readout_reg = readout_reg + kreg
        # core
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term

class LurzRotEq(encoding_model_fixed):
    """Lurz's model with RotEq core"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        # self.loss = PoissonLoss(avg=True)
        # self.corr = Corr()
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
            use_avg_reg=config["use_avg_reg"]
        )

        self.readout = readouts.FullFactorized2d(
            in_shape=( #TODO: stack???
                #TODO: ten shape si potvrdit
                self.config["core_hidden_channels"] * self.config["num_rotations"] * abs(self.config["stack"]),
                self.config["input_size_x"], # ocividne se to padduje, takze to neztraci rozmery
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
        return "Stacked_FullFactorized2d"
    
    def reg_readout_spatial_smoothness(self):
        nw = self.readout.normalized_spatial
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
        nw = self.readout.normalized_spatial.reshape(self.config["num_neurons"], -1)
        reg_term = self.config["reg_group_sparsity"] * torch.sum(
            torch.sqrt(torch.sum(torch.pow(nw, 2), dim=-1)), 0
        )
        return reg_term

    def reg_readout_spatial_sparsity(self):
        # is this already enforced somehow?
        nw = self.readout.normalized_spatial
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

        readout_reg = readout_l1_reg + spatial_smoothness + group_sparsity + spatial_sparsity
        # for k in self.config.keys():
        #     # print(k)
        #     if k.startswith("reg_readout"):
        #         kreg = getattr(self, k)()
        #         readout_reg = readout_reg + kreg
        # core
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term


class LurzRotEqGauss(encoding_model_fixed):
    """Lurz's model with RotEq core"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        # self.loss = PoissonLoss(avg=True)
        # self.corr = Corr()
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
            use_avg_reg=config["use_avg_reg"]
        )


        self.readout = readouts.FullGaussian2d(
            # TODO: Important note: you don't pass the batch_size dimension!
            in_shape=( #TODO: stack???
                #TODO: ten shape si potvrdit
                self.config["core_hidden_channels"] * self.config["num_rotations"] * abs(self.config["stack"]),
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
        return "RotEq_FullGaussian2d"
    

    def reg_readout_group_sparsity(self):
        nw = self.readout.features.reshape(self.config["num_neurons"], -1)
        reg_term = self.config["reg_group_sparsity"] * torch.sum(
            torch.sqrt(torch.sum(torch.pow(nw, 2), dim=-1)), 0
        )
        return reg_term

    def regularization(self):

        readout_l1_reg = self.readout.regularizer(reduction="mean")
        self.log("reg/readout_l1_reg", readout_l1_reg)
        # group_sparsity = self.reg_readout_group_sparsity()
        # self.log("reg/readout_group_sparsity", group_sparsity)

        readout_reg = readout_l1_reg
        # for k in self.config.keys():
        #     # print(k)
        #     if k.startswith("reg_readout"):
        #         kreg = getattr(self, k)()
        #         readout_reg = readout_reg + kreg
        # core
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term

class RotEqBottleneckGauss(encoding_model_fixed):
    """Lurz's model with RotEq core"""

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
        return "RotEq_Bottleneck_FullGaussian2d"
    
    def add_bottleneck(self):

        layer = OrderedDict()

        if self.hidden_padding is None:
            self.hidden_padding = self.bottleneck_kernel // 2

        layer["hermite_conv"] = HermiteConv2D(
            input_features=self.config["hidden_channels"] * self.config["num_rotations"],
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

class RotEqBottleneckTest_NoGauss(encoding_model_fixed):
    """Lurz's model with RotEq core"""

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
        return "RotEqBottleneckTest_NoGauss"
    
    def add_bottleneck(self):

        layer = OrderedDict()

        if self.hidden_padding is None:
            self.hidden_padding = self.bottleneck_kernel // 2

        layer["hermite_conv"] = HermiteConv2D(
            input_features=self.config["hidden_channels"] * self.config["num_rotations"],
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

        readout_reg = 0
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        return reg_term

class LurzGauss(encoding_model_fixed):
    """Lurz's model"""

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


class Lurz(encoding_model_fixed):
    """Lurz's model"""

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

            gamma_input=config["core_gamma_input"], # 0
            gamma_hidden=config["core_gamma_hidden"], # 0
            
            stack=config["stack"],
            depth_separable=config["depth_separable"],
            use_avg_reg=config["use_avg_reg"]
        )
        

        self.readout = readouts.FullFactorized2d(
            in_shape=( #TODO: stack???
                self.config["core_hidden_channels"] * abs(self.config["stack"]),
                self.config["input_size_x"], # ocividne se to padduje, takze to neztraci rozmery
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
        return "Stacked_FullFactorized2d"
    
    def reg_readout_spatial_smoothness(self):
        nw = self.readout.normalized_spatial
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
        nw = self.readout.normalized_spatial.reshape(self.config["num_neurons"], -1)
        reg_term = self.config["reg_group_sparsity"] * torch.sum(
            torch.sqrt(torch.sum(torch.pow(nw, 2), dim=-1)), 0
        )
        return reg_term

    def reg_readout_spatial_sparsity(self):
        # is this already enforced somehow?
        nw = self.readout.normalized_spatial
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

        readout_reg = readout_l1_reg + spatial_smoothness + group_sparsity + spatial_sparsity
        # for k in self.config.keys():
        #     # print(k)
        #     if k.startswith("reg_readout"):
        #         kreg = getattr(self, k)()
        #         readout_reg = readout_reg + kreg
        # core
        core_reg = self.core.regularizer()
        reg_term = readout_reg + core_reg
        self.log("reg/core reg", core_reg)
        self.log("reg/readout_reg", readout_reg)
        return reg_term

class RotationEquivariant2dCoreBottleneck(Stacked2dCore, nn.Module):
    """
    A core built of 2d rotation-equivariant layers. For more info refer to https://openreview.net/forum?id=H1fU8iAqKX.
    """

    def __init__(
        self,
        *args,
        num_rotations=8,
        stride=1,
        upsampling=2,
        rot_eq_batch_norm=True,
        input_regularizer="LaplaceL2norm",
        bottleneck_kernel=3,
        bottleneck_padding=None,
        **kwargs,
    ):
        self.num_rotations = num_rotations
        self.stride = stride
        self.upsampling = upsampling
        self.rot_eq_batch_norm = rot_eq_batch_norm
        self.bottleneck_kernel = bottleneck_kernel
        self.bottleneck_padding = bottleneck_padding
        super().__init__(*args,
            **kwargs, input_regularizer=input_regularizer)

    def set_batchnorm_type(self):
        if not self.rot_eq_batch_norm:
            self.batchnorm_layer_cls = nn.BatchNorm2d
            self.bias_layer_cls = Bias2DLayer
            self.scale_layer_cls = Scale2DLayer
        else:
            self.batchnorm_layer_cls = partial(RotationEquivariantBatchNorm2D, num_rotations=self.num_rotations)
            self.bias_layer_cls = partial(RotationEquivariantBias2DLayer, num_rotations=self.num_rotations)
            self.scale_layer_cls = partial(RotationEquivariantScale2DLayer, num_rotations=self.num_rotations)

    def add_first_layer(self):
        layer = OrderedDict()
        layer["hermite_conv"] = HermiteConv2D(
            input_features=self.input_channels,
            output_features=self.hidden_channels,
            num_rotations=self.num_rotations,
            upsampling=self.upsampling,
            filter_size=self.input_kern,
            stride=self.stride,
            padding=self.input_kern // 2 if self.pad_input else 0,
            first_layer=True,
        )
        self.add_bn_layer(layer)
        self.add_activation(layer)
        self.features.add_module("layer0", nn.Sequential(layer))

    # def add_subsequent_layers(self):
    #     if not isinstance(self.hidden_kern, Iterable):
    #         self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

    #     for l in range(1, self.num_layers):
    #         layer = OrderedDict()

    #         if self.hidden_padding is None:
    #             self.hidden_padding = self.hidden_kern[l - 1] // 2

    #         layer["hermite_conv"] = HermiteConv2D(
    #             input_features=self.hidden_channels * self.num_rotations,
    #             output_features=self.hidden_channels,
    #             num_rotations=self.num_rotations,
    #             upsampling=self.upsampling,
    #             filter_size=self.hidden_kern[l - 1],
    #             stride=self.stride,
    #             padding=self.hidden_padding,
    #             first_layer=False,
    #         )
    #         self.add_bn_layer(layer)
    #         self.add_activation(layer)
    #         self.features.add_module("layer{}".format(l), nn.Sequential(layer))
    
    def add_bottleneck_bn_layer(self, layer):
        """
        Same as add_bn_layer(), but self.hidden_channels are replaced by 1
        """
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = self.batchnorm_layer_cls(1, momentum=self.momentum)
            else:
                layer["norm"] = self.batchnorm_layer_cls(
                    1, momentum=self.momentum, affine=self.bias and self.batch_norm_scale
                )
                if self.bias:
                    if not self.batch_norm_scale:
                        layer["bias"] = self.bias_layer_cls(1)
                elif self.batch_norm_scale:
                    layer["scale"] = self.scale_layer_cls(1)

    def add_bottleneck_layer(self):
        """
        Adds the bottleneck layer
        """

        layer = OrderedDict()

        if self.bottleneck_padding is None:
            self.bottleneck_padding = self.bottleneck_kernel // 2

        layer["hermite_bottleneck_conv"] = HermiteConv2D(
            input_features=self.hidden_channels * self.num_rotations,
            output_features=1, # just one channel (num_rotations are not here specified)
            num_rotations=self.num_rotations,
            upsampling=self.upsampling,
            filter_size=self.bottleneck_kernel,
            stride=self.stride,
            padding=self.bottleneck_padding,
            first_layer=False,
        )

        self.add_bottleneck_bn_layer(layer)  # add the bottleneck bn_layer
        self.add_activation(layer)
        self.features.add_module("bottleneck_layer", nn.Sequential(layer))

    def add_subsequent_layers(self):
        if not isinstance(self.hidden_kern, Iterable):
            self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

        for l in range(1, self.num_layers):
            layer = OrderedDict()

            if self.hidden_padding is None:
                self.hidden_padding = self.hidden_kern[l - 1] // 2

            layer["hermite_conv"] = HermiteConv2D(
                input_features=self.hidden_channels * self.num_rotations,
                output_features=self.hidden_channels,
                num_rotations=self.num_rotations,
                upsampling=self.upsampling,
                filter_size=self.hidden_kern[l - 1],
                stride=self.stride,
                padding=self.hidden_padding,
                first_layer=False,
            )
            self.add_bn_layer(layer)
            self.add_activation(layer)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))
        
        # at the end, add also the bottleneck layer
        self.add_bottleneck_layer()
    
    def forward(self, input_):
        # print("FORWARD")
        num_of_layers = len(self.features)
        # print(input_.shape)

        for l, feat in enumerate(self.features):
            input_ = feat(input_)
            # if num_of_layers-1 == l:
                # print("last layer")
            # print(input_.shape)

        # print("------------------")

        return input_

    def initialize(self):
        self.apply(self.init_conv_hermite)

    @staticmethod
    def init_conv_hermite(m):
        if isinstance(m, HermiteConv2D):
            nn.init.normal_(m.coeffs.data, std=0.1)

    # def forward(self, input_):
    #     ret = []
    #     for l, feat in enumerate(self.features):
    #         input_ = feat(input_)
    #         ret.append(input_)

    #     return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].hermite_conv.weights_all_rotations, avg=self.use_avg_reg)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.num_layers):
            ret = (
                ret
                + self.features[l]
                .hermite_conv.weights_all_rotations.pow(2)
                .sum(3, keepdim=True)
                .sum(2, keepdim=True)
                .sqrt()
                .mean()
            )
        return ret / ((self.num_layers - 1) if self.num_layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels * self.num_rotations


class RotationEquivariant2dCoreBottleneckUltrasparse(Stacked2dCore, nn.Module):
    """
    A core built of 2d rotation-equivariant layers. For more info refer to https://openreview.net/forum?id=H1fU8iAqKX.
    """

    def __init__(
        self,
        *args,
        num_rotations=8,
        stride=1,
        upsampling=2,
        rot_eq_batch_norm=True,
        input_regularizer="LaplaceL2norm",
        bottleneck_kernel=3,
        bottleneck_padding=None,
        **kwargs,
    ):
        self.num_rotations = num_rotations
        self.stride = stride
        self.upsampling = upsampling
        self.rot_eq_batch_norm = rot_eq_batch_norm
        self.bottleneck_kernel = bottleneck_kernel
        self.bottleneck_padding = bottleneck_padding
        super().__init__(*args,
            **kwargs, input_regularizer=input_regularizer)

    def set_batchnorm_type(self):
        if not self.rot_eq_batch_norm:
            self.batchnorm_layer_cls = nn.BatchNorm2d
            self.bias_layer_cls = Bias2DLayer
            self.scale_layer_cls = Scale2DLayer
        else:
            self.batchnorm_layer_cls = partial(RotationEquivariantBatchNorm2D, num_rotations=self.num_rotations)
            self.bias_layer_cls = partial(RotationEquivariantBias2DLayer, num_rotations=self.num_rotations)
            self.scale_layer_cls = partial(RotationEquivariantScale2DLayer, num_rotations=self.num_rotations)

    def add_first_layer(self):
        layer = OrderedDict()
        layer["hermite_conv"] = HermiteConv2D(
            input_features=self.input_channels,
            output_features=self.hidden_channels,
            num_rotations=self.num_rotations,
            upsampling=self.upsampling,
            filter_size=self.input_kern,
            stride=self.stride,
            padding=self.input_kern // 2 if self.pad_input else 0,
            first_layer=True,
        )
        self.add_bn_layer(layer)
        self.add_activation(layer)
        self.features.add_module("layer0", nn.Sequential(layer))

    # def add_subsequent_layers(self):
    #     if not isinstance(self.hidden_kern, Iterable):
    #         self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

    #     for l in range(1, self.num_layers):
    #         layer = OrderedDict()

    #         if self.hidden_padding is None:
    #             self.hidden_padding = self.hidden_kern[l - 1] // 2

    #         layer["hermite_conv"] = HermiteConv2D(
    #             input_features=self.hidden_channels * self.num_rotations,
    #             output_features=self.hidden_channels,
    #             num_rotations=self.num_rotations,
    #             upsampling=self.upsampling,
    #             filter_size=self.hidden_kern[l - 1],
    #             stride=self.stride,
    #             padding=self.hidden_padding,
    #             first_layer=False,
    #         )
    #         self.add_bn_layer(layer)
    #         self.add_activation(layer)
    #         self.features.add_module("layer{}".format(l), nn.Sequential(layer))
    
    def add_bottleneck_bn_layer(self, layer):
        """
        Same as add_bn_layer(), but self.hidden_channels are replaced by 1
        """
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = self.batchnorm_layer_cls(1, momentum=self.momentum)
            else:
                layer["norm"] = self.batchnorm_layer_cls(
                    1, momentum=self.momentum, affine=self.bias and self.batch_norm_scale
                )
                if self.bias:
                    if not self.batch_norm_scale:
                        layer["bias"] = self.bias_layer_cls(1)
                elif self.batch_norm_scale:
                    layer["scale"] = self.scale_layer_cls(1)

    def add_bottleneck_layer(self):
        """
        Adds the bottleneck layer
        """

        layer = OrderedDict()

        if self.bottleneck_padding is None:
            self.bottleneck_padding = self.bottleneck_kernel // 2

        layer["hermite_bottleneck_conv"] = HermiteConv2D(
            input_features=self.hidden_channels * self.num_rotations,
            output_features=1, # just one channel (num_rotations are not here specified)
            num_rotations=self.num_rotations,
            upsampling=self.upsampling,
            filter_size=self.bottleneck_kernel,
            stride=self.stride,
            padding=self.bottleneck_padding,
            first_layer=False,
        )

        self.add_bottleneck_bn_layer(layer)  # add the bottleneck bn_layer
        self.add_activation(layer)
        self.features.add_module("bottleneck_layer", nn.Sequential(layer))

    def add_subsequent_layers(self):
        if not isinstance(self.hidden_kern, Iterable):
            self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

        for l in range(1, self.num_layers):
            layer = OrderedDict()

            if self.hidden_padding is None:
                self.hidden_padding = self.hidden_kern[l - 1] // 2

            layer["hermite_conv"] = HermiteConv2D(
                input_features=self.hidden_channels * self.num_rotations,
                output_features=self.hidden_channels,
                num_rotations=self.num_rotations,
                upsampling=self.upsampling,
                filter_size=self.hidden_kern[l - 1],
                stride=self.stride,
                padding=self.hidden_padding,
                first_layer=False,
            )
            self.add_bn_layer(layer)
            self.add_activation(layer)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))
        
        # at the end, add also the bottleneck layer
        self.add_bottleneck_layer()
    
    def forward(self, input_):
        # print("FORWARD")
        num_of_layers = len(self.features)
        # print(input_.shape)

        for l, feat in enumerate(self.features):
            input_ = feat(input_)
            # if num_of_layers-1 == l:
                # print("last layer")
            # print(input_.shape)

        # print("------------------")

        return input_

    def initialize(self):
        self.apply(self.init_conv_hermite)

    @staticmethod
    def init_conv_hermite(m):
        if isinstance(m, HermiteConv2D):
            nn.init.normal_(m.coeffs.data, std=0.1)

    # def forward(self, input_):
    #     ret = []
    #     for l, feat in enumerate(self.features):
    #         input_ = feat(input_)
    #         ret.append(input_)

    #     return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].hermite_conv.weights_all_rotations, avg=self.use_avg_reg)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.num_layers):
            ret = (
                ret
                + self.features[l]
                .hermite_conv.weights_all_rotations.pow(2)
                .sum(3, keepdim=True)
                .sum(2, keepdim=True)
                .sqrt()
                .mean()
            )
        return ret / ((self.num_layers - 1) if self.num_layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels * self.num_rotations

# class RotationEquivariant2dCoreBottleneck(RotationEquivariant2dCore):
#     def __init__(
#         self,
#         *args,
#         num_rotations=8,
#         stride=1,
#         upsampling=2,
#         rot_eq_batch_norm=True,
#         input_regularizer="LaplaceL2norm",
#         bottleneck_kernel=3,
#         bottleneck_padding=None,
#         **kwargs,
#     ):
#         self.num_rotations = num_rotations
#         self.stride = stride
#         self.upsampling = upsampling
#         self.rot_eq_batch_norm = rot_eq_batch_norm
#         self.bottleneck_kernel = bottleneck_kernel
#         self.bottleneck_padding = bottleneck_padding
#         super().__init__(*args,
#             **kwargs, input_regularizer=input_regularizer)
    
#     def add_bottleneck_bn_layer(self, layer):
#         """
#         Same as add_bn_layer(), but self.hidden_channels are replaced by 1
#         """
#         if self.batch_norm:
#             if self.independent_bn_bias:
#                 layer["norm"] = self.batchnorm_layer_cls(1, momentum=self.momentum)
#             else:
#                 layer["norm"] = self.batchnorm_layer_cls(
#                     1, momentum=self.momentum, affine=self.bias and self.batch_norm_scale
#                 )
#                 if self.bias:
#                     if not self.batch_norm_scale:
#                         layer["bias"] = self.bias_layer_cls(1)
#                 elif self.batch_norm_scale:
#                     layer["scale"] = self.scale_layer_cls(1)

#     def add_bottleneck_layer(self):
#         """
#         Adds the bottleneck layer
#         """

#         layer = OrderedDict()

#         if self.bottleneck_padding is None:
#             self.bottleneck_padding = self.bottleneck_kernel // 2

#         layer["hermite_bottleneck_conv"] = HermiteConv2D(
#             input_features=self.hidden_channels * self.num_rotations,
#             output_features=1, # just one channel (num_rotations are not here specified)
#             num_rotations=self.num_rotations,
#             upsampling=self.upsampling,
#             filter_size=self.bottleneck_kernel,
#             stride=self.stride,
#             padding=self.bottleneck_padding,
#             first_layer=False,
#         )

#         self.add_bottleneck_bn_layer(layer)  # add the bottleneck bn_layer
#         self.add_activation(layer)
#         self.features.add_module("bottleneck_layer", nn.Sequential(layer))

#     def add_subsequent_layers(self):
#         if not isinstance(self.hidden_kern, Iterable):
#             self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

#         for l in range(1, self.num_layers):
#             layer = OrderedDict()

#             if self.hidden_padding is None:
#                 self.hidden_padding = self.hidden_kern[l - 1] // 2

#             layer["hermite_conv"] = HermiteConv2D(
#                 input_features=self.hidden_channels * self.num_rotations,
#                 output_features=self.hidden_channels,
#                 num_rotations=self.num_rotations,
#                 upsampling=self.upsampling,
#                 filter_size=self.hidden_kern[l - 1],
#                 stride=self.stride,
#                 padding=self.hidden_padding,
#                 first_layer=False,
#             )
#             self.add_bn_layer(layer)
#             self.add_activation(layer)
#             self.features.add_module("layer{}".format(l), nn.Sequential(layer))
        
#         # at the end, add also the bottleneck layer
#         self.add_bottleneck_layer()
    
#     def forward(self, input_):
#         # print("FORWARD")
#         num_of_layers = len(self.features)
#         # print(input_.shape)

#         for l, feat in enumerate(self.features):
#             input_ = feat(input_)
#             # if num_of_layers-1 == l:
#                 # print("last layer")
#             # print(input_.shape)

#         # print("------------------")

#         return input_


