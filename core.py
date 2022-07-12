from predict_neural_responses.models import *
from collections import OrderedDict, Iterable
from functools import partial
from torch import nn
from neuralpredictors.layers.cores.conv2d import Stacked2dCore
from neuralpredictors.layers.affine import Bias2DLayer, Scale2DLayer
from neuralpredictors.layers.activations import AdaptiveELU
from neuralpredictors.layers.hermite import (
    HermiteConv2D,
    RotationEquivariantBatchNorm2D,
    RotationEquivariantBias2DLayer,
    RotationEquivariantScale2DLayer,
)


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
        bottleneck_kernel=3,  # added
        bottleneck_padding=None,  # added
        **kwargs,
    ):
        self.num_rotations = num_rotations
        self.stride = stride
        self.upsampling = upsampling
        self.rot_eq_batch_norm = rot_eq_batch_norm
        self.bottleneck_kernel = bottleneck_kernel  # added
        self.bottleneck_padding = bottleneck_padding  # added
        super().__init__(*args, **kwargs, input_regularizer=input_regularizer)

    def set_batchnorm_type(self):
        if not self.rot_eq_batch_norm:
            self.batchnorm_layer_cls = nn.BatchNorm2d
            self.bias_layer_cls = Bias2DLayer
            self.scale_layer_cls = Scale2DLayer
        else:
            self.batchnorm_layer_cls = partial(
                RotationEquivariantBatchNorm2D, num_rotations=self.num_rotations
            )
            self.bias_layer_cls = partial(
                RotationEquivariantBias2DLayer, num_rotations=self.num_rotations
            )
            self.scale_layer_cls = partial(
                RotationEquivariantScale2DLayer, num_rotations=self.num_rotations
            )

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

    def add_bottleneck_bn_layer(self, layer):  # added
        """
        Same as add_bn_layer(), but self.hidden_channels are replaced by 1
        """
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = self.batchnorm_layer_cls(1, momentum=self.momentum)
            else:
                layer["norm"] = self.batchnorm_layer_cls(
                    1,
                    momentum=self.momentum,
                    affine=self.bias and self.batch_norm_scale,
                )
                if self.bias:
                    if not self.batch_norm_scale:
                        layer["bias"] = self.bias_layer_cls(1)
                elif self.batch_norm_scale:
                    layer["scale"] = self.scale_layer_cls(1)

    def add_bottleneck_layer(self):  # added
        """
        Adds the bottleneck layer (into self.features)
        """

        layer = OrderedDict()

        if self.bottleneck_padding is None:
            self.bottleneck_padding = self.bottleneck_kernel // 2

        layer["hermite_bottleneck_conv"] = HermiteConv2D(
            input_features=self.hidden_channels * self.num_rotations,
            output_features=1,  # just one channel (num_rotations are not here specified) .. THE MAIN CHANGE (it is all what is needed)
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

    def add_activation(self, layer):
        """
        Overwritten only one line. Original:
            if len(self.features) < self.num_layers - 1 or self.final_nonlinearity:
        Now:
            if len(self.features) < self.num_layers or self.final_nonlinearity:
        """

        if self.linear:
            return
        if len(self.features) < self.num_layers or self.final_nonlinearity:
            layer["nonlin"] = AdaptiveELU(self.elu_xshift, self.elu_yshift)

    def add_subsequent_layers(self):  # edited
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
        self.add_bottleneck_layer()  # added

    def forward(self, input_):
        num_of_layers = len(self.features)

        for l, feat in enumerate(self.features):
            # goes also through the bottleneck (it is added into the features)
            input_ = feat(input_)

        return input_

    def initialize(self):
        self.apply(self.init_conv_hermite)

    @staticmethod
    def init_conv_hermite(m):
        if isinstance(m, HermiteConv2D):
            nn.init.normal_(m.coeffs.data, std=0.1)

    def laplace(self):
        return self._input_weights_regularizer(
            self.features[0].hermite_conv.weights_all_rotations, avg=self.use_avg_reg
        )

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
        return (
            self.group_sparsity() * self.gamma_hidden
            + self.gamma_input * self.laplace()
        )

    @property
    def outchannels(self):  # edited .. returns just 1 channel for each rotation
        return self.num_rotations
