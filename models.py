from predict_neural_responses.models import *
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
import predict_neural_responses.dnn_blocks.dnn_blocks as bl

class Picek(encoding_model):
    """My model with FullFactorized2d readout"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self.loss = PoissonLoss(avg=False)
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

        self.nonlin = bl.act_func()[config["nonlinearity"]]

    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x

class PicekGauss(encoding_model):
    """My model with Gaussian readout"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self.loss = PoissonLoss(avg=False)
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

        self.nonlin = bl.act_func()[config["nonlinearity"]]

    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x


class HelloWorld(encoding_model):
    def __init__(self, **config):
        super().__init__(**config)
        self.config = config

        self.core = cores.Stacked2dCore(
            input_channels=self.config["input_channels"],
            hidden_channels=self.config["core_hidden_channels"],
            input_kern=self.config["core_input_kern"],
            hidden_kern=self.config["core_hidden_kern"],
            layers=self.config["core_layers"],
            input_regularizer="LaplaceL2norm",
            gamma_input=config["core_gamma_input"],
            gamma_hidden=config["core_gamma_hidden"],
            stack=-1,
            depth_separable=False,
        )

        self.readout = readouts.FullFactorized2d(
            in_shape=(
                self.config["core_hidden_channels"],
                self.config["input_size_x"], # ocividne se to padduje, takze to neztraci rozmery
                self.config["input_size_y"],
            ),
            outdims=self.config["num_neurons"],
            bias=self.config["readout_bias"],
            mean_activity=self.config["mean_activity"],
            spatial_and_feature_reg_weight=self.config["readout_gamma"],
        )

        self.register_buffer("laplace", torch.from_numpy(laplace()))
        self.nonlin = bl.act_func()["softplus"]


    def forward(self, x):
        x = self.core(x)
        x = self.nonlin(self.readout(x))
        return x

