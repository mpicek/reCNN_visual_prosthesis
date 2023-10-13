import numpy as np
import torch
from neuralpredictors.measures.modules import Corr, PoissonLoss
from utils import get_fraction_oracles
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from neuralpredictors.measures.np_functions import corr as corr_from_neuralpredictors
from neuralpredictors.measures.np_functions import (
    oracle_corr_jackknife,
    oracle_corr_conservative,
)

import predict_neural_responses.dnn_blocks.dnn_blocks as bl

"""
An energy model consists of 2 gabor filters shifted by 90 degrees in phase.
In a regular energy model, each neuron would have own sigma_x, sigma_y, f,..., but
in our case, we want sigma_x, sigma_y and f to be shared (because we know only the orientation
and position of each neuron, nothing more).
A gabor filter has the following parameters (they are shared, except for phase):
    positions_x (np.array): defined by ground truth positions 
    positions_y (np.array): defined by ground truth positions 
    sigma_x (float): shared std of the gaussian filter in x direction
    sigma_y (float): shared std of the gaussian filter in y direction
    f (float): shared frequency of the cosine filter
    theta (np.array): defined by ground truth orientations
    phase (float): 0 and 90 (or 0 and np.pi/2), depending whether the filter is odd or even
    xlim, ylim (arrays): defined by the size of the image
    resolution (tuple): defined by the size of the image
"""

## Correct rotation matrix (clockwise), no learnable shift (not needed)


class EnergyModel(pl.LightningModule):
    def __init__(
        self,
        train_positions_x,
        train_positions_y,
        train_orientations,
        val_pos_x,
        val_pos_y,
        val_ori,
        test_pos_x,
        test_pos_y,
        test_ori,
        resolution,
        xlim,
        ylim,
        lr,
        default_ori_shift=0,
        counter_clockwise_rotation=True,
        multivariate=True,
        scale_init=None,
        f_init=None,
        bias_init=None,
        exact_init=False,
        sigma_x_init=None,
        sigma_y_init=None,
        nonlinearity=None,
        **config,
    ):
        """Constructor.

        Args:
            positions_x (np.array): Array of x positions in degrees of vis. field (shape [num_neurons,])
            positions_y (np.array): Array of y positions in degrees of vis. field (shape [num_neurons,])
            orientations (np.array): Array of preferred orientations in radians (shape [num_neurons,])
            resolution (tuple): resolution of the input image
            xlim (array): array of 2 elements - how much the visual stimulus spans to the right
                and to the left. If the stimulus spans 11 degrees of visual field, xlim is [-5.5, 5.5]
            ylim (array): array of 2 elements - how much the visual stimulus spans on the y axis (up and down).
                If the stimulus spans 11 degrees of visual field, xlim is [-5.5, 5.5]
            lr (float): learning rate
            default_ori_shift (int, optional): How much to shift the orientations (IN DEGREES!!). Defaults to 0.
            counter_clockwise_rotation (bool, optional): Whether the orientation is in clockwise
                or counterclockwise rotation. Defaults to True.
            multivariate (bool, optional): Whether we want to use multivariate Gaussian
                distribution (separate std for both x an y dimensions). Defaults to True.
        """
        super().__init__()

        self.minimum_sigma = 0.01 # minimum sigma value
        self.response_clamp_minimum = 0.00001 # minimum response value (cannot be negative)
        min_init_sigma_value = 0.01
        min_init_f_value = 0.00001

        self.test_average_batch = config["test_average_batch"]
        self.compute_oracle_fraction = config["compute_oracle_fraction"]
        self.conservative_oracle = config["conservative_oracle"]
        self.jackknife_oracle = config["jackknife_oracle"]
        self.generate_oracle_figure = config["generate_oracle_figure"]

        self.lr = lr
        self.loss = PoissonLoss()
        self.corr = Corr()
        self.num_neurons = config["num_neurons"]  # number of neurons
        self.val_neurons = config["val_neurons"]
        self.test_neurons = config["test_neurons"]
        self.train_neurons = self.num_neurons - self.val_neurons - self.test_neurons
        self.multivariate = multivariate
        self.resolution = resolution

        self.cor = torch.nn.Parameter(torch.ones(1) * 1.0)

        if nonlinearity is None:
            self.nonlin = None
        else:
            self.nonlin = bl.act_func()[nonlinearity]

        # initializing the parameters
        if exact_init:
            if scale_init is not None:
                self.scale = torch.nn.Parameter(torch.ones(1) * scale_init)
            else:
                self.scale = torch.nn.Parameter(torch.rand(1))

            if bias_init is not None:
                self.bias = torch.nn.Parameter(torch.ones(1) * bias_init)
            else:
                self.bias = torch.nn.Parameter(torch.rand(1))
            
            if sigma_x_init is not None and sigma_y_init is not None:
                self.sigma_x = torch.nn.Parameter(torch.ones(1) * sigma_x_init)
                self.sigma_y = torch.nn.Parameter(torch.ones(1) * sigma_y_init)
            else:
                self.sigma_x = torch.nn.Parameter(torch.rand(1) + min_init_sigma_value)
                self.sigma_y = torch.nn.Parameter(torch.rand(1) + min_init_sigma_value)
            
            if f_init is not None:
                self.f = torch.nn.Parameter(torch.ones(1) * f_init)
            else:
                self.f = torch.nn.Parameter(torch.rand(1) + min_init_f_value)
            
        else:
            if scale_init is not None:
                self.scale = torch.nn.Parameter(torch.FloatTensor(1).uniform_(scale_init - 0.05, scale_init + 0.05))
            else:
                self.scale = torch.nn.Parameter(torch.rand(1))

            if bias_init is not None:
                self.bias = torch.nn.Parameter(torch.FloatTensor(1).uniform_(bias_init - 0.05, bias_init + 0.05))
            else:
                self.bias = torch.nn.Parameter(torch.rand(1))
            
            if sigma_x_init is not None and sigma_y_init is not None:
                self.sigma_x = torch.nn.Parameter(torch.FloatTensor(1).uniform_(sigma_x_init - 0.05, sigma_x_init + 0.05))
                self.sigma_y = torch.nn.Parameter(torch.FloatTensor(1).uniform_(sigma_y_init - 0.05, sigma_y_init + 0.05))
            else:
                self.sigma_x = torch.nn.Parameter(torch.rand(1) + min_init_sigma_value)
                self.sigma_y = torch.nn.Parameter(torch.rand(1) + min_init_sigma_value)
            
            if f_init is not None:
                self.f = torch.nn.Parameter(torch.FloatTensor(1).uniform_(f_init - 0.05, f_init + 0.05))
            else:
                self.f = torch.nn.Parameter(torch.rand(1) + min_init_f_value)
            

        # convert the default orientation shift to radians
        self.default_ori_shift = (default_ori_shift / 180) * np.pi

        self.counter_clockwise_rotation = counter_clockwise_rotation

        self.register_buffer(
            "positions_x", torch.from_numpy(positions_x.astype("float32"))
        )
        self.register_buffer(
            "positions_y", torch.from_numpy(positions_y.astype("float32"))
        )
        self.register_buffer(
            "orientations", torch.from_numpy(orientations.astype("float32"))
        )

        self.register_buffer(
            "val_pos_x", torch.from_numpy(val_pos_x.astype("float32"))
        )
        self.register_buffer(
            "val_pos_y", torch.from_numpy(val_pos_y.astype("float32"))
        )
        self.register_buffer(
            "val_ori", torch.from_numpy(val_ori.astype("float32"))
        )

        self.register_buffer(
            "positions_x", torch.from_numpy(test_pos_x.astype("float32"))
        )
        self.register_buffer(
            "positions_y", torch.from_numpy(test_pos_y.astype("float32"))
        )
        self.register_buffer(
            "test_ori", torch.from_numpy(test_ori.astype("float32"))
        )

        self.init_gabor_filters(resolution, xlim, ylim)

        self.save_hyperparameters()

    def init_gabor_filters(self, resolution, xlim, ylim):
        """Creates a tensor (not trainable) of meshgrids of gabor filters
           with proper rotations.
           It registers it in the buffer (so that it can be moved to GPU easily).
           Gabors are of two types: even and odd -> therefore there are two types of all variables

           The tensors created are of shape [num_of_neurons, meshgrid_dimension]
             An example tensor (one of four created):
             - gabors_odd_meshgrid_X -> tensor of odd gabor filters with meshgrid for X axis

        Args:
            resolution (tuple): resolution of the input image
            xlim (array): array of 2 elements - how much the visual stimulus spans to the right
                and to the left. If the stimulus spans 11 degrees of visual field, xlim is [-5.5, 5.5]
            ylim (array): array of 2 elements - how much the visual stimulus spans on the y axis (up and down).
                If the stimulus spans 11 degrees of visual field, xlim is [-5.5, 5.5]
        """

        # num_neurons = None
        # if dataset_type == "train":
        #     num_neurons = self.train_neurons
        # elif dataset_type == "val":
        #     num_neurons = self.val_neurons
        # else:
        #     num_neurons = self.test_neurons

        # we create the sampling meshgrid
        x = np.linspace(xlim[0], xlim[1], resolution[0])
        y = np.linspace(ylim[0], ylim[1], resolution[1])
        X, Y = np.meshgrid(x, y)

        # connecting two tuples to get shape: (self.num_neurons, X.shape[0], X.shape[1]) ... and Y respectively
        # we create a grid for every neuron (therefore this shape)
        meshgrid_shape_x = (self.num_neurons,) + X.shape
        meshgrid_shape_y = (self.num_neurons,) + Y.shape

        # we create a meshgrid FOR EACH NEURON SEPARATELY, therefore the
        # broadcasted_x or _y are of shape (num_neurons, x_pixels, y_pixels)
        broadcasted_x = np.broadcast_to(X, meshgrid_shape_x)
        broadcasted_y = np.broadcast_to(Y, meshgrid_shape_y)

        # create a tensor from the broadcasted meshgrids
        meshgrid_x = torch.from_numpy(broadcasted_x.astype("float32"))
        meshgrid_y = torch.from_numpy(broadcasted_y.astype("float32"))

        # we shift the orientations by the default shift
        shifted_orientation = torch.remainder(
            self.orientations + self.default_ori_shift, np.pi ##################################################################################### 2*np.pi
        )

        meshgrid_x_rotated = None
        meshgrid_y_rotated = None

        self.original_positions_x = self.positions_x
        self.original_positions_y = self.positions_y


        # we rotate the meshgrids according to the shifted orientations
        if self.counter_clockwise_rotation:
            meshgrid_x_rotated = meshgrid_x * torch.cos(
                shifted_orientation[:, None, None]
            ) - meshgrid_y * torch.sin(shifted_orientation[:, None, None])
            meshgrid_y_rotated = meshgrid_x * torch.sin(
                shifted_orientation[:, None, None]
            ) + meshgrid_y * torch.cos(shifted_orientation[:, None, None])

            tmp_x = np.cos(shifted_orientation) * self.positions_x - np.sin(shifted_orientation) * self.positions_y
            tmp_y = np.sin(shifted_orientation) * self.positions_x + np.cos(shifted_orientation) * self.positions_y
            self.positions_x = tmp_x
            self.positions_y = tmp_y
        else:
            meshgrid_x_rotated = meshgrid_x * torch.cos(
                shifted_orientation[:, None, None]
            ) + meshgrid_y * torch.sin(shifted_orientation[:, None, None])
            meshgrid_y_rotated = -meshgrid_x * torch.sin(
                shifted_orientation[:, None, None]
            ) + meshgrid_y * torch.cos(shifted_orientation[:, None, None])

            tmp_x = np.cos(shifted_orientation) * self.positions_x + np.sin(shifted_orientation) * self.positions_y
            tmp_y = -np.sin(shifted_orientation) * self.positions_x + np.cos(shifted_orientation) * self.positions_y
            self.positions_x = tmp_x
            self.positions_y = tmp_y

            
        # and we register the meshgrids in pytorch
        self.register_buffer("meshgrid_x_rotated", meshgrid_x_rotated)
        self.register_buffer("meshgrid_y_rotated", meshgrid_y_rotated)
    

    def forward(self, x):

        # because sometimes the input is analogous to RGB, which has 3 channels,
        # but in our case (grayscale), we have 1 channel and with this line we delete
        # the redundant dimension of size 1
        if x.shape[1] == 1:
            x = torch.squeeze(x, 1)

        # The Gabor Filter consists of two filters:
        #   the gaussian filter and the cosine (or equivalently sine) filter
        # We firstly define the gaussian filter - each position in the sampling grid
        # is firstly moved to the neuron's position and at each sampling point
        # the gaussian is applied
        gaussian_filter = None
        if self.multivariate: # multivariate => sigma for both x and y dimensions
            
            # THE VALUE PASSED INTO EXP MUST NOT BE TOO LARGE, OTHERWISE THE EXP WILL EXPLODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            gaussian_filter = torch.exp(
                -0.5
                * (
                    # copy positions (vector) to each position. It is aligned with the 0th axis... [i, :, :] are same numbers (copied)
                    (
                        torch.square(
                            self.meshgrid_x_rotated - self.positions_x[:, None, None]
                        )
                        / (torch.square(self.sigma_x) + self.minimum_sigma) # + minimum_sigma as we don't want to divide by 0
                    )
                    + (
                        torch.square(
                            self.meshgrid_y_rotated - self.positions_y[:, None, None]
                        )
                        / (torch.square(self.sigma_y) + self.minimum_sigma) # + minimum_sigma as we don't want to divide by 0
                    )
                )
            )
        else: # if not multivariate => only one sigma for both directions
            gaussian_filter = torch.exp(
                -0.5
                * (
                    # copy positions (vector) to each position. It is aligned with the 0th axis... [i, :, :] are same numbers (copied)
                    (
                        torch.square(
                            self.meshgrid_x_rotated - self.positions_x[:, None, None]
                        )
                        / (torch.square(self.sigma_x) + self.minimum_sigma) # + minimum_sigma as we don't want to divide by 0
                    )
                    + (
                        torch.square(
                            self.meshgrid_y_rotated - self.positions_y[:, None, None]
                        )
                        / (torch.square(self.sigma_x) + self.minimum_sigma) # + minimum_sigma as we don't want to divide by 0
                    )
                )
            )

        # The cosine filter also takes into account the position of the neuron (only the x coordinate)
        # and samples the cosine function at each point (with a learned frequency f).
        # As Energy Model has two gabor filters - odd and even gabor filter, we create both (they
        # differ only in 90deg shifted phase (90deg = pi/2))
        cosine_filter = torch.cos(
            2
            * np.pi
            * (self.meshgrid_x_rotated + self.positions_x[:, None, None])
            * self.f
            + 0
        )  # phase = 0
        self.odd_gabor_filter = gaussian_filter * cosine_filter
        cosine_filter = torch.cos(
            2
            * np.pi
            * (self.meshgrid_x_rotated + self.positions_x[:, None, None])
            * self.f
            + (np.pi / 2)
        )  # phase = 90deg = np.pi/2
        self.even_gabor_filter = (
            gaussian_filter * cosine_filter
        )  # shape [num_neurons, x_pixels, y_pixels]

        # Now we apply the gabor filters on the image
        # because x is [batch_size, x_pixels, y_pixels] and gabor_filter is [num_neurons, x_pixels, y_pixels],
        # we do the tensordot in the corresponding dimensions matching x_pixels and y_pixels
        # it returns a tensor of shape [batch_size, num_neurons].. so there is a response of each filter
        # for each image in the batch
        # x += self.bias
        filtered_image_odd = torch.tensordot(
            x, self.odd_gabor_filter, dims=[[1, 2], [1, 2]]
        )
        filtered_image_even = torch.tensordot(
            x, self.even_gabor_filter, dims=[[1, 2], [1, 2]]
        )

        # response of each neuron to each image in a batch .. therefore shape of [batch_size, num_neurons]
        # we scale the response by self.scale and add bias self.bias
        
        energy_model_response = None
        
        # if self.nonlin is None:
        # energy_model_response = self.scale * torch.sqrt(
        #     torch.square(filtered_image_odd) + torch.square(filtered_image_even)
        # )# + self.bias

        energy_model_response = self.bias + self.scale * torch.sqrt(torch.square(filtered_image_odd) + torch.square(filtered_image_even))

        # energy_model_response = self.final_nonlinearity(self.odd_nonlinearity(torch.square(filtered_image_odd)) + self.even_nonlinearity(torch.square(filtered_image_even)))

        # else:
        #     energy_model_response = self.scale * self.nonlin(
        #         torch.square(filtered_image_odd) + torch.square(filtered_image_even)
        #     ) + self.bias

        # as the response has to be non-negative, we clamp the values
        # print("uz neklampuje _____")
        energy_model_response_clamped = torch.clamp(energy_model_response, min=self.response_clamp_minimum)
        # return energy_model_response
        # print(energy_model_response_clamped[0][0])

        return energy_model_response_clamped
    
    def configure_optimizers(self):
        """Configures the optimizer for the training of the model (Adam).

        Returns:
            torch.optimizer: torch optimizer class
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def training_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        self.log("train/loss", loss)
        self.log("val/loss", loss)
        self.log("val/sigma_x", self.sigma_x)
        self.log("val/sigma_y", self.sigma_y)
        self.log("val/f", self.f)
        self.log("val/scale", self.scale)
        self.log("val/bias", self.bias)
        return loss

    def validation_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        # print("pred")
        # print(prediction[0,0])
        # print("resp")
        # print(resp[0,0])
        loss = self.loss(prediction, resp)
        # print("loss")
        # print(loss)
        self.log("val/loss", loss)
        self.log("val/sigma_x", self.sigma_x)
        self.log("val/sigma_y", self.sigma_y)
        self.log("val/f", self.f)
        self.log("val/scale", self.scale)
        self.log("val/bias", self.bias)
        # print("----------------")
        # print(resp.mean())
        # print(resp.std())
        # print(resp.max())
        # print(resp.min())
        # print(prediction.mean())
        # print(prediction.std())
        # print(prediction.max())
        # print(prediction.min())

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
        self.predictions = predictions
        self.responses = responses
        print(np.mean(correlation))
        print(np.mean(correlation))
        self.log("val/corr", np.mean(correlation))

    def visualize(
        self,
        neuron_id,
        odd=True,
        init=True,
        title="",
        vmin=None,
        vmax=None,
        return_plt=False,
        ticks=True,
        cmap=None,
        graph_size=10,
    ):
        """Visualizes the neuron neuron_id.

        Args:
            neuron_id (int or str): Index of the neuron (within the filtered neurons so should be in [0, num_filtered))
                If neuron_id is "all", the visual field of all neurons is plotted
            odd (bool, optional): Show odd or even gabor filter. Defaults to True.
            title (str, optional): Title of the graph. If nothing provided, automatic title generated. Defaults to "".
            vmin (_type_, optional): _description_. Defaults to None.
            vmax (_type_, optional): _description_. Defaults to None.
            return_plt (bool, optional): Return plot or not. If not, it is plotted right away. Defaults to False.
            ticks (bool, optional): Use ticks or not. Defaults to True.
            cmap (str, optional): Which colormap to use. If None, cm.coolwarm is used. Defaults to None.
            graph_size (int, optional): Size of the graph. Defaults to 10.

        Returns:
            matplotlib.plot: The plot
        """

        # we need to have the whole network initialized, but some parts get initialized
        # only during the forward method execution. Therefore we have to create an artificial
        # input into the network and pass it through the network to initialize it (if it
        # is not initialized, otherwise this is redundant)
        if init:
            tmp_input = torch.rand((1, 1) + self.resolution)
            self.forward(tmp_input)

        if odd:
            gabor = self.odd_gabor_filter.cpu().detach().numpy().squeeze()
        else:
            gabor = self.even_gabor_filter.cpu().detach().numpy().squeeze()

        if neuron_id == "all":
            tmp = np.max(np.abs(gabor), axis=0)
            maximum = tmp.max()

            sorted = np.sort(tmp.flatten())
            plt.clf()
            plt.plot(sorted)
            plt.title("distribution of maximal values of positions in gabor filters")
            plt.show()
            # ten_percent = sorted[sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[2*sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[3*sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[4*sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[5*sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[sorted.shape[0]//10]

            # print(ten_percent)
            gabor = (tmp > (maximum/1000)) * 1
            

        else:
            gabor = gabor[neuron_id, :, :]
            gabor *= self.scale.cpu().detach().numpy()
            gabor += self.bias.cpu().detach().numpy()
        
        plt.clf()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gcf().set_size_inches(graph_size, graph_size)
        m = np.max(np.abs(gabor))
        if vmin is None:
            min = -m
        else:
            min = vmin
        if vmax is None:
            max = m
        else:
            max = vmax
        if cmap == "greys":
            color_map = "Greys_r"
        else:
            color_map = cm.coolwarm
        ahoj = plt.imshow(gabor, vmax=max, vmin=min, cmap=color_map)

        if title == "" and neuron_id != "all":
            shifted_orientation = np.mod(
                self.orientations + self.default_ori_shift, np.pi
            )
            title = f"Neuron {neuron_id} \n with position [{self.original_positions_x[neuron_id]:.2f}, {self.original_positions_y[neuron_id]:.2f}] \n and pref. orientation {self.orientations[neuron_id]:.2f}rad, (with default shift:{shifted_orientation[neuron_id]:.2f}rad)"
        elif title == "" and neuron_id == "all":
            title = "Visual field of all neurons\n(where the position influences the response\nby at least 1/1000 of maximal influence of the filter)"

        plt.title(title)
        plt.colorbar()
        if ticks == False:
            plt.xticks([])
            plt.yticks([])
        if return_plt:
            return plt
        else:
            plt.show()

    def __str__(self):
        return "EnergyModel"

class EnergyModelIndividual(pl.LightningModule):
    def __init__(
        self,
        positions_x,
        positions_y,
        orientations,
        resolution,
        xlim,
        ylim,
        lr,
        default_ori_shift=0,
        counter_clockwise_rotation=True,
        multivariate=True,
        scale_init=None,
        sigma_init=None,
        f_init=None,
        bias_init=None,
        **config,
    ):
        """Constructor.

        Args:
            positions_x (np.array): Array of x positions in degrees of vis. field (shape [num_neurons,])
            positions_y (np.array): Array of y positions in degrees of vis. field (shape [num_neurons,])
            orientations (np.array): Array of preferred orientations in radians (shape [num_neurons,])
            resolution (tuple): resolution of the input image
            xlim (array): array of 2 elements - how much the visual stimulus spans to the right
                and to the left. If the stimulus spans 11 degrees of visual field, xlim is [-5.5, 5.5]
            ylim (array): array of 2 elements - how much the visual stimulus spans on the y axis (up and down).
                If the stimulus spans 11 degrees of visual field, xlim is [-5.5, 5.5]
            lr (float): learning rate
            default_ori_shift (int, optional): How much to shift the orientations (IN DEGREES!!). Defaults to 0.
            counter_clockwise_rotation (bool, optional): Whether the orientation is in clockwise
                or counterclockwise rotation. Defaults to True.
            multivariate (bool, optional): Whether we want to use multivariate Gaussian
                distribution (separate std for both x an y dimensions). Defaults to True.
        """
        super().__init__()

        self.minimum_sigma = 0.01
        self.response_clamp_minimum = 0.00001
        self.f_clamp = 0.01
        min_init_sigma_value = 0.01
        min_init_f_value = 0.00001

        self.test_average_batch = config["test_average_batch"]
        self.compute_oracle_fraction = config["compute_oracle_fraction"]
        self.conservative_oracle = config["conservative_oracle"]
        self.jackknife_oracle = config["jackknife_oracle"]
        self.generate_oracle_figure = config["generate_oracle_figure"]

        self.lr = lr
        self.loss = PoissonLoss()
        # self.loss = torch.nn.MSELoss()
        self.corr = Corr()
        self.num_neurons = len(orientations)  # number of neurons
        self.multivariate = multivariate
        self.resolution = resolution

        # self.scale = torch.nn.Parameter(torch.rand(1))
        if scale_init is not None:
            self.scale = torch.nn.Parameter(torch.FloatTensor(self.num_neurons).uniform_(scale_init - 0.05, scale_init + 0.05))
        else:
            self.scale = torch.nn.Parameter(torch.rand(self.num_neurons))

        if bias_init is not None:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.num_neurons).uniform_(bias_init - 0.05, bias_init + 0.05))
        else:
            self.bias = torch.nn.Parameter(torch.rand(self.num_neurons))
        
        if sigma_init is not None:
            self.sigma_x = torch.nn.Parameter(torch.FloatTensor(self.num_neurons).uniform_(sigma_init - 0.05, sigma_init + 0.05))
            self.sigma_y = torch.nn.Parameter(torch.FloatTensor(self.num_neurons).uniform_(sigma_init - 0.05, sigma_init + 0.05))
        else:
            self.sigma_x = torch.nn.Parameter(torch.rand(self.num_neurons) + min_init_sigma_value)
            self.sigma_y = torch.nn.Parameter(torch.rand(self.num_neurons) + min_init_sigma_value)
        
        if f_init is not None:
            self.f = torch.nn.Parameter(torch.FloatTensor(self.num_neurons).uniform_(f_init - 0.05, f_init + 0.05))
        else:
            self.f = torch.nn.Parameter(torch.rand(self.num_neurons) + min_init_f_value)

        # convert the default orientation shift to radians
        self.default_ori_shift = (default_ori_shift / 180) * np.pi

        self.counter_clockwise_rotation = counter_clockwise_rotation

        self.register_buffer(
            "positions_x", torch.from_numpy(positions_x.astype("float32"))
        )
        self.register_buffer(
            "positions_y", torch.from_numpy(positions_y.astype("float32"))
        )
        self.register_buffer(
            "orientations", torch.from_numpy(orientations.astype("float32"))
        )

        self.init_gabor_filters(resolution, xlim, ylim)

        self.save_hyperparameters()

    def init_gabor_filters(self, resolution, xlim, ylim):
        """Creates a tensor (not trainable) of meshgrids of gabor filters
           with proper rotations.
           It registers it in the buffer (so that it can be moved to GPU easily).
           Gabors are of two types: even and odd -> therefore there are two types of all variables

           The tensors created are of shape [num_of_neurons, meshgrid_dimension]
             An example tensor (one of four created):
             - gabors_odd_meshgrid_X -> tensor of odd gabor filters with meshgrid for X axis

        Args:
            resolution (tuple): resolution of the input image
            xlim (array): array of 2 elements - how much the visual stimulus spans to the right
                and to the left. If the stimulus spans 11 degrees of visual field, xlim is [-5.5, 5.5]
            ylim (array): array of 2 elements - how much the visual stimulus spans on the y axis (up and down).
                If the stimulus spans 11 degrees of visual field, xlim is [-5.5, 5.5]
        """

        # we create the sampling meshgrid
        x = np.linspace(xlim[0], xlim[1], resolution[0])
        y = np.linspace(ylim[0], ylim[1], resolution[1])
        X, Y = np.meshgrid(x, y)

        # connecting two tuples to get shape: (self.num_neurons, X.shape[0], X.shape[1])
        meshgrid_shape_x = (self.num_neurons,) + X.shape
        meshgrid_shape_y = (self.num_neurons,) + Y.shape

        # we create a meshgrid FOR EACH NEURON SEPARATELY, therefore the
        # broadcasted_x or _y are of shape [num_neurons, x_pixels, y_pixels]
        broadcasted_x = np.broadcast_to(X, meshgrid_shape_x)
        broadcasted_y = np.broadcast_to(Y, meshgrid_shape_y)

        # create a tensor from the broadcasted meshgrids
        meshgrid_x = torch.from_numpy(broadcasted_x.astype("float32"))
        meshgrid_y = torch.from_numpy(broadcasted_y.astype("float32"))

        # we shift the orientations
        shifted_orientation = torch.remainder(
            self.orientations + self.default_ori_shift, np.pi
        )

        meshgrid_x_rotated = None
        meshgrid_y_rotated = None

        # we rotate the meshgrids according to the shifted orientations
        if self.counter_clockwise_rotation:
            meshgrid_x_rotated = meshgrid_x * torch.cos(
                shifted_orientation[:, None, None]
            ) - meshgrid_y * torch.sin(shifted_orientation[:, None, None])
            meshgrid_y_rotated = meshgrid_x * torch.sin(
                shifted_orientation[:, None, None]
            ) + meshgrid_y * torch.cos(shifted_orientation[:, None, None])
        else:
            meshgrid_x_rotated = meshgrid_x * torch.cos(
                shifted_orientation[:, None, None]
            ) + meshgrid_y * torch.sin(shifted_orientation[:, None, None])
            meshgrid_y_rotated = -meshgrid_x * torch.sin(
                shifted_orientation[:, None, None]
            ) + meshgrid_y * torch.cos(shifted_orientation[:, None, None])

        # and we register the meshgrids in pytorch
        self.register_buffer("meshgrid_x_rotated", meshgrid_x_rotated)
        self.register_buffer("meshgrid_y_rotated", meshgrid_y_rotated)

    # def on_after_backward(self):
    #     print("on_after_backward----------------------------")
    #     print("on_after_backward----------------------------")
    #     print("on_after_backward----------------------------")
    #     print("on_after_backward----------------------------")
    #     # global_step = self.global_step
    #     for name, param in self.named_parameters():
    #         if param.requires_grad:
    #             print(name)
    #             print(torch.isnan(param.grad).any())
    #             print(torch.isinf(param.grad).any())
    #             print(param.grad.max())

    def forward(self, x):

        # because sometimes the input is analogous to RGB, which has 3 channels,
        # but in our case (grayscale), we have 1 channel and with code this we delete
        # the redundant dimension of size 1
        if x.shape[1] == 1:
            x = torch.squeeze(x, 1)

        # The Gabor Filter consists of two filters:
        #   the gaussian filter and the cosinus (or equivalently sinus) filter
        # We firstly define the gaussian filter - each position in the sampling grid
        # is firstly moved to the neuron's position and at each sampling point
        # the gaussian is applied
        # assert ((torch.square(self.sigma_x[:, None, None]) + self.clamp_minimum) >= 0).all()
        assert torch.isnan(self.sigma_x).any() == False
        assert torch.isnan(self.sigma_y).any() == False

        # assert ((torch.square(self.sigma_x[:, None, None]) + self.clamp_minimum) >= 0).all()
        print((torch.square(self.sigma_x[:, None, None]) + self.minimum_sigma).min())

        assert ((torch.square(self.sigma_x[:, None, None]) + self.minimum_sigma) >= 0).all()
        assert ((torch.square(self.sigma_y[:, None, None]) + self.minimum_sigma) >= 0).all()
        

        gaussian_filter = None
        if self.multivariate:
            gaussian_filter = torch.exp(
                -0.5
                * (
                    # copy positions (vector) to each position. It is aligned with the 0th axis... [i, :, :] are same numbers (copied)
                    (
                        torch.square(
                            self.meshgrid_x_rotated + self.positions_x[:, None, None]
                        )
                        / (torch.square(self.sigma_x[:, None, None]) + self.minimum_sigma)
                    )
                    + (
                        torch.square(
                            self.meshgrid_y_rotated + self.positions_y[:, None, None]
                        )
                        / (torch.square(self.sigma_y[:, None, None]) + self.minimum_sigma)
                    )
                )
            )
        else:
            gaussian_filter = torch.exp(
                -0.5
                * (
                    # copy positions (vector) to each position. It is aligned with the 0th axis... [i, :, :] are same numbers (copied)
                    (
                        torch.square(
                            self.meshgrid_x_rotated + self.positions_x[:, None, None]
                        )
                        / (torch.square(self.sigma_x[:, None, None]) + self.minimum_sigma)
                    )
                    + (
                        torch.square(
                            self.meshgrid_y_rotated + self.positions_y[:, None, None]
                        )
                        / (torch.square(self.sigma_x[:, None, None]) + self.minimum_sigma)
                    )
                )
            )
        
        assert torch.isnan(gaussian_filter).any() == False
        assert torch.isnan(self.f).any() == False

        # The cosine filter also takes into account the position of the neuron (only the x coordinate)
        # and samples the cosine function at each point (with a learned frequency f).
        # As Energy Model has two gabor filters - odd and even gabor filter, we create both (they
        # differ only in 90deg shifted phase (90deg = pi/2))
        cosine_filter_odd = torch.cos(
            2
            * np.pi
            * (self.meshgrid_x_rotated + self.positions_x[:, None, None])
            * torch.clamp(self.f[:, None, None], min=self.f_clamp)
            + 0
        )  # phase = 0
        self.odd_gabor_filter = gaussian_filter * cosine_filter_odd
        cosine_filter_even = torch.cos(
            2
            * np.pi
            * (self.meshgrid_x_rotated + self.positions_x[:, None, None])
            * torch.clamp(self.f[:, None, None], min=self.f_clamp)
            + (np.pi / 2)
        )  # phase = 90deg = np.pi/2
        self.even_gabor_filter = (
            gaussian_filter * cosine_filter_even
        )  # shape [num_neurons, x_pixels, y_pixels]

        assert torch.isnan(self.odd_gabor_filter).any() == False
        assert torch.isnan(self.even_gabor_filter).any() == False

        

        # Now we apply the gabor filters on the image
        # because x is [batch_size, x_pixels, y_pixels] and gabor_filter is [num_neurons, x_pixels, y_pixels],
        # we do the tensordot in the corresponding dimensions matching x_pixels and y_pixels
        # it returns a tensor of shape [batch_size, num_neurons].. so there is a response of each filter
        # for each image in the batch
        filtered_image_odd = torch.tensordot(
            x, self.odd_gabor_filter, dims=[[1, 2], [1, 2]]
        )
        filtered_image_even = torch.tensordot(
            x, self.even_gabor_filter, dims=[[1, 2], [1, 2]]
        )

        assert torch.isnan(filtered_image_odd).any() == False
        assert torch.isnan(filtered_image_even).any() == False

        if torch.isnan(filtered_image_odd).any():
            print("----------------------------------")
            print(torch.isnan(self.odd_gabor_filter).any())
            print(torch.isnan(self.even_gabor_filter).any())
            print(torch.isnan(gaussian_filter).any())
            print(torch.isnan(self.sigma_x).any())
            print(torch.isnan(self.sigma_y).any())
            print(torch.isnan(self.f).any())

        # response of each neuron to each image in a batch .. therefore shape of [batch_size, num_neurons]
        # energy_model_response = self.scale * torch.sqrt(torch.square(filtered_image_odd) + torch.square(filtered_image_even)) + self.bias
        energy_model_response = self.scale[None, :] * torch.sqrt(
            torch.square(filtered_image_odd) + torch.square(filtered_image_even)
        ) + self.bias[None, :]

        


        # as the response has to be non-negative, we clamp the values
        energy_model_response_clamped = torch.clamp(energy_model_response, min=self.response_clamp_minimum)

        return energy_model_response_clamped

    def configure_optimizers(self):
        """Configures the optimizer for the training of the model (Adam).

        Returns:
            torch.optimizer: torch optimizer class
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def training_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        # print(torch.isnan(prediction).any())
        assert (prediction > 0).all()
        loss = self.loss(prediction, resp)
        # print('loss: ----------------------------------')
        # print(torch.isnan(loss).any())
        # print(torch.log(prediction + 1e-12).max())
        # print(torch.log(prediction + 1e-12).min())
        # print(torch.log(prediction + 1e-12).min())
        # # print(loss.max())
        # print(torch.isinf(torch.log(prediction + 1e-12)).any())
        # print(torch.isnan(torch.log(prediction + 1e-12)).any())
        # print(torch.isinf(loss).any())

        # print("kde je to posranyyyyyyyyyyyyyy")
        # print(prediction[torch.isnan(loss)])
        # print(resp[torch.isnan(loss)])

        self.log("train/loss", loss)
        # self.log("val/loss", loss)
        # self.log("val/sigma_x", self.sigma_x)
        # self.log("val/sigma_y", self.sigma_y)
        # self.log("val/f", self.f)
        # self.log("val/scale", self.scale)
        # self.log("val/bias", self.bias)
        return loss

    def validation_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        self.log("val/loss", loss)
        # self.log("val/sigma_x", self.sigma_x)
        # self.log("val/sigma_y", self.sigma_y)
        # self.log("val/f", self.f)
        # self.log("val/scale", self.scale)
        # self.log("val/bias", self.bias)

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
        self.predictions = predictions
        self.responses = responses
        print(correlation.max())
        print(correlation.argmax())
        print(correlation.min())
        print(correlation.argmin())
        self.log("val/corr", np.mean(correlation))
        print(np.mean(correlation))
        # self.visualize(444)

    def visualize(
        self,
        neuron_id,
        odd=True,
        init=False,
        title="",
        vmin=None,
        vmax=None,
        return_plt=False,
        ticks=True,
        cmap=None,
        graph_size=10,
    ):
        """Visualizes the neuron neuron_id.

        Args:
            neuron_id (int or str): Index of the neuron (within the filtered neurons so should be in [0, num_filtered))
                If neuron_id is "all", the visual field of all neurons is plotted
            odd (bool, optional): Show odd or even gabor filter. Defaults to True.
            title (str, optional): Title of the graph. If nothing provided, automatic title generated. Defaults to "".
            vmin (_type_, optional): _description_. Defaults to None.
            vmax (_type_, optional): _description_. Defaults to None.
            return_plt (bool, optional): Return plot or not. If not, it is plotted right away. Defaults to False.
            ticks (bool, optional): Use ticks or not. Defaults to True.
            cmap (str, optional): Which colormap to use. If None, cm.coolwarm is used. Defaults to None.
            graph_size (int, optional): Size of the graph. Defaults to 10.

        Returns:
            matplotlib.plot: The plot
        """

        # we need to have the whole network initialized, but some parts get initialized
        # only during the forward method execution. Therefore we have to create an artificial
        # input into the network and pass it through the network to initialize it (if it
        # is not initialized, otherwise this is redundant)
        if init:
            tmp_input = torch.rand((1, 1) + self.resolution)
            self.forward(tmp_input)

        if odd:
            gabor = self.odd_gabor_filter.cpu().detach().numpy().squeeze()
        else:
            gabor = self.even_gabor_filter.cpu().detach().numpy().squeeze()

        if neuron_id == "all":
            tmp = np.max(np.abs(gabor), axis=0)
            maximum = tmp.max()

            sorted = np.sort(tmp.flatten())
            plt.clf()
            plt.plot(sorted)
            plt.title("distribution of maximal values of positions in gabor filters")
            plt.show()
            # ten_percent = sorted[sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[2*sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[3*sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[4*sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[5*sorted.shape[0]//10]
            # print(ten_percent)
            # ten_percent = sorted[sorted.shape[0]//10]

            # print(ten_percent)
            gabor = (tmp > (maximum/1000)) * 1
            

        else:
            gabor = gabor[neuron_id, :, :]
        
        plt.clf()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gcf().set_size_inches(graph_size, graph_size)
        m = np.max(np.abs(gabor))
        if vmin is None:
            min = -m
        else:
            min = vmin
        if vmax is None:
            max = m
        else:
            max = vmax
        if cmap == "greys":
            color_map = "Greys_r"
        else:
            color_map = cm.coolwarm
        plt.imshow(gabor, vmax=max, vmin=min, cmap=color_map)

        if title == "" and neuron_id != "all":
            shifted_orientation = np.mod(
                self.orientations + self.default_ori_shift, np.pi
            )
            title = f"Neuron {neuron_id} \n with position [{self.positions_x[neuron_id]:.2f}, {self.positions_y[neuron_id]:.2f}] \n and pref. orientation {self.orientations[neuron_id]:.2f}rad, (with default shift:{shifted_orientation[neuron_id]:.2f}rad)"
        elif title == "" and neuron_id == "all":
            title = "Visual field of all neurons\n(where the position influences the response\nby at least 1/1000 of maximal influence of the filter)"

        plt.title(title)
        plt.colorbar()
        if ticks == False:
            plt.xticks([])
            plt.yticks([])
        if return_plt:
            return plt
        else:
            plt.show()

    def __str__(self):
        return "EnergyModel"


class EnergyModelLearnableShift(pl.LightningModule):
    def __init__(
        self, positions_x, positions_y, orientations, res, xlim, ylim, lr
    ):
        """

        Args:
            config (_type_): _description_
            positions_x (_type_): _description_
            positions_y (_type_): _description_
            orientations (_type_): in radians (0, np.pi)
            res (_type_): _description_
            xlim (_type_): _description_
            ylim (_type_): _description_
        """
        super().__init__()

        self.lr = lr
        self.loss = PoissonLoss()
        self.corr = Corr()
        self.num_neurons = len(orientations)  # number of neurons

        self.scale = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.rand(1))
        self.sigma_x = torch.nn.Parameter(torch.ones(1))
        self.sigma_y = torch.nn.Parameter(torch.ones(1))
        self.f = torch.nn.Parameter(torch.ones(1))
        self.ori_shift = torch.nn.Parameter(torch.rand(1))

        self.register_buffer(
            "positions_x", torch.from_numpy(positions_x.astype("float32"))
        )
        self.register_buffer(
            "positions_y", torch.from_numpy(positions_y.astype("float32"))
        )
        self.register_buffer(
            "orientations", torch.from_numpy(orientations.astype("float32"))
        )

        self.init_gabor_filters(res, xlim, ylim)

    def init_gabor_filters(self, res, xlim, ylim):
        """Creates a tensor (not trainable) of meshgrids of gabor filters
           with proper rotations.
           It registers it in the buffer (so that it can be moved to GPU easily).
           Gabors are of two types: even and odd -> therefore there are two types of all variables

           The tensors created are of shape [num_of_neurons, meshgrid_dimension]
             An example tensor (one of four created):
             - gabors_odd_meshgrid_X -> tensor of odd gabor filters with meshgrid for X axis

        Args:
            res (_type_): _description_
            xlim (_type_): _description_
            ylim (_type_): _description_
        """
        x = np.linspace(xlim[0], xlim[1], res[0])
        y = np.linspace(ylim[0], ylim[1], res[1])
        X, Y = np.meshgrid(x, y)

        # connecting two tuples to get shape: (self.num_neurons, X.shape[0], X.shape[1])
        meshgrid_shape_x = (self.num_neurons,) + X.shape
        meshgrid_shape_y = (self.num_neurons,) + Y.shape

        broadcasted_x = np.broadcast_to(X, meshgrid_shape_y)
        broadcasted_y = np.broadcast_to(Y, meshgrid_shape_y)

        # of shape [num_neurons, x_pixels, y_pixels]
        self.register_buffer(
            "meshgrid_x", torch.from_numpy(broadcasted_x.astype("float32"))
        )
        self.register_buffer(
            "meshgrid_y", torch.from_numpy(broadcasted_y.astype("float32"))
        )

    def forward(self, x):

        # because sometimes it is analogous to RGB, which has 3 channels, but in our case (grayscale), we have 1 channel
        # and with this we delete it
        if x.shape[1] == 1:
            x = torch.squeeze(x, 1)

        # we add the shift
        shifted_orientation = torch.remainder(self.orientations + self.ori_shift, np.pi)

        # and we rotate the grid
        meshgrid_x_rotated = self.meshgrid_x * torch.cos(
            shifted_orientation[:, None, None]
        ) + self.meshgrid_y * torch.sin(shifted_orientation[:, None, None])
        meshgrid_y_rotated = -self.meshgrid_x * torch.sin(
            shifted_orientation[:, None, None]
        ) + self.meshgrid_y * torch.cos(shifted_orientation[:, None, None])

        A = torch.exp(
            -0.5
            * (
                # copy positions (vector) to each position. It is aligned with the 0th axis... [i, :, :] are same numbers (copied)
                (
                    torch.square(meshgrid_x_rotated + self.positions_x[:, None, None])
                    / torch.square(self.sigma_x)
                )
                + (
                    torch.square(meshgrid_y_rotated + self.positions_y[:, None, None])
                    / torch.square(self.sigma_y)
                )
            )
        )

        B = torch.cos(
            2 * np.pi * (meshgrid_x_rotated + self.positions_x[:, None, None]) * self.f
            + 0
        )  # phase = 0
        odd_gabor_filter = A * B

        A = torch.exp(
            -0.5
            * (
                # copy positions (vector) to each position. It is aligned with the 0th axis... [i, :, :] are same numbers (copied)
                (
                    torch.square(meshgrid_x_rotated + self.positions_x[:, None, None])
                    / torch.square(self.sigma_x)
                )
                + (
                    torch.square(meshgrid_y_rotated + self.positions_y[:, None, None])
                    / torch.square(self.sigma_y)
                )
            )
        )

        B = torch.cos(
            2 * np.pi * (meshgrid_x_rotated + self.positions_x[:, None, None]) * self.f
            + (np.pi / 2)
        )  # phase = 90deg = np.pi/2
        even_gabor_filter = A * B  # shape [num_neurons, x_pixels, y_pixels]

        # Now we apply the gabor filters on the image
        # because x is [batch_size, x_pixels, y_pixels] and gabor_filter is [num_neurons, x_pixels, y_pixels],
        # we do the tensordot in the corresponding dimensions matching x_pixels and y_pixels
        # it returns a tensor of shape [batch_size, num_neurons].. so there is a response of each filter
        # for each image in the batch
        filtered_image_odd = torch.tensordot(x, odd_gabor_filter, dims=[[1, 2], [1, 2]])
        filtered_image_even = torch.tensordot(
            x, even_gabor_filter, dims=[[1, 2], [1, 2]]
        )

        # should be of shape [batch_size, num_neurons] ... response of each neuron to each image in a batch
        energy_model_response = (
            self.scale
            * torch.sqrt(
                torch.square(filtered_image_odd) + torch.square(filtered_image_even)
            )
            + self.bias
        )
        # energy_model_response = torch.sqrt(torch.square(filtered_image_odd) + torch.square(filtered_image_even))

        energy_model_response_clamped = torch.clamp(energy_model_response, min=0)

        return energy_model_response_clamped

    def configure_optimizers(self):
        """Configures the optimizer for the training of the model (Adam).

        Returns:
            torch.optimizer: torch optimizer class
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def training_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("val/loss", loss)
        self.log("val/corr", corr)
        self.log("ori_shift", self.ori_shift)

    def test_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("test/loss", loss)
        self.log("test/corr", corr)


### NO LEARNABLE ORIENTATION SHIFT


class EnergyModelNoShift(pl.LightningModule):
    def __init__(self, config, positions_x, positions_y, orientations, res, xlim, ylim):
        """

        Args:
            config (_type_): _description_
            positions_x (_type_): _description_
            positions_y (_type_): _description_
            orientations (_type_): in radians (0, np.pi)
            res (_type_): _description_
            xlim (_type_): _description_
            ylim (_type_): _description_
        """
        super().__init__()

        self.config = config
        self.loss = PoissonLoss()
        self.corr = Corr()
        self.num_neurons = len(orientations)  # number of neurons

        self.scale = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.rand(1))
        self.sigma_x = torch.nn.Parameter(torch.ones(1))
        self.sigma_y = torch.nn.Parameter(torch.ones(1))
        self.f = torch.nn.Parameter(torch.ones(1))

        self.register_buffer(
            "positions_x", torch.from_numpy(positions_x.astype("float32"))
        )
        self.register_buffer(
            "positions_y", torch.from_numpy(positions_y.astype("float32"))
        )

        self.orientations = orientations.astype(
            "float32"
        )  # we don't create a tensor, we won't use them (we will use only positions)

        self.init_gabor_filters(res, xlim, ylim)

    def init_gabor_filters(self, res, xlim, ylim):
        """Creates a tensor (not trainable) of meshgrids of gabor filters
           with proper rotations.
           It registers it in the buffer (so that it can be moved to GPU easily).
           Gabors are of two types: even and odd -> therefore there are two types of all variables

           The tensors created are of shape [num_of_neurons, meshgrid_dimension]
             An example tensor (one of four created):
             - gabors_odd_meshgrid_X -> tensor of odd gabor filters with meshgrid for X axis

        Args:
            res (_type_): _description_
            xlim (_type_): _description_
            ylim (_type_): _description_
        """
        x = np.linspace(xlim[0], xlim[1], res[0])
        y = np.linspace(ylim[0], ylim[1], res[1])
        X, Y = np.meshgrid(x, y)

        # connecting two tuples to get shape: (self.num_neurons, X.shape[0], X.shape[1])
        filters_shape_x = (self.num_neurons,) + X.shape
        filters_shape_y = (self.num_neurons,) + Y.shape

        gabors_odd_meshgrid_X = np.zeros(filters_shape_x)
        gabors_odd_meshgrid_Y = np.zeros(filters_shape_y)
        gabors_even_meshgrid_X = np.zeros(filters_shape_x)
        gabors_even_meshgrid_Y = np.zeros(filters_shape_y)

        for i in range(self.num_neurons):

            orientation = self.orientations[i]

            x = np.linspace(xlim[0], xlim[1], res[0])
            y = np.linspace(ylim[0], ylim[1], res[1])
            X, Y = np.meshgrid(x, y)

            X_rot = X * np.cos(orientation) + Y * np.sin(orientation)
            Y_rot = -X * np.sin(orientation) + Y * np.cos(orientation)

            gabors_odd_meshgrid_X[i, :, :] = X_rot
            gabors_odd_meshgrid_Y[i, :, :] = Y_rot
            gabors_even_meshgrid_X[i, :, :] = X_rot
            gabors_even_meshgrid_Y[i, :, :] = Y_rot

        self.register_buffer(
            "gabors_odd_x", torch.from_numpy(gabors_odd_meshgrid_X.astype("float32"))
        )
        self.register_buffer(
            "gabors_odd_y", torch.from_numpy(gabors_odd_meshgrid_Y.astype("float32"))
        )
        self.register_buffer(
            "gabors_even_x", torch.from_numpy(gabors_even_meshgrid_X.astype("float32"))
        )
        self.register_buffer(
            "gabors_even_y", torch.from_numpy(gabors_even_meshgrid_Y.astype("float32"))
        )

    def forward(self, x):

        if x.shape[1] == 1:
            x = torch.squeeze(x, 1)

        A = torch.exp(
            -0.5
            * (
                # copy positions (vector) to each position. It is aligned with the 0th axis... [i, :, :] are same numbers (copied)
                (
                    torch.square(self.gabors_odd_x + self.positions_x[:, None, None])
                    / torch.square(self.sigma_x)
                )
                + (
                    torch.square(self.gabors_odd_y + self.positions_y[:, None, None])
                    / torch.square(self.sigma_y)
                )
            )
        )

        B = torch.cos(
            2 * np.pi * (self.gabors_odd_x + self.positions_x[:, None, None]) * self.f
            + 0
        )  # phase = 0
        odd_gabor_filter = A * B

        A = torch.exp(
            -0.5
            * (
                # copy positions (vector) to each position. It is aligned with the 0th axis... [i, :, :] are same numbers (copied)
                (
                    torch.square(self.gabors_even_x + self.positions_x[:, None, None])
                    / torch.square(self.sigma_x)
                )
                + (
                    torch.square(self.gabors_even_y + self.positions_y[:, None, None])
                    / torch.square(self.sigma_y)
                )
            )
        )

        B = torch.cos(
            2 * np.pi * (self.gabors_even_x + self.positions_x[:, None, None]) * self.f
            + (np.pi / 2)
        )  # phase = 90deg = np.pi/2
        even_gabor_filter = A * B  # shape [num_neurons, x_pixels, y_pixels]

        # because x is [batch_size, x_pixels, y_pixels] and gabor_filter is [num_neurons, x_pixels, y_pixels],
        # we do the tensordot in the corresponding dimensions matching x_pixels and y_pixels
        # it returns a tensor of shape [batch_size, num_neurons].. so there is a response of each filter
        # for each image in the batch
        filtered_image_odd = torch.tensordot(x, odd_gabor_filter, dims=[[1, 2], [1, 2]])
        filtered_image_even = torch.tensordot(
            x, even_gabor_filter, dims=[[1, 2], [1, 2]]
        )

        # should be of shape [batch_size, num_neurons] ... response of each neuron to each image in a batch
        energy_model_response = (
            self.scale
            * torch.sqrt(
                torch.square(filtered_image_odd) + torch.square(filtered_image_even)
            )
            + self.bias
        )
        # energy_model_response = torch.sqrt(torch.square(filtered_image_odd) + torch.square(filtered_image_even))

        energy_model_response_clamped = torch.clamp(energy_model_response, min=0)

        return energy_model_response_clamped

    def configure_optimizers(self):
        """Configures the optimizer for the training of the model (Adam).

        Returns:
            torch.optimizer: torch optimizer class
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return opt

    def training_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("val/loss", loss)
        self.log("val/corr", corr)

    def test_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("test/loss", loss)
        self.log("test/corr", corr)


### EXTREMELY SLOW


class GaborFilterTorch(pl.LightningModule):
    def __init__(
        self, config, pos_x, pos_y, theta, sigma_x, sigma_y, sf, phase, res, xlim, ylim
    ):
        super().__init__()
        self.config = config

        self.loss = PoissonLoss()
        self.corr = Corr()

        # orientations and positions based on ground truth
        self.orientation = theta
        # self.pos_x = torch.from_numpy(pos_x).as_type(sigma_x)
        # self.pos_y = torch.from_numpy(pos_y).as_type(sigma_x)
        self.pos_x = pos_x.astype("float32")
        self.pos_y = pos_y.astype("float32")
        # self.register_buffer("pos_x", pos_x)
        # self.register_buffer("pos_y", pos_y)

        # learnable parameters shared between all Gabor Filters
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sf = sf

        # either 0 or np.pi/2
        self.phase = phase

        x = np.linspace(xlim[0], xlim[1], res[0])
        y = np.linspace(ylim[0], ylim[1], res[1])
        X, Y = np.meshgrid(x, y)

        # X = Parameter(torch.from_numpy(X), requires_grad=False)
        # Y = Parameter(torch.from_numpy(Y), requires_grad=False)
        # X = torch.from_numpy(X).type_as(self.sigma_x)
        # Y = torch.from_numpy(Y).type_as(self.sigma_x)

        # clockwise rotation
        # self.X_rot = X * np.cos(self.orientation) + Y * np.sin(self.orientation)
        # self.Y_rot = -X * np.sin(self.orientation) + Y * np.cos(self.orientation)
        X_rot = X * np.cos(self.orientation) + Y * np.sin(self.orientation)
        Y_rot = -X * np.sin(self.orientation) + Y * np.cos(self.orientation)

        self.register_buffer("X_rot", torch.from_numpy(X_rot.astype("float32")))
        self.register_buffer("Y_rot", torch.from_numpy(Y_rot.astype("float32")))

    def forward(self, x):
        """Returns the response of the Gabor Filter

        Args:
            x (torch.tensor): input image

        Returns:
            torch.float: response of the Gabor Filter
        """

        A = torch.exp(
            -0.5
            * (
                (torch.square(self.X_rot + self.pos_x) / torch.square(self.sigma_x))
                + (torch.square(self.Y_rot + self.pos_y) / torch.square(self.sigma_y))
            )
        )

        B = torch.cos(2 * np.pi * (self.X_rot + self.pos_x) * self.sf + self.phase)
        filter = A * B

        return torch.tensordot(x, filter)

    def configure_optimizers(self):
        """Configures the optimizer for the training of the model (Adam).

        Returns:
            torch.optimizer: torch optimizer class
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return opt

    def training_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        return loss

    def validation_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("val/loss", loss)
        self.log("val/corr", corr)

    def test_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("test/loss", loss)
        self.log("test/corr", corr)


class EnergyModelSlow(pl.LightningModule):
    def __init__(self, config, positions_x, positions_y, orientations, res, xlim, ylim):
        super().__init__()

        self.config = config
        self.loss = PoissonLoss()
        self.corr = Corr()
        self.num_neurons = len(orientations)

        self.scale = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.sigma_x = torch.nn.Parameter(torch.zeros(1))
        self.sigma_y = torch.nn.Parameter(torch.zeros(1))
        self.f = torch.nn.Parameter(torch.zeros(1))

        self.neurons_even = torch.nn.ModuleList(
            [
                GaborFilterTorch(
                    self.config,
                    positions_x[i],
                    positions_y[i],
                    orientations[i],
                    self.sigma_x,
                    self.sigma_y,
                    self.f,
                    0,
                    res,
                    xlim,
                    ylim,
                )
                for i in range(self.num_neurons)
            ]
        )
        self.neurons_odd = torch.nn.ModuleList(
            [
                GaborFilterTorch(
                    self.config,
                    positions_x[i],
                    positions_y[i],
                    orientations[i],
                    self.sigma_x,
                    self.sigma_y,
                    self.f,
                    np.pi / 2,
                    res,
                    xlim,
                    ylim,
                )
                for i in range(self.num_neurons)
            ]
        )

    def forward(self, x):
        responses = torch.zeros(
            (x.shape[0], self.num_neurons), device=self.device
        )  # (batch_size, num_of_neurons)
        for i in range(self.num_neurons):
            x1 = self.neurons_even[i](x)
            x2 = self.neurons_odd[i](x)

            y = self.scale * torch.sqrt(torch.square(x1) + torch.square(x2)) + self.bias
            responses[:, i] = y.squeeze()
        return responses

    def configure_optimizers(self):
        """Configures the optimizer for the training of the model (Adam).

        Returns:
            torch.optimizer: torch optimizer class
        """
        opt = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return opt

    def training_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        return loss

    def validation_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("val/loss", loss)
        self.log("val/corr", corr)

    def test_step(self, batch, batch_idx):
        img, resp = batch
        prediction = self.forward(img)
        loss = self.loss(prediction, resp)
        corr = self.corr(prediction, resp)
        self.log("test/loss", loss)
        self.log("test/corr", corr)
