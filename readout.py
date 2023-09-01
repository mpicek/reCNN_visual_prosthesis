from predict_neural_responses.models import *
import torch
from torch.nn import Parameter
from torch.nn import functional as F
import warnings
from experiments.utils import pickle_read


class Gaussian3dCyclic(readouts.Readout):
    """
    This readout instantiates an object that can be used to learn a point in the core feature space for each neuron,
    sampled from a Gaussian distribution with some mean and variance at train but set to mean at test time, that best predicts its response.

    In the third dimension, the Gaussian distribution is cyclical, therefore when it gets out of the range [-1, 1], it gets
    back to a cyclically corresponding position (1.5 = -0.5).

    The readout receives the shape of the core as 'in_shape', the number of units/neurons being predicted as 'outdims', 'bias' specifying whether
    or not bias term is to be used and 'init_range' range for initialising the mean and variance of the gaussian distribution from which we sample to
    uniform distribution, U(-init_mu_range,init_mu_range) and  uniform distribution, U(0.0, init_sigma_range) respectively.
    The grid parameter contains the normalized locations (x, y coordinates in the core feature space) and is clipped to [-1.1] as it a
    requirement of the torch.grid_sample function. The third parameter of the grid is the orientation, which is cyclical, as
    stated above. The feature parameter learns the best linear mapping between the feature
    map from a given location, sample from Gaussian at train time but set to mean at eval time, and the unit's response with or without an additional elu non-linearity.

    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]
        init_sigma_range (float): initialises sigma with Uniform([0.0, init_sigma_range]).
                It is recommended however to use a fixed initialization, for faster convergence.
                For this, set fixed_sigma to True.
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        fixed_sigma (bool). Recommended behavior: True. But set to false for backwards compatibility.
                If true, initialized the sigma not in a range, but with the exact value given for all neurons.
    """

    def __init__(
        self,
        in_shape,
        outdims,
        bias,
        init_mu_range,
        init_sigma_range,
        batch_sample=True,
        fixed_sigma=False,
        mean_activity=None,
        feature_reg_weight=1.0,
        gamma_readout=None,  # depricated, use feature_reg_weight instead
        do_not_sample=False,
        **kwargs,
    ):
        """The constructor

        Args:
            in_shape (list): shape of the input feature map [channels, width, height]
            outdims (int): number of output units
            bias (bool): adds a bias term
            init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                                [expected: positive value <=1]
            init_sigma_range (float): initialises sigma with Uniform([0.0, init_sigma_range]).
                    It is recommended however to use a fixed initialization, for faster convergence.
                    For this, set fixed_sigma to True.
            batch_sample (bool): if True, samples a position for each image in the batch separately
                                [default: True as it decreases convergence time and performs just as well]
            fixed_sigma (bool). Recommended behavior: True. But set to false for backwards compatibility.
                    If true, initialized the sigma not in a range, but with the exact value given for all neurons.
            mean_activity (tensor, optional): Tensor of mean activity of the neurons. Defaults to None.
            feature_reg_weight (float, optional): Regularization strength for the readout. Defaults to 1.0.
            gamma_readout (float, optional): Regularization for the readout. DO NOT USE, DEPRECATED Defaults to None.

        Raises:
            ValueError: If init_mu_range or init_sigma_range are not within required limit
        """
        super().__init__()
        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma_range <= 0.0:
            raise ValueError(
                "init_mu_range or init_sigma_range is not within required limit!"
            )
        self.in_shape = in_shape
        self.outdims = outdims
        self.feature_reg_weight = self.resolve_deprecated_gamma_readout(
            feature_reg_weight, gamma_readout
        )
        self.batch_sample = batch_sample
        self.grid_shape = (1, 1, outdims, 1, 3)
        self.do_not_sample = do_not_sample
        self.mu = Parameter(
            torch.Tensor(*self.grid_shape), requires_grad=False
        )  # mean location of gaussian for each neuron
        self.sigma = Parameter(
            torch.Tensor(*self.grid_shape)
        )  # standard deviation for gaussian for each neuron
        self.features = Parameter(
            torch.Tensor(1, 1, 1, outdims)
        )  # saliency weights for each channel from core
        self.mean_activity = mean_activity
        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.init_mu_range = init_mu_range
        self.init_sigma_range = init_sigma_range
        self.fixed_sigma = fixed_sigma
        self.initialize(mean_activity)

    def sample_grid(self, batch_size, sample=None): # significantly edited
        """Returns the grid locations from the core by sampling from a Gaussian distribution
        
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """

        # We won't clamp it to the interval [-1, 1]
        with torch.no_grad():
            #     self.mu.clamp_(min=-1, max=1)  # at eval time, only self.mu is used so it must belong to [-1,1]
            self.sigma.clamp_(min=0)  # sigma/variance is always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample

        if self.do_not_sample: # hardcore way to turn off sampling
            sample = False
        
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(
                *grid_shape
            ).zero_()  # for consistency and CUDA capability

        corrected_distribution = norm * self.sigma + self.mu

        # move the interval [-1, 1] by 1 to right to [0, 2], but values can be somewhere else (not in these intervals)
        # then compute the reminder when dividing by 2 (= move the values into the [0, 2] interval)
        # then move the interval back to [-1, 1] by subtracting 1
        # .. now we will have every dimension periodic
        all_periodic = torch.remainder((corrected_distribution + 1), 2) - 1

        # we clamp everything into the [-1, 1] interval
        all_clamped = torch.clamp(corrected_distribution, min=-1, max=1)

        # but we want to clamp only x and y dimensions and not z (channel) dimension,
        # therefore in all_clamped we replace clamped channels by periodic channels
        all_clamped[:, :, :, :, 2] = all_periodic[:, :, :, :, 2]

        return all_clamped

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    def initialize(self, mean_activity=None):
        """Initializes the readout.

        Args:
            mean_activity (tensor, optional): Tensor of mean activity of the neurons. Defaults to None.
        """
        if mean_activity is None:
            mean_activity = self.mean_activity
        self.mu.data.uniform_(-self.init_mu_range, self.init_mu_range)
        if self.fixed_sigma:
            self.sigma.data.uniform_(self.init_sigma_range, self.init_sigma_range)
        else:
            self.sigma.data.uniform_(self.init_sigma_range/2, self.init_sigma_range)
            warnings.warn(
                "sigma is sampled from uniform distribuiton, instead of a fixed value. Consider setting "
                "fixed_sigma to True"
            )
        self.features.data.fill_(1 / self.in_shape[0])
        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def regularizer(self, reduction="sum", average=None):
        return 0

    def forward(self, x, sample=None, shift=None, out_idx=None, **kwargs): # edited
        """Propagates the input forwards through the readout
        
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y (tensor): neuronal activity
        """

        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError(
                "the specified feature map dimension is not the readout's expected input dimension"
            )

        # we copy the first channel to the end to make it periodic
        with_copied_first_orientation = torch.cat(
            [x, x[:, 0, :, :].view(N, 1, w, h)], dim=1
        )

        with_copied_first_orientation = with_copied_first_orientation.view(
            N, 1, c + 1, w, h
        )

        feat = self.features
        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(
                batch_size=N, sample=sample
            )  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(
                N, outdims, 1, 3
            )

        if out_idx is not None:
            # out_idx specifies the indices to subset of neurons for training/testing
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, :, out_idx]
            grid = grid[:, :, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        #  - Gets values from a grid
        #  - align_corners=True, because we need to have -1 as the center of the
        #    orientation 0 and value 1 mapped to the center of (again) orientation
        #    0, but it is copied at the end
        y = F.grid_sample(
            with_copied_first_orientation,
            grid,
            align_corners=True,
            padding_mode="border",
            mode="bilinear",
        )

        # reshapes to a better shape
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y


class Gaussian3dCyclicNoScale(Gaussian3dCyclic):
    """
    This is just Gaussian3dCyclic but without parameters for scaling of the neural response for each individual neuron.

    Args:
        in_shape (list): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]
        init_sigma_range (float): initialises sigma with Uniform([0.0, init_sigma_range]).
                It is recommended however to use a fixed initialization, for faster convergence.
                For this, set fixed_sigma to True.
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        fixed_sigma (bool). Recommended behavior: True. But set to false for backwards compatibility.
                If true, initialized the sigma not in a range, but with the exact value given for all neurons.
    """

    def __init__(
        self,
        *args,
        ground_truth_positions_file_path="data/antolik/positions_reparametrized.pickle",
        ground_truth_orientations_file_path="data/antolik/oris_reparametrized.pickle",
        init_to_ground_truth_positions=False,
        init_to_ground_truth_orientations=False,
        freeze_positions=False,
        freeze_orientations=False,
        orientation_shift=87.42857142857143, #in degrees
        factor = 5.5,
        filtered_neurons=None,
        # dataloader=None,
        positions_minus_x=False,
        positions_minus_y=False,
        do_not_sample=False,
        **kwargs,
    ):
        """The constructor
        """
        self.ground_truth_positions_file_path = ground_truth_positions_file_path
        self.ground_truth_orientations_file_path = ground_truth_orientations_file_path
        self.init_to_ground_truth_positions = init_to_ground_truth_positions
        self.init_to_ground_truth_orientations = init_to_ground_truth_orientations
        self.freeze_positions = freeze_positions
        self.freeze_orientations = freeze_orientations
        self.orientation_shift = orientation_shift
        self.factor = factor
        self.filtered_neurons = filtered_neurons
        # self.dataloader = dataloader
        self.positions_minus_x = positions_minus_x
        self.positions_minus_y = positions_minus_y
        self.do_not_sample = do_not_sample

        self.config = kwargs

        super().__init__(*args, do_not_sample=do_not_sample, **kwargs)
        

    def initialize(self, mean_activity=None):
        """Initializes the readout.

        Args:
            mean_activity (tensor, optional): Tensor of mean activity of the neurons. Defaults to None.
        """

        if mean_activity is None:
            mean_activity = self.mean_activity

        # firstly init the data, then can be reinitialized (bellow)
        self.mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.fixed_sigma:
            self.sigma.data.uniform_(self.init_sigma_range, self.init_sigma_range)
        else:
            self.sigma.data.uniform_(0, self.init_sigma_range)
            warnings.warn(
                "sigma is sampled from uniform distribution, instead of a fixed value. Consider setting "
                "fixed_sigma to True"
            )
        
        # TODO
        # # if we are supposed to freeze positions from the beginning, we set the
        # # corresponding part of the sigmas to zero (=> no update)
        # if self.freeze_positions:
        #     self.sigma.data[0,0,:,0,:2] = torch.zeros(self.sigma.data[...,:2].shape)
        
        # if self.freeze_orientations:
        #     self.sigma.data[0,0,:,0,2] = torch.zeros(self.sigma.data[...,2].shape)

        if self.bias is not None:
            self.initialize_bias(mean_activity=mean_activity)

    def init_neurons(self, dataloader):
        pos_x = None
        pos_y = None
        target_ori = None

        if dataloader and (self.init_to_ground_truth_positions or self.init_to_ground_truth_orientations):
            pos_x, pos_y, target_ori = dataloader.get_ground_truth(
                ground_truth_positions_file_path=self.ground_truth_positions_file_path,
                ground_truth_orientations_file_path=self.ground_truth_orientations_file_path,
                positions_minus_x=self.positions_minus_x,
                positions_minus_y=self.positions_minus_y,
                in_degrees=True,
            )

        if self.init_to_ground_truth_positions:
            pos_x = torch.from_numpy(pos_x)
            pos_y = torch.from_numpy(pos_y)

            # works also when the stimulus is cropped (self.get_stimulus_visual_angle()
            # returns the visual angle corrected after the stimulus crop)
            self.mu.data[0,0,:,0,0] = pos_x / (dataloader.get_stimulus_visual_angle() / 2)
            self.mu.data[0,0,:,0,1] = pos_y / (dataloader.get_stimulus_visual_angle() / 2) # TODO: tady jsem to delil jen self.factor


        if self.init_to_ground_truth_orientations:
            shifted_ori = (target_ori + self.orientation_shift) % 180 # TODO: minus or plus
            normalized_ori = shifted_ori / 180 # from [0, 180] to [0, 1].. for the network
            normalized_ori = torch.from_numpy(normalized_ori)

            self.mu.data[0,0,:,0,2] = (normalized_ori) # = normalized_ori


    def forward(self, x, sample=None, shift=None, out_idx=None, **kwargs): # edited
        """Propagates the input forwards through the readout
        
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted

        Returns:
            y (tensor): neuronal activity
        """

        N, c, w, h = x.size()
        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError(
                f"the specified feature map dimension is not the readout's expected input dimension. Came in: {x.shape[1:]}, but we want {self.in_shape}"
            )

        # we copy the first channel to the end to make it periodic
        with_copied_first_orientation = torch.cat(
            [x, x[:, 0, :, :].view(N, 1, w, h)], dim=1
        )

        with_copied_first_orientation = with_copied_first_orientation.view(
            N, 1, c + 1, w, h
        )

        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(
                batch_size=N, sample=sample
            )  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(
                N, outdims, 1, 3
            )

        if out_idx is not None:
            # out_idx specifies the indices to subset of neurons for training/testing
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            grid = grid[:, :, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        #  - Gets values from a grid
        #  - align_corners=True, because we need to have -1 as the center of the
        #    orientation 0 and value 1 mapped to the center of (again) orientation
        #    0, but it is copied at the end
        y = F.grid_sample(
            with_copied_first_orientation,
            grid,
            align_corners=True,
            padding_mode="border",
            mode="bilinear",
        )

        # reshapes to a better shape
        y = y.squeeze(-1).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y