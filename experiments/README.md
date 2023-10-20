# Experiments folder

This file provides a brief description of files in this folder

# Maximal Excitatory Image (MEI)

### `MEI.ipynb` notebook
 - Find and visualize MEI for Bottleneck Model (BM), Energy Model (EM) and the encoder

### `generate_data_for_analysis.ipynb` notebook
 - generates data for further analysis of the BM, EM and encoder models.
 - the script takes approximately 1 hour to run
 - the results are saved to `experiments/MEI_experiments.csv` file, which is subsequently used in `analysis_of_models.ipynb`

### `analysis_of_models.ipynb` notebook
 - in order to run this notebook, run `generate_data_for_analysis.ipynb` notebook first, it'll generate data for this notebook (`experiments/MEI_experiments.csv`)
 - we mostly plot the dependence of stimulus grating frequency on the response of particular neurons
 - neurons from 0 to 10 and 100, 187 and 615 were analyzed
 - plotting done for different stimulus phases and also for averaged responses over different phases
 - we found a bias in BM and encoder models to be more responsive when the grating has a color change in the middle of the visual field
    - it's most probably the cause of Large Spiking V1 Model (LSV1M)
 - the neurons 100, 187 and 615 were analyzed as they are in the middle of the visual field. When the grating is generated, it is generated relative to this center. Therefore, the phase behaves as expected at this point. On other locations of the visual field, the rotation of the grating may also shift the phase, which causes a problem when trying different frequencies - the different frequency changes the phase, too, leading to oscillating response (because even though we might think the phase is constant, it is not).
 - then we masked the grating according to the visual field of each neuron (according to the MEI) and ordered the stimuli according to their response. From that it was clear that the change from white to black has to be in the middle to maximize the response.

# Other experiments

### `visualize_reCNN_model.ipynb` notebook
 - visualization of reCNN model (orientation map)

### `nice_orientation_map.py` notebook
 - a nicer visualization of the orientation map 