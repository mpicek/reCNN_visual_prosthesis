# Software documentation

## Utilized libraries

The code in this repository is based on the following two fundamental libraries:
 - [neuralpredictors](https://github.com/sinzlab/neuralpredictors)
 - [predict_neural_responses](https://github.com/lucabaroni/predict_neural_responses)

In this document, we describe how these libraries are incorporated in the code.

### predict_neural_responses

**predict_neural_responses** is **Luca Baroni**'s repository. I have also some minor
contribution to this library. **I forked predict_neural_responses and added this fork into this
repository as a subrepository.** As a consequence, it is clear that the original author and the
owner of the code is Luca Baroni.


The foremost purpose of this library is to give some shared structure in our lab to the ML models for neural response prediction
and, as a consequence, have the models more easily reusable by other colleagues.
It is primarily accomplished by the class `encoding_model` in the `models.py` module.
We extended this library in `models.py` (file in reCNN_visual_prosthesis repository!)
and created a class `ExtendedEncodingModel` from which all our models inherit.


### Neuralpredictors

This library is mostly imported as a regular Python module by our scripts.

However, we also copied some pieces of this library and extended them for our
purposes. To be concrete, the following files reuse some code from Neuralpredictors:
 - `Lurz_dataset.py` - This is a reimplementation of Lurz's dataset that can be used with newer versions of Neuralpredictors library.
Moreover, it implements Pytorch Lightning LightningDataModule so that we can handle it more conveniently.
 - `core.py` - we copied class `RotationEquivariant2dCore` from `layers/cores/conv2d.py` located in Neuralpredictors. Then we modified
it for our purposes.
 - `readout.py` - similarly to `core.py`, `readout.py` contains a class `Gaussian3dCyclic` 
that was made from `Gaussian3d` class in `layers/readouts/gaussian.py` from Neuralpredictors. It was extended and modified
for the purposes of our work.

In all three files it is usually very clear what functions were added/edited/not changed.
However, it is always possible to use `diff` with the original Neuralpredictors library..

## metacentrum folder

This folder contains scripts for deployment of the models onto the MetaCentrum servers.

## experiments folder



