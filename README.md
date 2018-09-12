# ML-Recon

## Objective:

ML project to predict Nbody simulation output from initial condition.
Both input and output are particle displacement fields.

## File descriptions:

* `reconLPT2Nbody_uNet.py` : main excute files
* `periodic_padding.py` : code to fulfill periodic boundary padding
* `data_utils.py` : how to load data + test/analysis
* `model/BestModel.pt` : Best trained model
* `configs/config_unet.json` : most of the hyperparameters
* `Unet/uNet.py` : architecture
* `plot.py` : plot the result

## To run the code:

`python reconLPT2Nbody_uNet.py --config_file_path configs/config_unet.json`

or

`./reconLPT2Nbody_uNet.py -c configs/config_unet.json`

## Instruction:

1. Input raw data should be in the format of `x_y.npy` (y is in range of
(0,1000,1) and x is controled by `lIndex` and `hIndex` in
`configs/config_unet.json`  e.g. `0_0.npy`, `1_999.npy`). The shape of the data
in each file should be `(32,32,32,10)`, where the first coloumn is density, the
second to forth coloumn is (\phi_x, \phi_y,\phi_z) for ZA, the fifth to seventh
column is for 2LPT, and the eighth to tenth is for fastPM.
(Yu provides simulation files and each file contains 1000 simulations. I stored
the 1000 simulations in each file into separate files. The reason why I did
this is because GPU doesn't have enough memory to store all the files. Thus I
only provide the name and the path to each files.)

2. The output of the model is in the shape of `(6,32,32,32)` where
`(0:3,32,32,32)` stores the predicted fastPM simulations from uNet model and
`(3:6,32,32,32)` stores the corresponding real simulations.

3. The best trained model is stored in `model/BestModel.pt`. All the tests
(pancake, cosmology, etc) should be tested on this model.
You should only change the following parameters in `configs/config_unet.json`
to do different tests:
    * `base_data_path`: tell where the input (LPT/ZA) is stored.
    * `output_path`:  where do you want to store the output

4. The ZA/2LPT/fastPM data Yu provides are all stored in the following directory
on Nersc: `/global/homes/y/yfeng1/m3035/yfeng1/siyu-ml/`

5. I have wrote code `plot.py` to do all the plots. You can use it as a reference.
