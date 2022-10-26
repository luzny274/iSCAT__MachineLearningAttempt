# iSCAT__MachineLearningAttempt
## Description
This is an attempt to simulate images in the iSCAT microscope and then train some generic neural networks on them for focus prediction and microtubule localization.

## Usage
### Sample Generation
We can approximate the iSCAT image as a convolution between a Point spread function and a sample. Putting an image of nanoparticle or microtubule to some speckle background is then additive, which allows us to optimize the process.

We generated Point spread function using [Piscat](https://piscat.readthedocs.io/). To create the simulated image we then add speckle pattern with nanoparticles and microtubules, then we apply three different types of noises (add some out focus particles, perlin noise and most importantly poisson noise).

#### genMT.py
Convolves the PSF with lines of different lengths to simulate microtubule images and saves them, to be used by other scripts.

#### speckleGen.py
Simulates glass roughness and convolves it with the PSF to generate speckle patterns that are then used by other scripts for background.

#### sampleGen.py
Generates images with focus classification.

#### sampleGenBinary.py
Generates images with binary in focus and out of focus classification.

#### sampleGenSegmentation.py
Generates images with microtubules and corresponding images with segmented microtubules and background.

### Focus prediction
Training and testing Efficient net and Mobile net for focus classification.

### Image segmentation
Training and testing Deeplab and Unet for microtubule localization.

## Results
Resulted models work well on the simulated data, but not so well on real life iSCAT measurements.
