# ARCTiC
Automated Removal of Corrupted Tilts in Cryo-ET


## Installation

*   Get ARCTiC source codes

```
git clone https://github.com/turonova/ARCTiC
cd ARCTiC
```

### Using environment.yml

*   Create ARCTiC conda environment using .yml file

```
conda env create -f environment.yml
```

The environment is based on Python 3.10 and uses Conda (conda-forge channel) with additional pip packages.
CUDA-Enabled PyTorch: The environment uses torch==2.5.0+cu118 (CUDA 11.8).


## Fine-trained Models

*   Create a directory with fine-trained models.

```
mkdir -p <models>
```

*   Download fine-trained binary and multiclass models from 
    [owncloud](https://oc.biophys.mpg.de/owncloud/s/zmMZPr2TEB4Bwda)
    and put them into `<models>`.