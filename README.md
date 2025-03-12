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


## Usage

* To run the script from the command line, use the following syntax:

```
python run_TS_cleaning.py --input_ts 'input_TS.mrc' --cleaned_ts 'cleaned_TS.mrc' --angle_start -50 --angle_step 2 --pdf_output 'output_visualization.pdf' --model 'models/swin_tiny_fine_tuned.pth'
```

### This command will:

1. Load the input tomogram data from \`input_TS.mrc\`.
2. Use the \`swin_tiny_fine_tuned.pth\` model to clean TS and visualize tilt angles.
3. Start tilt visualization at \`-50\` degrees with a step of \`2\` degrees.
4. Generate and save the visualizations (tilt angle and classification probability scale bars) into \`output_visualization.pdf\`.
5. Save the cleaned 3D volume to \`cleaned_TS.mrc\`.

## Additional Notes

- Ensure that the model file (\`.pth\`) is compatible with the architecture defined in the script (e.g., "swin_tiny" or "swin_large").
- You can adjust the \`--angle_start\` and \`--angle_step\` to suit the dataset's tilt range or imaging setup.
- The \`--pdf_output\` argument generates a PDF containing the tilt angle visualization and image classifications for easy review.
