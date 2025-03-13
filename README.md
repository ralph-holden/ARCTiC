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


### Manual installation

*   The code has the following dependencies:

1. `torch` – PyTorch for deep learning.
2. `torchvision` – Computer vision utilities (datasets, transforms, models).
3. `timm` – Pretrained models from `rwightman/pytorch-image-models`.
4. `cryocat` – Includes `cryomap` for handling cryo-ET images.
5. `numpy` – Numerical computing.
6. `matplotlib` – Visualization library for plots.
7. `tqdm` – Progress bars.
8. `scikit-learn` – Machine learning utilities, including classification reports and confusion matrices.
9. `pillow` – Image processing (`PIL.Image`).

## Fine-tuned Models

*   Create a directory with fine-tuned models.

```
mkdir -p <models>
```

*   Download fine-tuned binary and multiclass models from 
    [ownCloud](https://oc.biophys.mpg.de/owncloud/s/zmMZPr2TEB4Bwda)
    and put them into `<models>`.


## Usage

* To run the script from the command line, use the following syntax:

```
python run_TS_cleaning.py --input_ts 'input_TS.mrc' --cleaned_ts 'cleaned_TS.mrc' --angle_start -50 --angle_step 2 --pdf_output 'output_visualization.pdf' --model 'models/swin_tiny_fine-tuned.pth'
```

## Arguments:
1. `--input_ts` `<path to input .mrc file>` (Required)
   - **Description:** Path to the input `.mrc` file, which contains the tilt series data to be processed.
   - **Example:** `'input_TS.mrc'`

2. `--cleaned_ts` `<path to output .mrc file>` (Required)
   - **Description:** Path to the output `.mrc` file where the cleaned tilt series will be saved.
   - **Example:** `'cleaned_TS.mrc'`

3. `--angle_start` `<float>` (Optional, default: `-50`)
   - **Description:** The starting tilt angle for visualizing tilt images. 
   - **Example:** `-50`

4. `--angle_step` `<float>` (Optional, default: `2`)
   - **Description:** The increment (step size) for the tilt angles between consecutive tilts.
   - **Example:** `2`

5. `--pdf_output` `<path to output PDF file>` (Required)
   - **Description:** Path to the PDF file where the visualizations (tilt angles and excluded images with probability bars) will be saved.
   - **Example:** `'output_visualization.pdf'`

6. `--model` `<path to model file>` (Required)
   - **Description:** Path to the pre-trained model file (e.g., a Swin transformer model) that will be used for classifying images. The model should be compatible with the network architecture specified in the script.
   - **Example:** `'models/swin_tiny_fine-tuned.pth'`


### This command will:

1. Load the input tilt series data from `input_TS.mrc`.
2. Use the `swin_tiny_fine_tuned.pth` model to clean TS and visualize tilt angles.
3. Start tilt visualization at `-50` degrees with a step of `2` degrees.
4. Generate and save the visualizations (tilt angle and classification probability scale bars) into `output_visualization.pdf`.
5. Save the cleaned tilt series to `cleaned_TS.mrc`.


## Additional Notes

- Ensure that the model file (`.pth`) is compatible with the architecture defined in the script (e.g., `swin_tiny` or `swin_large`).
- You can adjust the `--angle_start` and `--angle_step` to suit the dataset's tilt range or imaging setup.
- The `--pdf_output` argument generates a PDF containing the tilt angle visualization and image classifications for easy review.


## Jupyter Notebooks

- In the `notebooks` directory, there are additional Jupyter Notebooks that were used for augmentation (`augmentation.ipynb`), data split (`split_train_val_test.ipynb`), and examples of training and evaluation scripts.


## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
