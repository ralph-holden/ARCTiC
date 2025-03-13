import torch
import torch.nn as nn
import timm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import transforms, models
from cryocat import cryomap
from PIL import Image

# Define a function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images using a pre-trained model.")
    
    # Define arguments
    parser.add_argument('--input_ts', type=str, required=True, help="Input MRC file")
    parser.add_argument('--cleaned_ts', type=str, required=True, help="Output cleaned MRC file")
    parser.add_argument('--angle_start', type=float, required=True, help="Start angle for tilt angle visualization")
    parser.add_argument('--angle_step', type=float, required=True, help="Step size for angle increment")
    parser.add_argument('--pdf_output', type=str, default="output_visualization.pdf", help="Output PDF file for visualizations")
    parser.add_argument('--model', type=str, required=True, help="Path to the pre-trained model")

    return parser.parse_args()

# Parse the command-line arguments
args = parse_arguments()

# Assign parsed arguments to variables
INPUT_TS = args.input_ts
CLEANED_TS = args.cleaned_ts
ANGLE_START = args.angle_start
ANGLE_STEP = args.angle_step
PDF_OUTPUT = args.pdf_output
MODEL = args.model

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model based on the provided model type
model_mapping = {
    'swin_tiny': lambda: timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=2),
    'swin_large': lambda: timm.create_model('swin_large_patch4_window7_224', pretrained=False, num_classes=2),
    'resnet': lambda: modify_resnet(),
    'efficientnet': lambda: modify_efficientnet(),
}

def modify_resnet():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def modify_efficientnet():
    model = models.efficientnet_b3(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

for key in model_mapping:
    if key in MODEL:
        model = model_mapping[key]()
        break
else:
    raise ValueError("MODEL file must contain 'swin_tiny', 'swin_large', 'resnet', or 'efficientnet'")

# Load the model's state_dict
model.load_state_dict(torch.load(MODEL))
model = model.to(device)
model.eval()
print('The model has been loaded successfully!')

# Define image transformation (match your test_transforms)
size = (320, 320) if 'efficientnet' in MODEL else (224, 224)
image_transforms = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
])


def evaluate_single_image(image_input, index, class_0_info, class_1_info):
    # Load and preprocess the image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    image = image_transforms(image).unsqueeze(0).to(device)

    # Forward pass through the model
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

    # Display probabilities and classification
    predicted_class = np.argmax(probabilities)

    if predicted_class == 0:
        class_0_info.append((index, probabilities[0]))  # Append index and probability for class 0
    else:
        class_1_info.append((index, probabilities[1]))  # Append index and probability for class 1

    return predicted_class


# Initialize variables
mrc = cryomap.read(f"{INPUT_TS}")

tomo3d = []
class_0_info = []  # To store (index, probability) for class 0
class_1_info = []  # To store (index, probability) for class 1

# Plot setup
prev_text = False

# Create a PDF to store all figures
with PdfPages(PDF_OUTPUT) as pdf:
    
    # First figure (Tilt Angle Visualization)
    fig = plt.figure(figsize=(5, 5))
    plt.axis('off')

    # Evaluate selected slices
    print('Processing TS now!')
    for i in range(0, mrc.shape[2]):  
        angle = ANGLE_START + i * ANGLE_STEP  # Increment the angle by angle_step for each slice
        image_b16 = cryomap.scale(mrc[:, :, i], 0.0625)
        image_b16 = ((image_b16 - image_b16.min()) * (1 / (image_b16.max() - image_b16.min()) * 255)).astype('uint8')
        image_b16 = Image.fromarray(image_b16)
        
        if image_b16.mode != 'RGB':
            image_b16 = image_b16.convert('RGB')

        correct_tilt = evaluate_single_image(image_b16, i, class_0_info, class_1_info)
        angle = np.radians(angle)   # Convert angle to radians for plotting
        if correct_tilt:
            tomo3d.append(mrc[:, :, i])
            plt.plot([-np.cos(angle), np.cos(angle)], [-np.sin(angle), np.sin(angle)], color='black', linewidth=1)
            prev_text = False
        else:
            plt.plot([-np.cos(angle), np.cos(angle)], [-np.sin(angle), np.sin(angle)], color='red', linewidth=1, linestyle='--')
            if not prev_text:
                plt.text(np.cos(angle) * 1.01, np.sin(angle) * 1.09, str(i+1), fontsize=12, color='red')
            prev_text = not prev_text

    # Add caption to the first page
    fig.text(0.5, 0.95, "Tilt Angle Visualization", ha='center', fontsize=14, weight='bold')

    # Save the first figure to the PDF
    pdf.savefig()
    plt.close()

    # Second figure (Images with Probability Scale Bar)
    num_images = len(class_0_info)
    cols = 3  # Number of columns in the grid layout
    rows = (num_images // cols) + (num_images % cols > 0)

    # Create a new figure for images with probability scale bar
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 3))
    axes = axes.flatten()  # Flatten to iterate easily

    fig.subplots_adjust(top=0.8, hspace=0.5, wspace=0.5)
    
    print('Exporting PDF now!')
    for i, (index, prob) in enumerate(class_0_info):
        # Load the corresponding image_b16 image
        image_b16 = cryomap.scale(mrc[:, :, index], 0.0625)
        image_b16 = ((image_b16 - image_b16.min()) * (1 / (image_b16.max() - image_b16.min()) * 255)).astype('uint8')
        image_b16 = Image.fromarray(image_b16)

        # Display image on subplot
        ax = axes[i]
        ax.imshow(image_b16, cmap='gray')
        ax.axis('off')  # Remove axis for cleaner presentation

        colors = ['red'] * int(prob * 100) + ['black'] * int((1 - prob) * 100)
        discrete_cmap = ListedColormap(colors)

        # Add probability scale bar next to each image
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=discrete_cmap, norm=plt.Normalize(vmin=0, vmax=1)),
            ax=ax, orientation='vertical', fraction=0.046, pad=0.04
        )
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels([f'{int(t * 100)}%' for t in [0, 0.5, 1]])

        # Title showing image index and probability
        ax.set_title(f"Index: {index+1} | Prob: {prob:.2%}")

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add caption to the second page
    fig.text(0.5, 0.9, "Excluded Tilt Images with Probability Scale Bar", ha='center', fontsize=14, weight='bold')

    # Save the second figure to the PDF
    pdf.savefig()
    plt.close()

# Stack the tomo3d list to create a 3D volume and save it
print('Saving cleaned TS!')
tomo3d = np.stack(tomo3d, axis=2)
cryomap.write(tomo3d, CLEANED_TS, data_type=np.single)
print('Process completed successfully!')
