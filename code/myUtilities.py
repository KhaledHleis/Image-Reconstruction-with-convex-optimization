import numpy as np
import matplotlib.pyplot as plt

def define_noise_mask(dimension, ratio, seed=0):
    """
    This function generates an array of unique random indices.
    The length of the array is determined by a percentage of the image pixels.
    The returned shape is (round(dim*dim*per), 2).
    """
    np.random.seed(seed)
    num_pixels = int(np.round(dimension * dimension * ratio))
    unique_indices = set()

    while len(unique_indices) < num_pixels:
        # Generate random coordinates
        x = np.random.randint(0, dimension)
        y = np.random.randint(0, dimension)
        unique_indices.add((x, y))
    
    # Convert the set of unique indices to a NumPy array
    mask = np.array(list(unique_indices))
    return mask

def define_noise_mask_gaussien(dimension, ratio, seed=0):
    """
    This function generates an array of unique random indices.
    The length of the array is determined by a percentage of the image pixels.
    The returned shape is (round(dim*dim*per), 2).
    """
    np.random.seed(seed)
    num_pixels = int(np.round(dimension * dimension * ratio))
    unique_indices = set()

    mu = dimension / 2  # Center of the image
    sigma = dimension / 6  # Standard deviation: spread of noise

    while len(unique_indices) < num_pixels:
        # Generate Gaussian-distributed coordinates
        x = int(np.clip(np.random.normal(mu, sigma), 0, dimension - 1))
        y = int(np.clip(np.random.normal(mu, sigma), 0, dimension - 1))
        unique_indices.add((x, y))
    
    # Convert the set of unique indices to a NumPy array
    mask = np.array(list(unique_indices))
    return mask

def mask_Image(image, indeces):
    img = image.copy()
    for i, j in indeces:
        img[i, j] = 0
    return img


def get_mask_matrix(dimension, indeces):
    return mask_Image(np.ones((dimension, dimension)), indeces)


def invert_mask(dimension, mask_indeces):
    img_mask = get_mask_matrix(dimension,mask_indeces)
    inv_mask = np.where(img_mask == 1)
    return np.column_stack((inv_mask[0],inv_mask[1]))


def reset_pixels(img_donneur, img_recepteur, inv_mask_indeces):
    recep = img_recepteur.copy()
    for i, j in inv_mask_indeces:
        recep[i, j] = img_donneur[i, j]
    return recep

def NMSE(original,denoised):
    dist = np.linalg.norm(original - denoised)**2    
    return dist / np.linalg.norm(original)**2

def MSE(original,denoised):
    dist = np.mean((original - denoised)**2)
    return dist
def fill_zero(image,noise_range = (0,1),seed = 0):    
    """
    Replaces zero-valued pixels in the image with random noise.
    
    Parameters:
    - image: NumPy array representing the image.
    - noise_range: Tuple specifying the range (min, max) for noise values.
    - seed: Random seed for reproducibility.
    
    Returns:
    - Modified image with noise filling zero pixels.
    """
    np.random.seed(seed)
    
    # Create a copy to avoid modifying the original image
    modified_image = image.copy()
    
    # Find indices of zero-valued pixels
    zero_pixel_indices = np.where(modified_image == 0)
    
    # Generate random noise for the zero pixels
    noise = np.random.randint(noise_range[0], noise_range[1]+1, size=len(zero_pixel_indices[0]))
    
    # Replace zero pixels with noise
    modified_image[zero_pixel_indices] = noise
    
    return modified_image

def plot_3_images_side_by_side(original, noisy, denoised , titles = ['Original Image', 'Noisy Image', 'Denoised Image']):
    """
    Plots three grayscale images side by side with titles.
    
    Parameters:
    - original: NumPy array representing the original image.
    - noisy: NumPy array representing the noisy image.
    - denoised: NumPy array representing the denoised image.
    """
    # Set up the figure and axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define the images and titles
    images = [original, noisy, denoised]
    
    # Plot each image
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')  # Hide the axes for clarity
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def plot_2_images_side_by_side(original, reconstructed, titles = ['Original Image', 'Reconstructed Image']):
    """
    Plots two grayscale images side by side with titles: 'Original Image' and 'Reconstructed Image'.
    
    Parameters:
    - original: NumPy array representing the original image.
    - reconstructed: NumPy array representing the reconstructed image.
    """
    # Set up the figure and axis
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Define the images and titles
    images = [original, reconstructed]

    
    # Plot each image
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')  # Hide the axes for clarity
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_nmse_curves(NMSE_values_list, labels):
    """
    Plots multiple NMSE curves on a log-log scale.

    Parameters:
    - NMSE_values_list: List of arrays, each containing NMSE values for a different curve.
    - labels: List of strings, each corresponding to the label for a curve.
    """
    def start_from_one(curve,starting_point=1):
        return curve - np.max(curve) + starting_point
    plt.figure(figsize=(8, 6))

    for NMSE_values, label in zip(NMSE_values_list, labels):
        temp_NMSE_value = start_from_one(NMSE_values)
        plt.yscale('log')
        plt.plot(temp_NMSE_value, label=label)
    

    #plt.xscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('NMSE log')
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    # Show the plot
    plt.show()