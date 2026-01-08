import numpy as np
from scipy.ndimage import convolve

def conditional_smooth(image):
    """
    Apply 3x3 smoothing to pixels that are not completely surrounded by black pixels.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        
    Returns:
        Smoothed image as numpy array
    """
    # Convert to numpy array if needed
    image = np.array(image, dtype=np.float32)
    
    # Handle both grayscale and color images
    if len(image.shape) == 2:
        # Grayscale image
        return _smooth_single_channel(image)
    elif len(image.shape) == 3:
        # Color image - process each channel separately
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = _smooth_single_channel(image[:, :, c])
        return result
    else:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")


def _smooth_single_channel(channel):
    """
    Apply conditional smoothing to a single channel.
    Special case: If a black pixel is surrounded by 8 non-black pixels, 
    it takes the average of those 8 neighbors (excluding itself).
    
    Args:
        channel: 2D numpy array representing a single channel
        
    Returns:
        Smoothed channel
    """
    h, w = channel.shape
    result = channel.copy()
    
    # Create a mask for black pixels (value == 0)
    is_black = (channel == 0)
    
    # Check if pixel and all neighbors are black using max pooling
    # If max in 3x3 window > 0, then at least one pixel is non-black
    local_max = convolve(channel, np.ones((3, 3)), mode='constant', cval=0.0)
    should_smooth = (local_max > 0)
    
    # Count non-black neighbors (excluding center pixel)
    # Create a kernel that doesn't count the center
    neighbor_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]], dtype=np.float32)
    
    # Count non-black neighbors
    non_black_mask = (channel > 0).astype(np.float32)
    non_black_count = convolve(non_black_mask, neighbor_kernel, mode='constant', cval=0.0)
    
    # Sum of neighbor values (excluding center)
    neighbor_sum = convolve(channel, neighbor_kernel, mode='constant', cval=0.0)
    
    # For black pixels with all 8 neighbors non-black, use average of neighbors only
    black_with_nonblack_neighbors = is_black & (non_black_count == 8)
    
    # For other pixels, use standard 3x3 average including center
    kernel_with_center = np.ones((3, 3)) / 9.0
    smoothed_with_center = convolve(channel, kernel_with_center, mode='constant', cval=0.0)
    
    # Average of 8 neighbors (excluding center)
    smoothed_neighbors_only = neighbor_sum / 8.0
    
    # Apply appropriate smoothing based on condition
    result = channel.copy()
    result = np.where(black_with_nonblack_neighbors, smoothed_neighbors_only, result)
    result = np.where(should_smooth & ~black_with_nonblack_neighbors, smoothed_with_center, result)
    
    return result


def conditional_smooth_optimized(image):
    """
    Optimized version using vectorized operations.
    Apply 3x3 smoothing to pixels that are not completely surrounded by black pixels.
    Special case: If a black pixel is surrounded by 8 non-black pixels, 
    it takes the average of those 8 neighbors (excluding itself).
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        
    Returns:
        Smoothed image as numpy array
    """
    # Convert to numpy array if needed
    image = np.array(image, dtype=np.float32)
    original_dtype = image.dtype
    
    # Handle both grayscale and color images
    if len(image.shape) == 2:
        # Add channel dimension for uniform processing
        image = image[:, :, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False
    
    h, w, c = image.shape
    result = image.copy()
    
    # Process each channel
    for ch in range(c):
        channel = image[:, :, ch]
        
        # Create a mask for black pixels (value == 0)
        is_black = (channel == 0)
        
        # Check if pixel and all neighbors are black using max pooling
        local_max = convolve(channel, np.ones((3, 3)), mode='constant', cval=0.0)
        should_smooth = (local_max > 0)
        
        # Count non-black neighbors (excluding center pixel)
        neighbor_kernel = np.array([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=np.float32)
        
        # Count non-black neighbors
        non_black_mask = (channel > 0).astype(np.float32)
        non_black_count = convolve(non_black_mask, neighbor_kernel, mode='constant', cval=0.0)
        
        # Sum of neighbor values (excluding center)
        neighbor_sum = convolve(channel, neighbor_kernel, mode='constant', cval=0.0)
        
        # For black pixels with all 8 neighbors non-black, use average of neighbors only
        black_with_nonblack_neighbors = is_black & (non_black_count == 8)
        
        # For other pixels, use standard 3x3 average including center
        kernel_with_center = np.ones((3, 3)) / 9.0
        smoothed_with_center = convolve(channel, kernel_with_center, mode='constant', cval=0.0)
        
        # Average of 8 neighbors (excluding center)
        smoothed_neighbors_only = neighbor_sum / 8.0
        
        # Apply appropriate smoothing based on condition
        result[:, :, ch] = channel.copy()
        result[:, :, ch] = np.where(black_with_nonblack_neighbors, smoothed_neighbors_only, result[:, :, ch])
        result[:, :, ch] = np.where(should_smooth & ~black_with_nonblack_neighbors, smoothed_with_center, result[:, :, ch])
    
    # Remove channel dimension if input was grayscale
    if squeeze_output:
        result = result[:, :, 0]
    
    return result.astype(original_dtype)


if __name__ == "__main__":
    import argparse
    from PIL import Image
    
    parser = argparse.ArgumentParser(
        description='Apply conditional 3x3 smoothing to an image. '
                    'Smoothing is only applied to pixels where at least one pixel '
                    'in the 3x3 neighborhood (including itself) is non-black.'
    )
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('output', type=str, help='Path to output image')
    parser.add_argument('--visualize', action='store_true', 
                        help='Create a side-by-side comparison visualization')
    
    args = parser.parse_args()
    
    try:
        # Load image
        print(f"Loading image from: {args.input}")
        img = Image.open(args.input)
        img_array = np.array(img)
        
        print(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Apply conditional smoothing
        print("Applying conditional smoothing...")
        smoothed = conditional_smooth_optimized(img_array)
        
        # Save result
        print(f"Saving result to: {args.output}")
        result_img = Image.fromarray(smoothed.astype(np.uint8))
        result_img.save(args.output)
        print("Done!")
        
        # Create visualization if requested
        if args.visualize:
            import matplotlib.pyplot as plt
            
            vis_path = args.output.rsplit('.', 1)[0] + '_comparison.png'
            print(f"Creating visualization: {vis_path}")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(img_array)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            axes[1].imshow(smoothed.astype(np.uint8))
            axes[1].set_title('Conditionally Smoothed')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {vis_path}")
            
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
