import numpy as np
import cv2
import argparse
from PIL import Image


def inpaint_holes(image, method='telea', radius=3):
    """
    Fill holes (black regions) in an image using inpainting.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        method: Inpainting method - 'telea' or 'ns' (Navier-Stokes)
        radius: Radius of circular neighborhood of each point to inpaint
        
    Returns:
        Inpainted image as numpy array
    """
    # Convert to numpy array if needed
    image = np.array(image)
    original_dtype = image.dtype
    
    # Handle grayscale vs color
    if len(image.shape) == 2:
        # Grayscale image
        img_to_inpaint = image
        is_grayscale = True
    elif len(image.shape) == 3:
        # Color image
        img_to_inpaint = image
        is_grayscale = False
    else:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    # Create mask: 1 for black pixels (holes to fill), 0 for valid pixels
    if is_grayscale:
        mask = (img_to_inpaint == 0).astype(np.uint8) * 255
    else:
        # For color images, consider a pixel black if all channels are 0
        mask = (np.all(img_to_inpaint == 0, axis=2)).astype(np.uint8) * 255
    
    # Check if there are any holes to fill
    if not np.any(mask):
        print("No holes detected in the image.")
        return image
    
    # Convert image to uint8 if needed (OpenCV inpaint requires uint8)
    if img_to_inpaint.dtype != np.uint8:
        # Normalize to 0-255 range
        img_min, img_max = img_to_inpaint.min(), img_to_inpaint.max()
        if img_max > img_min:
            img_normalized = ((img_to_inpaint - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_normalized = img_to_inpaint.astype(np.uint8)
    else:
        img_normalized = img_to_inpaint
    
    # Select inpainting method
    if method.lower() == 'telea':
        inpaint_method = cv2.INPAINT_TELEA
    elif method.lower() == 'ns':
        inpaint_method = cv2.INPAINT_NS
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'telea' or 'ns'.")
    
    # Perform inpainting
    result = cv2.inpaint(img_normalized, mask, radius, inpaint_method)
    
    # Convert back to original dtype if needed
    if original_dtype != np.uint8:
        if img_max > img_min:
            result = (result.astype(np.float32) / 255.0 * (img_max - img_min) + img_min).astype(original_dtype)
        else:
            result = result.astype(original_dtype)
    
    return result


def inpaint_holes_adaptive(image, method='telea', initial_radius=3, max_iterations=5):
    """
    Iteratively fill holes using inpainting with increasing radius.
    Useful for filling larger holes.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W)
        method: Inpainting method - 'telea' or 'ns' (Navier-Stokes)
        initial_radius: Starting radius for inpainting
        max_iterations: Maximum number of iterations
        
    Returns:
        Inpainted image as numpy array
    """
    result = image.copy()
    
    for i in range(max_iterations):
        radius = initial_radius * (2 ** i)
        print(f"Iteration {i+1}/{max_iterations}: Using radius {radius}")
        
        # Check if there are still holes
        if len(result.shape) == 2:
            remaining_holes = np.any(result == 0)
        else:
            remaining_holes = np.any(np.all(result == 0, axis=2))
        
        if not remaining_holes:
            print(f"All holes filled after {i+1} iterations.")
            break
        
        result = inpaint_holes(result, method=method, radius=radius)
    
    return result


def create_mask_from_threshold(image, threshold=1):
    """
    Create a binary mask where pixels below threshold are considered holes.
    
    Args:
        image: Input image as numpy array
        threshold: Threshold value (pixels < threshold are holes)
        
    Returns:
        Binary mask (0 or 255)
    """
    if len(image.shape) == 2:
        mask = (image < threshold).astype(np.uint8) * 255
    else:
        # For color images, consider pixel as hole if all channels < threshold
        mask = (np.all(image < threshold, axis=2)).astype(np.uint8) * 255
    
    return mask


def visualize_mask(image, mask_output_path=None):
    """
    Create and optionally save a visualization of the mask.
    
    Args:
        image: Input image
        mask_output_path: Path to save mask visualization (optional)
        
    Returns:
        mask as numpy array
    """
    if len(image.shape) == 2:
        mask = (image == 0).astype(np.uint8) * 255
    else:
        mask = (np.all(image == 0, axis=2)).astype(np.uint8) * 255
    
    if mask_output_path:
        Image.fromarray(mask).save(mask_output_path)
        print(f"Mask saved to: {mask_output_path}")
    
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fill holes (black regions) in an image using inpainting algorithms.'
    )
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('output', type=str, help='Path to output image')
    parser.add_argument('--method', type=str, default='telea', 
                        choices=['telea', 'ns'],
                        help='Inpainting method: "telea" (fast, default) or "ns" (Navier-Stokes, slower but better quality)')
    parser.add_argument('--radius', type=int, default=3,
                        help='Radius of circular neighborhood for inpainting (default: 3)')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive multi-pass inpainting for larger holes')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Maximum iterations for adaptive inpainting (default: 5)')
    parser.add_argument('--save-mask', type=str, default=None,
                        help='Save the hole mask to this path')
    parser.add_argument('--visualize', action='store_true',
                        help='Create a side-by-side comparison visualization')
    
    args = parser.parse_args()
    
    try:
        # Load image
        print(f"Loading image from: {args.input}")
        img = Image.open(args.input)
        img_array = np.array(img)
        
        print(f"Image shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Count holes
        if len(img_array.shape) == 2:
            num_holes = np.sum(img_array == 0)
        else:
            num_holes = np.sum(np.all(img_array == 0, axis=2))
        
        print(f"Number of hole pixels: {num_holes}")
        
        # Save mask if requested
        if args.save_mask:
            visualize_mask(img_array, args.save_mask)
        
        # Apply inpainting
        if args.adaptive:
            print(f"Applying adaptive inpainting (method: {args.method})...")
            result = inpaint_holes_adaptive(img_array, method=args.method, 
                                           initial_radius=args.radius,
                                           max_iterations=args.iterations)
        else:
            print(f"Applying inpainting (method: {args.method}, radius: {args.radius})...")
            result = inpaint_holes(img_array, method=args.method, radius=args.radius)
        
        # Save result
        print(f"Saving result to: {args.output}")
        result_img = Image.fromarray(result.astype(np.uint8))
        result_img.save(args.output)
        print("Done!")
        
        # Create visualization if requested
        if args.visualize:
            import matplotlib.pyplot as plt
            
            vis_path = args.output.rsplit('.', 1)[0] + '_comparison.png'
            print(f"Creating visualization: {vis_path}")
            
            # Create mask for visualization
            if len(img_array.shape) == 2:
                mask = (img_array == 0).astype(np.uint8) * 255
            else:
                mask = (np.all(img_array == 0, axis=2)).astype(np.uint8) * 255
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original
            if len(img_array.shape) == 2:
                axes[0].imshow(img_array, cmap='gray')
            else:
                axes[0].imshow(img_array)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Hole Mask')
            axes[1].axis('off')
            
            # Result
            if len(result.shape) == 2:
                axes[2].imshow(result.astype(np.uint8), cmap='gray')
            else:
                axes[2].imshow(result.astype(np.uint8))
            axes[2].set_title(f'Inpainted ({args.method})')
            axes[2].axis('off')
            
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
