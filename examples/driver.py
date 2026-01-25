import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from goph547lab00.arrays import square_ones

def main():
    print("=" * 60)
    print("GOPH 547 Lab 0B - Numpy and Matplotlib Exercises")
    print("=" * 60)
    
    # Part 1: Numpy Array Exercises
    print("\n" + "=" * 60)
    print("PART 1: NUMPY ARRAY EXERCISES")
    print("=" * 60)
    
    # 1. Create an array of ones with 3 rows and 5 columns
    ones_array = np.ones((3, 5))
    print("1. Array of ones (3x5):")
    print(ones_array)
    print(f"   Shape: {ones_array.shape}")
    
    # 2. Produce an array of NaN with 6 rows and 3 columns
    nan_array = np.full((6, 3), np.nan)
    print("\n2. Array of NaN (6x3):")
    print(nan_array)
    print(f"   Shape: {nan_array.shape}")
    
    # 3. Create a column vector of odd numbers between 44 and 75
    odd_numbers = np.arange(45, 76, 2)  # Start at 45, end at 75, step 2
    odd_vector = odd_numbers.reshape(-1, 1)  # Make it a column vector
    print("\n3. Column vector of odd numbers between 44 and 75:")
    print(odd_vector)
    print(f"   Shape: {odd_vector.shape}")
    print(f"   Values: {odd_numbers.tolist()}")
    
    # 4. Find the sum of the vector produced in #3
    odd_sum = np.sum(odd_numbers)
    print(f"\n4. Sum of odd numbers vector: {odd_sum}")
    
    # 5. Produce the array A
    A = np.array([[5, 7, 2],
                  [1, -2, 3],
                  [4, 4, 4]])
    print("\n5. Array A:")
    print(A)
    print(f"   Shape: {A.shape}")
    
    # 6. Using a single command, produce the identity matrix B
    B = np.eye(3)
    print("\n6. Identity matrix B (3x3):")
    print(B)
    
    # 7. Perform element-wise multiplication of A and B
    elementwise_mult = np.multiply(A, B)  # or A * B
    print("\n7. Element-wise multiplication of A and B:")
    print(elementwise_mult)
    
    # 8. Calculate the dot product (matrix multiplication) of A and B
    dot_product = np.dot(A, B)
    print("\n8. Dot product (matrix multiplication) of A and B:")
    print(dot_product)
    print("   Note: A * I = A, so this should equal A")
    
    # 9. Calculate the cross product of A and B
    # For 3x3 matrices, cross product is typically for vectors
    # We'll compute cross product of first rows as example
    cross_product = np.cross(A[0], B[0])
    print("\n9. Cross product of first rows of A and B:")
    print(f"   Cross(A[0], B[0]) = {cross_product}")
    
    # Part 2: Matplotlib Visualization Exercises
    print("\n" + "=" * 60)
    print("PART 2: MATPLOTLIB VISUALIZATION EXERCISES")
    print("=" * 60)
    
    try:
        # 10. Load the image rock_canyon.jpg
        print("\n10. Loading image 'rock_canyon.jpg'...")
        img = Image.open('examples/rock_canyon.jpg')
        img_array = np.asarray(img)
        
        # 11. Plot the image and get shape
        print("11. Original image:")
        print(f"    Shape: {img_array.shape}")
        print(f"    Data type: {img_array.dtype}")
        
        # Display original image
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(img_array)
        plt.title('Original Image (rock_canyon.jpg)')
        plt.axis('off')
        
        # 12. Convert to grayscale and get shape
        print("\n12. Grayscale image:")
        gray_img = Image.open('examples/rock_canyon.jpg').convert('L')
        gray_array = np.asarray(gray_img)
        print(f"    Shape: {gray_array.shape}")
        print(f"    Data type: {gray_array.dtype}")
        
        # Display grayscale image
        plt.subplot(2, 2, 2)
        plt.imshow(gray_array, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')
        
        # 13. Create smaller image of pinnacle/pillar
        print("\n13. Creating smaller image of pinnacle/pillar...")
        # Assuming the image shape is (height, width, channels)
        height, width = gray_array.shape
        
        # Define region for pinnacle (adjust these based on your image)
        # These are example coordinates - you'll need to adjust them
        y_start = height // 4      # Start 1/4 from top
        y_end = 3 * height // 4    # End 3/4 down
        x_start = width // 8       # Start 1/8 from left
        x_end = width // 3         # End 1/3 from left
        
        small_gray_image = gray_array[y_start:y_end, x_start:x_end]
        print(f"    Small image shape: {small_gray_image.shape}")
        
        # Display small image
        plt.subplot(2, 2, 3)
        plt.imshow(small_gray_image, cmap='gray')
        plt.title('Small Pinnacle Region')
        plt.axis('off')
        
        # 14-16. Create RGB analysis subplot
        print("\n14-16. Creating RGB analysis plots...")
        
        # Calculate means
        mean_r = np.mean(img_array[:, :, 0], axis=0)  # Mean along y for R
        mean_g = np.mean(img_array[:, :, 1], axis=0)  # Mean along y for G
        mean_b = np.mean(img_array[:, :, 2], axis=0)  # Mean along y for B
        mean_rgb = np.mean(img_array, axis=0).mean(axis=1)  # Mean of RGB
        
        mean_r_y = np.mean(img_array[:, :, 0], axis=1)  # Mean along x for R
        mean_g_y = np.mean(img_array[:, :, 1], axis=1)  # Mean along x for G
        mean_b_y = np.mean(img_array[:, :, 2], axis=1)  # Mean along x for B
        mean_rgb_y = np.mean(img_array, axis=1).mean(axis=1)  # Mean of RGB
        
        # Create subplots
        plt.subplot(2, 2, 4)
        
        # Plot 1: x-coordinate vs color values
        plt.subplot(2, 2, 4)
        x_coords = np.arange(len(mean_r))
        plt.plot(x_coords, mean_r, 'r-', label='Red channel', alpha=0.7)
        plt.plot(x_coords, mean_g, 'g-', label='Green channel', alpha=0.7)
        plt.plot(x_coords, mean_b, 'b-', label='Blue channel', alpha=0.7)
        plt.plot(x_coords, mean_rgb, 'k-', linewidth=2, label='Mean RGB', alpha=0.9)
        plt.xlabel('X-coordinate (pixels)')
        plt.ylabel('Color Value (mean along y)')
        plt.title('Horizontal Color Profile')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the figure
        output_file = 'rock_canyon_RGB_summary.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    Saved RGB analysis plot to: {output_file}")
        
        # Show all plots
        plt.show()
        
        print("\n" + "=" * 60)
        print("EXERCISES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files in examples/ directory:")
        print("  - driver.py (updated)")
        print(f"  - {output_file} (RGB analysis plot)")
        print("\nCheck your directory for the output image!")
        
    except FileNotFoundError:
        print("\nERROR: 'rock_canyon.jpg' not found in examples/ directory!")
        print("Please download the image from D2L and place it in the examples/ folder.")
        print("Then run this script again.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
