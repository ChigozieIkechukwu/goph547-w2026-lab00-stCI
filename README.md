**GOPH 547 - Lab Assignment #0**
* **Semester: W2026**

* **Instructor: B. Karchewski**

* **Author: Chigozie Ikechukwu**

* **Repository: https://github.com/ChigozieIkechukwu/goph547-w2026-lab00-stCI**

**1. Introduction**
* The purpose of this lab was to familiarize myself with the essential software tools and development practices for scientific computing in Python. This involved setting up my development environment on Windows, using Git and GitHub for version control, and creating a structured Python package that utilizes NumPy for numerical operations and Matplotlib for data visualization.

**2. Installation Instructions (Windows PowerShell)**
* I have designed these instructions to allow my package to be downloaded and installed into a virtual environment on a Windows system.

**Step 1: Clone my Repository**
* I opened my Windows PowerShell and navigate to my desired directory, then I used the GitHub CLI to clone the repository:

#powershell

gh repo clone ChigozieIkechukwu/goph547-w2026-lab00-stCI
cd goph547-w2026-lab00-stCI

**Step 2: Create and Activate my Virtual Environment**
* I used virtualenv to create an isolated environment named .venv:

#powershell:
virtualenv .venv

#To activate the environment in PowerShell, I ran:
#powershell
.\.venv\Scripts\Activate.ps1

#(Note: I did not encounter an execution policy error, hence I did not use Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser to allow the script to run.)

**Step 3: Install the Package in Development Mode**

#I installed my local package and its dependencies (NumPy and Matplotlib) using the following command:

#powershell

pip install -e ./

**3. Script Description: driver.py**
* My primary entry point script is located at examples/driver.py. This script performs the following tasks:

* **NumPy Exercises**: It creates arrays of ones and NaNs, generates a column vector of odd numbers, and performs matrix operations including element-wise multiplication, dot products, and cross products.

* **Image Processing**: I loaded the rock_canyon.jpg image, converted it to grayscale, and used NumPy slicing to create a cropped image of the central pinnacle.

* **Visualization**: I generated a subplot figure that analyzes the mean R, G, and B channel values against the x and y coordinates, including a black line for the overall RGB mean.

**3.1 Detailed Line-by-Line Code Explanation**

#python

import numpy as np

* #Purpose: Imports the NumPy library and assigns it the alias np. NumPy provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

#python

import matplotlib.pyplot as plt

* #Purpose: Imports the pyplot module from Matplotlib and assigns it the alias plt. This module provides a MATLAB-like plotting interface for creating static, animated, and interactive visualizations in Python.

#python

from PIL import Image

* #Purpose: Imports the Image module from the Python Imaging Library (PIL), also known as Pillow. This library adds image processing capabilities, such as opening, manipulating, and saving various image file formats.

#python

from goph547lab00.arrays import square_ones

#Purpose: Imports a custom function called square_ones from the arrays module within my local package goph547lab00. This function is defined in src/goph547lab00/arrays.py.

#python

def main():
#Purpose: Defines the main function of the script. This function encapsulates all the code that will be executed when the script runs, following a structured programming approach.

#python

    print("=" * 60)
    print("GOPH 547 Lab 0B - Numpy and Matplotlib Exercises")
    print("=" * 60)

#Purpose: Prints a formatted header to the console, creating a visual separation and clearly indicating the start of the program output.

#python

    # Part 1: Numpy Array Exercises
    print("\n" + "=" * 60)
    print("PART 1: NUMPY ARRAY EXERCISES")
    print("=" * 60)

#Purpose: Prints a section header for Part 1 of the exercises, which focuses on NumPy array operations.

#python
    # 1. Create an array of ones with 3 rows and 5 columns
    ones_array = np.ones((3, 5))
    print("1. Array of ones (3x5):")
    print(ones_array)
    print(f"   Shape: {ones_array.shape}")

#Purpose: Creates a 3x5 array filled with the value 1.0 using np.ones(), prints the array, and displays its shape (dimensions).

#python
    # 2. Produce an array of NaN with 6 rows and 3 columns
    nan_array = np.full((6, 3), np.nan)
    print("\n2. Array of NaN (6x3):")
    print(nan_array)
    print(f"   Shape: {nan_array.shape}")

#Purpose: Creates a 6x3 array filled with NaN (Not a Number) values using np.full(). NaN is a special floating-point value used to represent missing or undefined data.

#python
    # 3. Create a column vector of odd numbers between 44 and 75
    odd_numbers = np.arange(45, 76, 2)  # Start at 45, end at 75, step 2
    odd_vector = odd_numbers.reshape(-1, 1)  # Make it a column vector
    print("\n3. Column vector of odd numbers between 44 and 75:")
    print(odd_vector)
    print(f"   Shape: {odd_vector.shape}")
    print(f"   Values: {odd_numbers.tolist()}")

#Purpose:

* np.arange(45, 76, 2) generates a 1D array of numbers starting at 45, up to (but not including) 76, with a step of 2, resulting in odd numbers.

* reshape(-1, 1) transforms the 1D array into a 2D column vector (an Nx1 array). The -1 tells NumPy to automatically calculate the size of that dimension.

#python

    # 4. Find the sum of the vector produced in #3
    odd_sum = np.sum(odd_numbers)
    print(f"\n4. Sum of odd numbers vector: {odd_sum}")

#Purpose: Calculates the sum of all elements in the odd_numbers array using np.sum() and prints the result.

#python

    # 5. Produce the array A
    A = np.array([[5, 7, 2],
                  [1, -2, 3],
                  [4, 4, 4]])
    print("\n5. Array A:")
    print(A)
    print(f"   Shape: {A.shape}")

#Purpose: Creates a 3x3 array A from a nested list using np.array() and prints it along with its shape.

#python

    # 6. Using a single command, produce the identity matrix B
    B = np.eye(3)
    print("\n6. Identity matrix B (3x3):")
    print(B)
#Purpose: Creates a 3x3 identity matrix B using np.eye(3). An identity matrix has 1's on the main diagonal and 0's elsewhere.

#python
  
    # 7. Perform element-wise multiplication of A and B
    elementwise_mult = np.multiply(A, B)  # or A * B
    print("\n7. Element-wise multiplication of A and B:")
    print(elementwise_mult)

#Purpose: Performs element-wise (Hadamard) multiplication between arrays A and B using np.multiply(). This multiplies corresponding elements: C[i,j] = A[i,j] * B[i,j].

#python
    # 8. Calculate the dot product (matrix multiplication) of A and B
    dot_product = np.dot(A, B)
    print("\n8. Dot product (matrix multiplication) of A and B:")
    print(dot_product)
    print("   Note: A * I = A, so this should equal A")

#Purpose: Performs standard matrix multiplication (dot product) between A and B using np.dot(). Since B is an identity matrix, the result should equal A.

#python
    # 9. Calculate the cross product of A and B
    # For 3x3 matrices, cross product is typically for vectors
    # We'll compute cross product of first rows as example
    cross_product = np.cross(A[0], B[0])
    print("\n9. Cross product of first rows of A and B:")
    print(f"   Cross(A[0], B[0]) = {cross_product}")

#Purpose: Computes the cross product (vector product) of the first rows of A and B using np.cross(). The cross product is defined for 3D vectors and produces a vector perpendicular to both input vectors.

#python
    # Part 2: Matplotlib Visualization Exercises
    print("\n" + "=" * 60)
    print("PART 2: MATPLOTLIB VISUALIZATION EXERCISES")
    print("=" * 60)
#Purpose: Prints a section header for Part 2, which focuses on image processing and visualization with Matplotlib.

#python
    try:
#Purpose: Begins a try-except block to handle potential errors (like missing files) gracefully during the image processing section.

#python
        # 10. Load the image rock_canyon.jpg
        print("\n10. Loading image 'rock_canyon.jpg'...")
        img = Image.open('examples/rock_canyon.jpg')
        img_array = np.asarray(img)

#Purpose:

* Opens the image file rock_canyon.jpg from the examples directory using PIL's Image.open().

* Converts the image to a NumPy array using np.asarray(), allowing pixel data to be manipulated numerically.

#python
        # 11. Plot the image and get shape
        print("11. Original image:")
        print(f"    Shape: {img_array.shape}")
        print(f"    Data type: {img_array.dtype}")

#Purpose: Prints the shape (height, width, channels) and data type of the image array. For an RGB image, the shape would typically be (height, width, 3).

#python
        # Display original image
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(img_array)
        plt.title('Original Image (rock_canyon.jpg)')
        plt.axis('off')
#Purpose:

* plt.figure(figsize=(10, 8)): Creates a new figure with specified dimensions (10x8 inches).

* plt.subplot(2, 2, 1): Creates a 2x2 grid of subplots and selects the first position (top-left).

* plt.imshow(img_array): Displays the image array in the current subplot.

* plt.title(): Adds a title to the subplot.

* plt.axis('off'): Turns off axis labels and ticks for cleaner image display.

#python
        # 12. Convert to grayscale and get shape
        print("\n12. Grayscale image:")
        gray_img = Image.open('examples/rock_canyon.jpg').convert('L')
        gray_array = np.asarray(gray_img)
        print(f"    Shape: {gray_array.shape}")
        print(f"    Data type: {gray_array.dtype}")

#Purpose:

* Opens the image again and converts it to grayscale using the .convert('L') method.

* Converts the grayscale image to a NumPy array and prints its properties. Grayscale images typically have shape (height, width) with no channel dimension.

#python
        # Display grayscale image
        plt.subplot(2, 2, 2)
        plt.imshow(gray_array, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')

#Purpose: Displays the grayscale image in the second subplot (top-right), using the 'gray' colormap to ensure proper grayscale rendering.

#python
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

#Purpose:

* Extracts height and width from the grayscale array shape.

* Calculates bounding box coordinates for cropping a region of interest (the central pinnacle).

* Uses NumPy array slicing gray_array[y_start:y_end, x_start:x_end] to extract the specified rectangular region.

* Prints the shape of the cropped image.

#python
        # Display small image
        plt.subplot(2, 2, 3)
        plt.imshow(small_gray_image, cmap='gray')
        plt.title('Small Pinnacle Region')
        plt.axis('off')

#Purpose: Displays the cropped region in the third subplot (bottom-left).

#python
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

#Purpose: Calculates various mean values for RGB analysis:

* mean_r, mean_g, mean_b: Mean intensity along the y-axis (vertical) for each color channel, resulting in a 1D array for each channel representing the average color value across each column.

* mean_rgb: Overall mean RGB value along the y-axis, then averaged across channels.

* mean_r_y, mean_g_y, mean_b_y: Mean intensity along the x-axis (horizontal) for each channel.

* mean_rgb_y: Overall mean RGB value along the x-axis.

#python
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

#Purpose: Creates the fourth subplot (bottom-right) showing horizontal color profiles:

* np.arange(len(mean_r)) creates x-coordinate values.

* Four line plots show the mean red, green, blue, and overall RGB values across the x-axis.

* Formatting includes axis labels, title, legend, and semi-transparent grid.

#python
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the figure
        output_file = 'rock_canyon_RGB_summary.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"    Saved RGB analysis plot to: {output_file}")

#Purpose:

* plt.tight_layout(): Automatically adjusts subplot parameters to prevent label overlapping.

* plt.savefig(): Saves the complete figure as a PNG file with specified DPI (dots per inch) and tight bounding box.

* Prints confirmation message with the output filename.

#python
        # Show all plots
        plt.show()
#Purpose: Displays the complete figure with all four subplots in a window (blocks execution until the window is closed).

#python
        print("\n" + "=" * 60)
        print("EXERCISES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files in examples/ directory:")
        print("  - driver.py (updated)")
        print(f"  - {output_file} (RGB analysis plot)")
        print("\nCheck your directory for the output image!")

#Purpose: Prints a success message and summary of generated files upon successful completion.

#python
    except FileNotFoundError:
        print("\nERROR: 'rock_canyon.jpg' not found in examples/ directory!")
        print("Please download the image from D2L and place it in the examples/ folder.")
        print("Then run this script again.")
#Purpose: Catches and handles the specific error when the required image file is not found, providing helpful instructions to the user.

#python
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
#Purpose: Catches any other unexpected exceptions, prints the error message, and uses traceback.print_exc() to display the full traceback for debugging.

#python
if __name__ == '__main__':
    main()
#Purpose: Standard Python idiom that checks if this script is being run directly (not imported as a module). If so, it calls the main() function to execute the program.

**4. How to Run my Code**
#To execute the script and see my results, I ensure my virtual environment is active and run the following from my project's root directory:

#powershell
python examples/driver.py
**Expected Output**
* Terminal: I saw the printed results of my NumPy calculations and the shapes of my processed image arrays.

* Visualization: A new file named rock_canyon_RGB_summary.png was generated and saved in my examples/ directory.

**5. My Repository Structure**
My project is organized as follows:

**text**
goph547-w2026-lab00-stCI/
├── .git/                # Git version control
├── .venv/               # My Windows virtual environment
├── src/
│   └── goph547lab00/    # My source code package
│       ├── __init__.py
│       └── arrays.py    # Custom NumPy functions
├── examples/
│   ├── driver.py        # My main driver script
│   ├── rock_canyon.jpg  # Input image from D2L
│   └── rock_canyon_RGB_summary.png
├── pyproject.toml       # My package configuration
└── README.md            # This report
