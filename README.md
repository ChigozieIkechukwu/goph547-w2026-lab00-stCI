**GOPH 547 - Lab Assignment #0**
* Semester: W2026
* Instructor: B. Karchewski
* Author: Chigozie Ikechukwu
* Repository: https://github.com/ChigozieIkechukwu/goph547-w2026-lab00-stCI.

**1. Introduction**
The purpose of this lab was to familiarize myself with the essential software tools and development practices for scientific computing in Python. This involved setting up my development environment on Windows, using Git and GitHub for version control, and creating a structured Python package that utilizes NumPy for numerical operations and Matplotlib for data visualization.

**2. Installation Instructions (Windows PowerShell):**
I have designed these instructions to allow my package to be downloaded and installed into a virtual environment on a Windows system.

* Step 1: Clone my Repository
I open my Windows PowerShell and navigate to my desired directory, then I use the GitHub CLI to clone the repository:
gh repo clone ChigozieIkechukwu/goph547-w2026-lab00-stCI
cd goph547-w2026-lab00-stCI

* Step 2: Create and Activate my Virtual Environment
I use virtualenv to create an isolated environment named .venv:
virtualenv .venv
To activate the environment in PowerShell, I run:
.\.venv\Scripts\Activate.ps1
(Note: I did not encounter an execution policy error, hence I did not use Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser to allow the script to run.)

* Step 3: Install the Package in Development Mode
I install my local package and its dependencies (NumPy and Matplotlib) using the following command:
pip install -e ./


**3. Script Description: driver.py:**
My primary entry point script is located at examples/driver.py. This script performs the following tasks:
* NumPy Exercises: It creates arrays of ones and NaNs, generates a column vector of odd numbers, and performs matrix operations including element-wise multiplication, dot products, and cross products.
* Image Processing: I loaded the rock_canyon.jpg image, converted it to grayscale, and used NumPy slicing to create a cropped image of the central pinnacle.
* Visualization: I generated a subplot figure that analyzes the mean R, G, and B channel values against the x and y coordinates, including a black line for the overall RGB mean.

**4. How to Run my Code:**
To execute the script and see my results, I ensure my virtual environment is active and run the following from my project's root directory:
python examples/driver.py
Expected Output
* Terminal: I saw the printed results of my NumPy calculations and the shapes of my processed image arrays.
* Visualization: A new file named rock_canyon_RGB_summary.png was generated and saved in my examples/ directory.

**5. My Repository Structure:**
My project is organized as follows:
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
