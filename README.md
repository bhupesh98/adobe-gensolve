# Adobe GenSolve Hackathon Submission
## Team Name: Snorlox
## Team Members:
- Debjyoti Ray
- Bhupesh Dewangan
- Akshay Waghmare

Welcome to my submission for the Adobe GenSolve Hackathon! This project tackles the problem statement outlined in `Curvetopia 1.pdf`. It involves three primary tasks: Regularizing Curves, Exploring Symmetry in Curves, and Completing Incomplete Curves. These tasks are executed across various files and notebooks within this repository.

## Problem Statement

The objective of this project is to identify, regularize, and beautify curves in 2D Euclidean space. We start by focusing on closed curves and progressively work with more complex shapes. This project also covers symmetry and curve completion techniques.

### Tasks Breakdown

1. **Regularize Curves**
   - **Objective:** Identify regular shapes (e.g., straight lines, circles, ellipses, rectangles, and polygons) from a given set of curves.
   - **Approach:** Algorithms are designed to detect and regularize these shapes from hand-drawn or scanned images.
   - **Implementation:** The task is implemented in the notebook `adobe.ipynb`.

2. **Exploring Symmetry in Curves**
   - **Objective:** Identify symmetry in closed shapes, starting with reflection symmetries. This includes detecting lines of symmetry and fitting identical Bezier curves on points that are symmetric.
   - **Implementation:** This task is also covered in the notebook `adobe.ipynb`.

3. **Completing Incomplete Curves**
   - **Objective:** Complete curves in 2D space that have gaps due to occlusion or other factors, ensuring smoothness and regularity in the completion.
   - **Implementation:** This task is documented and explained in the file `Adobe-GenSolve-docs.pdf`.

## Repository Structure

The repository is structured as follows:

- **`adobe.ipynb`**: Contains the implementation for Regularizing Curves and Exploring Symmetry in Curves.
- **`Adobe-GenSolve-docs.pdf`**: Provides detailed documentation and explanations for the problem statement and the third task, Completing Incomplete Curves.
- **`Curvetopia 1.pdf`**: Outlines the problem statement and gives an overview of the tasks and objectives.

## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/bhupesh98/adobe-gensolve.git
   ```
2. Navigate to the project directory:
   ```bash
   cd adobe-gensolve
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the `adobe.ipynb` notebook to explore the code for the first two tasks.
5. Refer to the `Adobe-GenSolve-docs.pdf` for detailed explanations on Completing Incomplete Curves.

## Example Usage

> **Website Link:** https://adobe-gensolve-tensor.streamlit.app/

### Regularize Curves
Run the cells in `adobe.ipynb` to process images and extract regular shapes from hand-drawn sketches.

### Symmetry in Curves
The notebook also includes code to detect symmetry in closed shapes.

### Complete Incomplete Curves
Refer to `Adobe-GenSolve-docs.pdf` for the methodology and code snippets used for this task.

## Results

The expected results for the tasks are as follows:

1. **Regularized Curves**: Curves identified and regularized based on the geometric shapes they resemble.
2. **Symmetry in Curves**: Symmetry detected in closed curves, with corresponding Bezier curves fitted.
3. **Completed Curves**: Incomplete curves filled and smoothed to restore the original shape.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

Special thanks to Adobe for organizing the GenSolve Hackathon and providing a platform for this exciting challenge.
