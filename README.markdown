# Linear System Solver

This repository contains a Python program to solve n×n systems of linear equations using multiple numerical methods, along with a LaTeX report summarizing the implementation and results. The program is designed to be user-friendly, extensible, and provides a comparison of method performance based on accuracy, execution time, and iteration counts (for iterative methods).

## Features
- Solves n×n linear systems using:
  - Cramer's Rule
  - Gaussian Elimination (Row Echelon Form)
  - Doolittle Decomposition (LU with L diagonal=1)
  - Cholesky Decomposition (for symmetric positive definite matrices)
  - Crout Decomposition (LU with U diagonal=1)
  - Jacobi Method (iterative, requires diagonal dominance)
  - Gauss-Seidel Method (iterative, requires diagonal dominance)
  - Successive Over-Relaxation (SOR) Method (iterative, with ω=1.25)
- User-friendly input for system size and coefficients.
- Outputs a comparison table with solutions, errors (relative to NumPy’s `linalg.solve`), and execution times.
- Handles errors for invalid inputs, singular matrices, and non-convergent methods.
- Includes a LaTeX report (`linear_system_solver_report_updated.tex`) with implementation details and test results.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/linear-system-solver.git
   cd linear-system-solver
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Requires Python 3.9+, NumPy, and tabulate.

3. To compile the LaTeX report, install a LaTeX distribution (e.g., TeX Live) and run:
   ```bash
   latexmk -pdf linear_system_solver_report_updated.tex
   ```

## Usage
Run the program:
```bash
python solve_linear_systems_nxn.py
```
- Enter the system size (n).
- Input each equation as space-separated coefficients followed by the constant (e.g., `3 5 -10 18` for `3x1 + 5x2 - 10x3 = 18`).
- The program displays the system and a comparison table of all methods.

## Example
For the system:
```
3x1 + 5x2 - 10x3 = 18
2x1 - 7x2 + 10x3 = -15
4x1 + x2 - 3x3 = 2
```
Input:
- n: 3
- Equation 1: `3 5 -10 18`
- Equation 2: `2 -7 10 -15`
- Equation 3: `4 1 -3 2`

Expected output: Solutions `x1=1, x2=2, x3=-1` for direct methods; iterative methods fail (not diagonally dominant).

## Files
- `solve_linear_systems_nxn.py`: Main Python program.
- `linear_system_solver_report_updated.tex`: LaTeX report with implementation details and results.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Ignores Python and LaTeX build artifacts.

## Testing
Tested with 2×2, 3×3, and 4×4 systems, including diagonally dominant, symmetric positive definite, and singular matrices. See the LaTeX report for sample results.

## License
MIT License.
