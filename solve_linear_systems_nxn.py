import numpy as np
from tabulate import tabulate
import time


# Helper function to check if matrix is diagonally dominant
def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag <= off_diag:
            return False
    return True


# Helper function to check if matrix is symmetric positive definite
def is_symmetric_positive_definite(A):
    if not np.allclose(A, A.T):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


# Cramer's Rule
def cramer_rule(A, b):
    n = len(b)
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-10:
        return None, "Singular matrix"
    x = np.zeros(n)
    for i in range(n):
        A_temp = A.copy()
        A_temp[:, i] = b
        x[i] = np.linalg.det(A_temp) / det_A
    return x, ""


# Standard Gaussian Elimination (Row Echelon Form)
def gaussian_elimination(A, b):
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)
    # Forward elimination to row echelon form
    for k in range(n - 1):
        if abs(A[k, k]) < 1e-10:
            return None, "Singular matrix or requires pivoting"
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    # Back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-10:
            return None, "Singular matrix"
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x, ""


# Doolittle Decomposition (LU with L diagonal=1)
def doolittle_decomposition(A, b):
    A = A.copy().astype(float)
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i + 1, n):
            if abs(U[i, i]) < 1e-10:
                return None, "Singular matrix or requires pivoting"
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]
    # Solve Ly = b (forward substitution)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    # Solve Ux = y (back substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x, ""


# Cholesky Decomposition (for SPD matrices)
def cholesky_decomposition(A, b):
    if not is_symmetric_positive_definite(A):
        return None, "Matrix not symmetric positive definite"
    A = A.copy().astype(float)
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            if j == i:
                L[i, j] = np.sqrt(A[i, i] - np.dot(L[i, :j], L[i, :j]))
            else:
                L[i, j] = (A[i, j] - np.dot(L[i, :j], L[j, :j])) / L[j, j]
    # Solve Ly = b (forward substitution)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    # Solve L^T x = y (back substitution)
    LT = L.T
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i + 1:], x[i + 1:])) / LT[i, i]
    return x, ""


# Crout Decomposition (LU with U diagonal=1)
def crout_decomposition(A, b):
    A = A.copy().astype(float)
    n = len(A)
    L = np.zeros((n, n))
    U = np.eye(n)
    for j in range(n):
        for i in range(j, n):
            L[i, j] = A[i, j] - np.dot(L[i, :j], U[:j, j])
        for i in range(j + 1, n):
            if abs(L[j, j]) < 1e-10:
                return None, "Singular matrix or requires pivoting"
            U[j, i] = (A[j, i] - np.dot(L[j, :j], U[:j, i])) / L[j, j]
    # Solve Ly = b (forward substitution)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    # Solve Ux = y (back substitution)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i] - np.dot(U[i, i + 1:], x[i + 1:])
    return x, ""


# Jacobi Method
def jacobi_method(A, b, tolerance=1e-6, max_iterations=100):
    if not is_diagonally_dominant(A):
        return None, "Matrix not diagonally dominant"
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    iterations = 0
    while iterations < max_iterations:
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        if np.max(np.abs(x_new - x)) < tolerance:
            break
        x = x_new.copy()
        iterations += 1
    return x, f"Iterations: {iterations}"


# Gauss-Seidel Method
def gauss_seidel_method(A, b, tolerance=1e-6, max_iterations=100):
    if not is_diagonally_dominant(A):
        return None, "Matrix not diagonally dominant"
    n = len(b)
    x = np.zeros(n)
    iterations = 0
    while iterations < max_iterations:
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x_old[i + 1:])) / A[i, i]
        if np.max(np.abs(x - x_old)) < tolerance:
            break
        iterations += 1
    return x, f"Iterations: {iterations}"


# SOR Method
def sor_method(A, b, omega=1.25, tolerance=1e-6, max_iterations=100):
    if not is_diagonally_dominant(A):
        return None, "Matrix not diagonally dominant"
    n = len(b)
    x = np.zeros(n)
    iterations = 0
    while iterations < max_iterations:
        x_old = x.copy()
        for i in range(n):
            x[i] = (1 - omega) * x_old[i] + omega * (
                    b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x_old[i + 1:])
            ) / A[i, i]
        if np.max(np.abs(x - x_old)) < tolerance:
            break
        iterations += 1
    return x, f"Iterations: {iterations}"


# User input for n×n system
def get_user_input():
    while True:
        try:
            n = int(input("Enter the number of equations (n for n×n system): "))
            if n <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    print(f"Enter the coefficients for an {n}×{n} system of linear equations:")
    print(f"Format: a1 a2 ... an b (e.g., for 3x1 + 5x2 - 10x3 = 18, enter '3 5 -10 18')")
    A = np.zeros((n, n))
    b = np.zeros(n)
    for i in range(n):
        while True:
            eq = input(f"Equation {i + 1}: ")
            coeffs = list(map(float, eq.split()))
            if len(coeffs) != n + 1:
                print(f"Expected {n} coefficients and 1 constant (total {n + 1} numbers). Try again.")
                continue
            A[i, :] = coeffs[:n]
            b[i] = coeffs[n]
            break
    return A, b


# Compare all methods
def compare_methods(A, b):
    true_solution = np.linalg.solve(A, b)  # Reference solution
    methods = [
        ("Cramer's Rule", cramer_rule),
        ("Gaussian Elimination", gaussian_elimination),
        ("Doolittle Decomposition", doolittle_decomposition),
        ("Cholesky Decomposition", cholesky_decomposition),
        ("Crout Decomposition", crout_decomposition),
        ("Jacobi", jacobi_method),
        ("Gauss-Seidel", gauss_seidel_method),
        ("SOR", sor_method)
    ]
    results = []

    for name, method in methods:
        start_time = time.time()
        solution, info = method(A, b)
        time_taken = time.time() - start_time
        error = np.nan if solution is None else np.max(np.abs(solution - true_solution))
        results.append([
            name,
            info if info else "-",
            solution if solution is not None else "Failed",
            error,
            time_taken
        ])

    # Print comparison table
    headers = ["Method", "Info", f"Solution ({', '.join(f'x{i + 1}' for i in range(len(b)))})", "Max Error", "Time (s)"]
    formatted_results = []
    for row in results:
        if isinstance(row[2], np.ndarray):
            sol = f"({', '.join(f'{x:.6f}' for x in row[2])})"
        else:
            sol = row[2]
        formatted_results.append([row[0], row[1], sol, row[3], row[4]])
    print("\nComparison of Linear System Solvers:")
    print(tabulate(formatted_results, headers=headers, tablefmt="grid", floatfmt=".6f"))


# Main program
def main():
    print("Solve an n×n system of linear equations")
    A, b = get_user_input()
    print("\nSystem of equations:")
    for i in range(len(b)):
        coeffs = [f"{A[i, j]:.1f}x{j + 1}" for j in range(len(b)) if A[i, j] != 0]
        print(" + ".join(coeffs).replace("+ -", "- ") + f" = {b[i]:.1f}")
    compare_methods(A, b)


if __name__ == "__main__":
    main()
