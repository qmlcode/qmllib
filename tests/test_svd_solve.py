import numpy as np
import pytest

from qmllib.solvers import svd_solve


def test_svd_solve_overdetermined():
    """Test SVD solve with overdetermined system (more equations than unknowns)"""
    # Create a simple overdetermined system: Ax = y where A is 3x2
    # This represents a least-squares problem
    A = np.array([[1.0, 2.0], 
                  [3.0, 4.0], 
                  [5.0, 6.0]])
    
    # True solution
    x_true = np.array([1.0, 2.0])
    
    # Generate y with exact values
    y = A @ x_true
    
    # Solve using SVD
    x = svd_solve(A, y, rcond=1e-10)
    
    # Should recover the true solution (within numerical precision)
    assert np.allclose(x, x_true), f"Expected {x_true}, got {x}"
    print(f"✅ Overdetermined system test passed: x = {x}")


def test_svd_solve_square():
    """Test SVD solve with square system"""
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    y = np.array([5.0, 7.0])
    
    x = svd_solve(A, y)
    
    # Check that Ax ≈ y
    residual = np.linalg.norm(A @ x - y)
    assert residual < 1e-10, f"Large residual: {residual}"
    print(f"✅ Square system test passed: x = {x}, residual = {residual}")


def test_svd_solve_preserves_input():
    """Test that svd_solve preserves the input matrix A"""
    A = np.array([[1.0, 2.0], 
                  [3.0, 4.0]])
    A_original = A.copy()
    y = np.array([1.0, 2.0])
    
    x = svd_solve(A, y)
    
    # A should not be modified
    assert np.allclose(A, A_original), "svd_solve modified the input matrix A"
    print(f"✅ Input preservation test passed")


def test_svd_solve_rcond():
    """Test SVD solve with different rcond values"""
    # Create a rank-deficient matrix
    A = np.array([[1.0, 2.0, 3.0],
                  [2.0, 4.0, 6.0],  # This row is linearly dependent
                  [4.0, 5.0, 6.0]])
    y = np.array([6.0, 12.0, 15.0])
    
    # With different rcond values
    x1 = svd_solve(A, y, rcond=1e-10)
    x2 = svd_solve(A, y, rcond=1e-5)
    
    # Both should solve the system (within tolerance), but may differ slightly
    residual1 = np.linalg.norm(A @ x1 - y)
    residual2 = np.linalg.norm(A @ x2 - y)
    
    assert residual1 < 1e-8, f"Large residual with rcond=1e-10: {residual1}"
    assert residual2 < 1e-8, f"Large residual with rcond=1e-5: {residual2}"
    print(f"✅ rcond test passed: residuals = {residual1:.2e}, {residual2:.2e}")


if __name__ == "__main__":
    test_svd_solve_overdetermined()
    test_svd_solve_square()
    test_svd_solve_preserves_input()
    test_svd_solve_rcond()
    print("\n✅ All fsvd_solve tests passed!")
