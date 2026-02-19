# Test Coverage Analysis: Unit Tests vs Integration Tests

## Overview

This analysis identifies which qmllib modules and functions are covered by **unit tests only** (run in CI) versus those that require **integration tests** for coverage.

## Test Statistics

- **Total tests**: 77
- **Unit tests** (fast, CI): 47 tests (~5 seconds)
- **Integration tests** (manual): 30 tests (~30+ seconds)

## Module Coverage by Test Type

### ✅ Modules Covered by Unit Tests (CI)

These modules have unit test coverage and will be tested on every PR:

#### Representations
- **`qmllib.representations`**
  - `generate_coulomb_matrix()` - tested in test_representations.py
  - `generate_atomic_coulomb_matrix()` - tested in test_representations.py  
  - `generate_bob()` - tested in test_representations.py
  - `generate_eigenvalue_coulomb_matrix()` - tested in test_representations.py

- **`qmllib.representations.slatm`**
  - `generate_slatm()` - tested in test_slatm.py (global and local variants)

- **`qmllib.representations.fchl`**
  - `generate_fchl18()` - tested in test_fchl_acsf.py
  - Atomic local representations - tested in test_fchl_atomic_local.py

#### Kernels
- **`qmllib.kernels.distance`**
  - `manhattan_distance()` - tested in test_distance.py, test_fdistance.py
  - `l2_distance()` - tested in test_distance.py, test_fdistance.py
  - `p_distance()` - tested in test_distance.py, test_fdistance.py

- **`qmllib.kernels`**
  - Kernel derivatives - tested in test_kernel_derivatives.py
  - Gradient kernels - tested in test_kernel_derivatives.py
  - Atomic local kernels - tested in test_fchl_atomic_local.py
  - GP kernels - tested in test_kernel_derivatives.py
  - GDML kernels - tested in test_kernel_derivatives.py

#### Solvers
- **`qmllib.solvers`**
  - `cho_solve()` - tested in test_solvers.py
  - `cho_invert()` - tested in test_solvers.py
  - `bkf_solve()` - tested in test_solvers.py
  - `bkf_invert()` - tested in test_solvers.py
  - `qrlq_solve()` - tested in test_solvers.py
  - `condition_number()` - tested in test_solvers.py
  - `svd_solve()` - tested in test_svd_solve.py

#### FCHL Components
- **`qmllib.representations.fchl`**
  - ACSF representations - tested in test_fchl_acsf.py
  - ACSF force kernels - tested in test_fchl_acsf_forces.py
  - Atomic local kernels (simple cases) - tested in test_fchl_atomic_local.py
  - Symmetric local kernels - tested in test_symmetric_local_kernel.py

### ⚠️ Modules Requiring Integration Tests

These modules are primarily tested through integration tests and have **limited or no unit test coverage**:

#### End-to-End ML Workflows
- **`qmllib.kernels.kernels`**
  - `laplacian_kernel()` - only tested in integration tests (energy KRR tests)
  - Various kernel types with full ML pipelines - integration tests only

- **`qmllib.kernels.gradient_kernels`**
  - Full gradient/hessian kernel workflows - integration tests (test_fchl_force.py)

- **`qmllib.representations.fchl.fchl_scalar_kernels`**
  - 15 different kernel variants (linear, polynomial, sigmoid, multiquadratic, bessel, matern, cauchy, etc.)
  - Only tested through integration tests in test_fchl_scalar.py

- **`qmllib.representations.fchl.fchl_force_kernels`**
  - Gaussian process derivative kernels - integration tests only
  - Normal equation derivatives - integration tests only
  - GDML derivatives - integration tests only

#### Specialized Components
- **`qmllib.representations.bob`**
  - Full BoB representation → kernel → training → prediction
  - Tested only in test_energy_krr_bob.py (integration)

- **`qmllib.utils.xyz_format`**
  - `read_xyz()` - used throughout tests but not directly unit tested
  - Implicitly tested through integration tests

- **`qmllib.utils.alchemy`**
  - Alchemical transformations - not covered by current unit tests
  - Tested in test_fchl_scalar.py::test_krr_fchl_alchemy (integration)

## Functions NOT Covered by Unit Tests

Based on the analysis, these specific areas lack unit test coverage:

### 1. High-Level Kernel Functions
```python
# qmllib.kernels.kernels
laplacian_kernel()  # Only in integration tests
gaussian_kernel()   # Only in integration tests
```

### 2. FCHL Scalar Kernels (15 variants)
```python
# qmllib.representations.fchl.fchl_scalar_kernels
# All tested only in test_fchl_scalar.py (integration)
- Linear kernel
- Polynomial kernel (degrees 2, 3)
- Sigmoid kernel
- Multiquadratic kernel
- Inverse multiquadratic kernel
- Bessel kernel
- L2 distance kernel
- Matern kernel
- Cauchy kernel
```

### 3. Force Field Components
```python
# qmllib.representations.fchl.fchl_force_kernels
# All tested only in test_fchl_force.py (integration)
get_gaussian_process_kernels()
get_local_gradient_kernels()
get_local_hessian_kernels()
get_local_symmetric_hessian_kernels()
```

### 4. Utility Functions
```python
# qmllib.utils.xyz_format
read_xyz()  # Used but not unit tested

# qmllib.utils.alchemy
QNum_distance()  # No unit tests
```

## Recommendations

To improve unit test coverage without slowing down CI:

### Priority 1: Add Unit Tests for Core Functions
1. **laplacian_kernel()** - Add simple unit test with known input/output
2. **gaussian_kernel()** - Add simple unit test with known input/output
3. **read_xyz()** - Add test for basic XYZ file parsing

### Priority 2: Add Lightweight FCHL Kernel Tests
Create fast unit tests for FCHL kernels using:
- Small, pre-computed representations (avoid generation overhead)
- Simple test cases with 2-3 molecules
- Known kernel properties (symmetry, positive definiteness)

Example:
```python
def test_fchl_linear_kernel_properties():
    """Test linear kernel basic properties without full ML workflow"""
    # Use small pre-computed representations
    rep1 = np.array([...])  # Small, fixed representation
    rep2 = np.array([...])
    
    K = get_linear_kernel(rep1, rep2)
    
    # Test symmetry
    assert np.allclose(K, K.T)
    # Test positive semi-definiteness
    assert np.all(np.linalg.eigvalsh(K) >= -1e-10)
```

### Priority 3: Integration Test Strategy
Keep integration tests for:
- Full ML pipelines (representation → kernel → solve → predict)
- Accuracy validation with real molecular datasets
- End-to-end regression testing
- Performance benchmarking

## Current CI Impact

**Unit tests (47 tests, ~5 seconds)** provide coverage for:
- ✅ Core distance calculations
- ✅ Solver functions (all variants)
- ✅ Basic representations (Coulomb, SLATM, BoB)
- ✅ Kernel derivatives
- ✅ FCHL ACSF workflows

**Integration tests (30 tests, manual)** are required for:
- ⚠️ Full ML prediction accuracy
- ⚠️ FCHL scalar kernel variants
- ⚠️ Force field predictions
- ⚠️ Energy prediction workflows

## Conclusion

The unit test suite provides good coverage of core computational primitives and will catch:
- Solver bugs
- Distance calculation errors
- Representation generation issues
- Basic kernel derivative problems

However, full ML workflow validation and specialized kernel variants require integration tests. This is acceptable because:
1. Integration tests are still run manually before releases
2. CI gets fast feedback on core functionality
3. Most bugs will be caught by unit tests
4. Integration tests validate accuracy, not correctness of primitives

The 85% reduction in CI time (from ~30s to ~5s) is worth the trade-off of moving comprehensive validation to manual/pre-release testing.
