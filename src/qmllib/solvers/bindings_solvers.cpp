#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>

namespace py = pybind11;

// Declare C ABI Fortran functions
extern "C" {
    void c_fcho_solve(double* A, const double* y, double* x, int n);
    void c_fcho_invert(double* A, int n);
    void c_fbkf_invert(double* A, int n);
    void c_fbkf_solve(double* A, const double* y, double* x, int n);
}

// Wrapper for fcho_solve
// Python signature: fcho_solve(A, y, x) where x is output array
void fcho_solve_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> A,
    py::array_t<double, py::array::f_style | py::array::forcecast> y,
    py::array_t<double, py::array::f_style | py::array::forcecast> x
) {
    auto bufA = A.request();
    auto bufY = y.request();
    auto bufX = x.request();

    if (bufA.ndim != 2 || bufA.shape[0] != bufA.shape[1]) {
        throw std::runtime_error("A must be a square 2D array");
    }
    if (bufY.ndim != 1) {
        throw std::runtime_error("y must be a 1D array");
    }
    if (bufX.ndim != 1) {
        throw std::runtime_error("x must be a 1D array");
    }

    int n = static_cast<int>(bufA.shape[0]);
    
    if (bufY.shape[0] != n || bufX.shape[0] != n) {
        throw std::runtime_error("Array dimensions must match");
    }

    // Make a copy of A since it will be modified by LAPACK
    py::array_t<double, py::array::f_style> A_copy({n, n});
    auto bufA_copy = A_copy.request();
    std::memcpy(bufA_copy.ptr, bufA.ptr, n * n * sizeof(double));

    double* A_ptr = static_cast<double*>(bufA_copy.ptr);
    const double* y_ptr = static_cast<const double*>(bufY.ptr);
    double* x_ptr = static_cast<double*>(bufX.ptr);

    c_fcho_solve(A_ptr, y_ptr, x_ptr, n);
}

// Wrapper for fcho_invert
// Returns the inverted matrix
py::array_t<double> fcho_invert_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> A
) {
    auto bufA = A.request();

    if (bufA.ndim != 2 || bufA.shape[0] != bufA.shape[1]) {
        throw std::runtime_error("A must be a square 2D array");
    }

    int n = static_cast<int>(bufA.shape[0]);

    // Make a copy since the function modifies the array
    py::array_t<double, py::array::f_style> A_inv({n, n});
    auto bufA_inv = A_inv.request();
    std::memcpy(bufA_inv.ptr, bufA.ptr, n * n * sizeof(double));

    double* A_ptr = static_cast<double*>(bufA_inv.ptr);
    
    c_fcho_invert(A_ptr, n);

    // Copy lower triangle to upper triangle
    // In Fortran column-major: A[i,j] accessed as data[i + j*n]
    double* data = static_cast<double*>(bufA_inv.ptr);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            data[i + j * n] = data[j + i * n];  // A[i,j] = A[j,i]
        }
    }

    return A_inv;
}

// Wrapper for fbkf_invert
// Returns the inverted matrix
py::array_t<double> fbkf_invert_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> A
) {
    auto bufA = A.request();

    if (bufA.ndim != 2 || bufA.shape[0] != bufA.shape[1]) {
        throw std::runtime_error("A must be a square 2D array");
    }

    int n = static_cast<int>(bufA.shape[0]);

    // Make a copy since the function modifies the array
    py::array_t<double, py::array::f_style> A_inv({n, n});
    auto bufA_inv = A_inv.request();
    std::memcpy(bufA_inv.ptr, bufA.ptr, n * n * sizeof(double));

    double* A_ptr = static_cast<double*>(bufA_inv.ptr);
    
    c_fbkf_invert(A_ptr, n);

    // Copy lower triangle to upper triangle
    // In Fortran column-major: A[i,j] accessed as data[i + j*n]
    double* data = static_cast<double*>(bufA_inv.ptr);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            data[i + j * n] = data[j + i * n];  // A[i,j] = A[j,i]
        }
    }

    return A_inv;
}

// Wrapper for fbkf_solve
// Python signature: fbkf_solve(A, y, x) where x is output array
void fbkf_solve_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> A,
    py::array_t<double, py::array::f_style | py::array::forcecast> y,
    py::array_t<double, py::array::f_style | py::array::forcecast> x
) {
    auto bufA = A.request();
    auto bufY = y.request();
    auto bufX = x.request();

    if (bufA.ndim != 2 || bufA.shape[0] != bufA.shape[1]) {
        throw std::runtime_error("A must be a square 2D array");
    }
    if (bufY.ndim != 1) {
        throw std::runtime_error("y must be a 1D array");
    }
    if (bufX.ndim != 1) {
        throw std::runtime_error("x must be a 1D array");
    }

    int n = static_cast<int>(bufA.shape[0]);
    
    if (bufY.shape[0] != n || bufX.shape[0] != n) {
        throw std::runtime_error("Array dimensions must match");
    }

    // Make a copy of A since it will be modified by LAPACK
    py::array_t<double, py::array::f_style> A_copy({n, n});
    auto bufA_copy = A_copy.request();
    std::memcpy(bufA_copy.ptr, bufA.ptr, n * n * sizeof(double));

    double* A_ptr = static_cast<double*>(bufA_copy.ptr);
    const double* y_ptr = static_cast<const double*>(bufY.ptr);
    double* x_ptr = static_cast<double*>(bufX.ptr);

    c_fbkf_solve(A_ptr, y_ptr, x_ptr, n);
}

PYBIND11_MODULE(_solvers, m) {
    m.doc() = "qmllib: Fortran solver routines with pybind11 bindings";
    
    m.def("fcho_solve", &fcho_solve_wrapper,
          py::arg("A"), py::arg("y"), py::arg("x"),
          "Solve Ax=y using Cholesky decomposition (LAPACK dpotrf/dpotrs)");
    
    m.def("fcho_invert", &fcho_invert_wrapper,
          py::arg("A"),
          "Invert positive definite matrix using Cholesky decomposition (LAPACK dpotrf/dpotri)");
    
    m.def("fbkf_invert", &fbkf_invert_wrapper,
          py::arg("A"),
          "Invert symmetric matrix using Bunch-Kaufman decomposition (LAPACK dsytrf/dsytri)");
    
    m.def("fbkf_solve", &fbkf_solve_wrapper,
          py::arg("A"), py::arg("y"), py::arg("x"),
          "Solve Ax=y using Bunch-Kaufman decomposition (LAPACK dsytrf/dsytrs)");
}
