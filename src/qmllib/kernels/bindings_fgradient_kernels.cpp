#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>
#include <iostream>

namespace py = pybind11;

// Declare C ABI Fortran functions
extern "C" {
    void fglobal_kernel(const double* x1, const double* x2,
                       const int* q1, const int* q2,
                       const int* n1, const int* n2,
                       int nm1, int nm2, double sigma, double* kernel,
                       int max_atoms1, int max_atoms2, int rep_size);
    
    void flocal_kernels(const double* x1, const double* x2,
                       const int* q1, const int* q2,
                       const int* n1, const int* n2,
                       int nm1, int nm2, const double* sigmas, int nsigmas,
                       double* kernel, int max_atoms1, int max_atoms2, int rep_size);
    
    void fsymmetric_local_kernels(const double* x1, const int* q1,
                                 const int* n1, int nm1,
                                 const double* sigmas, int nsigmas,
                                 double* kernel, int max_atoms1, int rep_size);
    
    void flocal_kernel(const double* x1, const double* x2,
                      const int* q1, const int* q2,
                      const int* n1, const int* n2,
                      int nm1, int nm2, double sigma, double* kernel,
                      int max_atoms1, int max_atoms2, int rep_size);
    
    void fsymmetric_local_kernel(const double* x1, const int* q1,
                                const int* n1, int nm1, double sigma,
                                double* kernel, int max_atoms1, int rep_size);
    
    void fatomic_local_kernel(const double* x1, const double* x2,
                             const int* q1, const int* q2,
                             const int* n1, const int* n2,
                             int nm1, int nm2, int na1, double sigma,
                             double* kernel, int max_atoms1, int max_atoms2, int rep_size);
    
    void fatomic_local_gradient_kernel(const double* x1, const double* x2,
                                      const double* dx2,
                                      const int* q1, const int* q2,
                                      const int* n1, const int* n2,
                                      int nm1, int nm2, int na1, int naq2,
                                      double sigma, double* kernel,
                                      int max_atoms1, int max_atoms2, int rep_size);
    
    void flocal_gradient_kernel(const double* x1, const double* x2,
                               const double* dx2,
                               const int* q1, const int* q2,
                               const int* n1, const int* n2,
                               int nm1, int nm2, int naq2, double sigma,
                               double* kernel, int max_atoms1, int max_atoms2, int rep_size);
    
    void fgdml_kernel(const double* x1, const double* x2,
                     const double* dx1, const double* dx2,
                     const int* q1, const int* q2,
                     const int* n1, const int* n2,
                     int nm1, int nm2, int na1, int na2, double sigma,
                     double* kernel, int max_atoms1, int max_atoms2, int rep_size);
    
    void fsymmetric_gdml_kernel(const double* x1, const double* dx1,
                               const int* q1, const int* n1,
                               int nm1, int na1, double sigma, double* kernel,
                               int max_atoms1, int rep_size);
    
    void fgaussian_process_kernel(const double* x1, const double* x2,
                                 const double* dx1, const double* dx2,
                                 const int* q1, const int* q2,
                                 const int* n1, const int* n2,
                                 int nm1, int nm2, int na1, int na2,
                                 double sigma, double* kernel,
                                 int max_atoms1, int max_atoms2, int rep_size);
    
    void fsymmetric_gaussian_process_kernel(const double* x1, const double* dx1,
                                           const int* q1, const int* n1,
                                           int nm1, int na1, double sigma,
                                           double* kernel, int max_atoms1, int rep_size);
}

// Wrapper for fglobal_kernel
py::array_t<double> global_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1,
    py::array_t<int, py::array::f_style | py::array::forcecast> q2,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2,
    int nm1,
    int nm2,
    double sigma
) {
    auto bufX1 = x1.request();
    auto bufX2 = x2.request();
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    
    if (bufX1.ndim != 3 || bufX2.ndim != 3) {
        throw std::runtime_error("X1 and X2 must be 3D arrays");
    }
    
    if (bufQ1.ndim != 2 || bufQ2.ndim != 2) {
        throw std::runtime_error("Q1 and Q2 must be 2D arrays");
    }
    
    int actual_nm1 = static_cast<int>(bufX1.shape[0]);
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    int actual_nm2 = static_cast<int>(bufX2.shape[0]);
    int max_atoms2 = static_cast<int>(bufX2.shape[1]);
    
    if (actual_nm1 != nm1 || actual_nm2 != nm2) {
        throw std::runtime_error("Molecule count mismatch");
    }
    
    // Create output array (nm2, nm1) - Fortran column-major
    std::vector<ssize_t> shape = {nm2, nm1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nm2};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    fglobal_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufX2.ptr),
        static_cast<const int*>(bufQ1.ptr),
        static_cast<const int*>(bufQ2.ptr),
        static_cast<const int*>(n1.request().ptr),
        static_cast<const int*>(n2.request().ptr),
        nm1, nm2, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, max_atoms2, rep_size
    );
    
    return kernel;
}

// Wrapper for flocal_kernels
py::array_t<double> local_kernels_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1,
    py::array_t<int, py::array::f_style | py::array::forcecast> q2,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2,
    int nm1,
    int nm2,
    py::array_t<double, py::array::f_style | py::array::forcecast> sigmas,
    int nsigmas
) {
    auto bufX1 = x1.request();
    auto bufX2 = x2.request();
    
    int actual_nm1 = static_cast<int>(bufX1.shape[0]);
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    int actual_nm2 = static_cast<int>(bufX2.shape[0]);
    int max_atoms2 = static_cast<int>(bufX2.shape[1]);
    
    // Create output array (nsigmas, nm2, nm1) - Fortran column-major
    std::vector<ssize_t> shape = {nsigmas, nm2, nm1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm2};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    flocal_kernels(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufX2.ptr),
        static_cast<const int*>(q1.request().ptr),
        static_cast<const int*>(q2.request().ptr),
        static_cast<const int*>(n1.request().ptr),
        static_cast<const int*>(n2.request().ptr),
        nm1, nm2,
        static_cast<const double*>(sigmas.request().ptr),
        nsigmas,
        static_cast<double*>(bufK.ptr),
        max_atoms1, max_atoms2, rep_size
    );
    
    return kernel;
}

// Wrapper for fsymmetric_local_kernels
py::array_t<double> symmetric_local_kernels_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1,
    int nm1,
    py::array_t<double, py::array::f_style | py::array::forcecast> sigmas,
    int nsigmas
) {
    auto bufX1 = x1.request();
    
    int actual_nm1 = static_cast<int>(bufX1.shape[0]);
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    
    // Create output array (nsigmas, nm1, nm1) - Fortran column-major
    std::vector<ssize_t> shape = {nsigmas, nm1, nm1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    fsymmetric_local_kernels(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const int*>(q1.request().ptr),
        static_cast<const int*>(n1.request().ptr),
        nm1,
        static_cast<const double*>(sigmas.request().ptr),
        nsigmas,
        static_cast<double*>(bufK.ptr),
        max_atoms1, rep_size
    );
    
    return kernel;
}

// Wrapper for flocal_kernel
py::array_t<double> local_kernel_wrapper(
    py::array_t<double> x1_in,
    py::array_t<double> x2_in,
    py::array_t<int> q1_in,
    py::array_t<int> q2_in,
    py::array_t<int> n1_in,
    py::array_t<int> n2_in,
    int nm1,
    int nm2,
    double sigma
) {
    // Explicitly convert to F-contiguous if needed and keep alive
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto x2 = py::array_t<double, py::array::f_style | py::array::forcecast>(x2_in);
    auto q1 = py::array_t<int, py::array::f_style | py::array::forcecast>(q1_in);
    auto q2 = py::array_t<int, py::array::f_style | py::array::forcecast>(q2_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto n2 = py::array_t<int, py::array::f_style | py::array::forcecast>(n2_in);
    
    auto bufX1 = x1.request();
    auto bufX2 = x2.request();
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    auto bufN1 = n1.request();
    auto bufN2 = n2.request();
    
    // Extract dimensions from X arrays (they're already padded correctly)
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int max_atoms2 = static_cast<int>(bufX2.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    
    // Create output array (nm2, nm1) - Fortran column-major
    std::vector<ssize_t> shape = {nm2, nm1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nm2};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    flocal_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufX2.ptr),
        static_cast<const int*>(bufQ1.ptr),
        static_cast<const int*>(bufQ2.ptr),
        static_cast<const int*>(bufN1.ptr),
        static_cast<const int*>(bufN2.ptr),
        nm1, nm2, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, max_atoms2, rep_size
    );
    
    return kernel;
}

// Wrapper for fsymmetric_local_kernel
py::array_t<double> symmetric_local_kernel_wrapper(
    py::array_t<double> x1_in,
    py::array_t<int> q1_in,
    py::array_t<int> n1_in,
    int nm1,
    double sigma
) {
    // Explicitly convert to F-contiguous if needed and keep alive
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto q1 = py::array_t<int, py::array::f_style | py::array::forcecast>(q1_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    
    auto bufX1 = x1.request();
    auto bufQ1 = q1.request();
    auto bufN1 = n1.request();
    
    // Extract dimensions from X1
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    
    // Create output array (nm1, nm1) - Fortran column-major
    std::vector<ssize_t> shape = {nm1, nm1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nm1};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    fsymmetric_local_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const int*>(bufQ1.ptr),
        static_cast<const int*>(bufN1.ptr),
        nm1, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, rep_size
    );
    
    return kernel;
}

// Wrapper for fatomic_local_kernel
py::array_t<double> atomic_local_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2_in,
    int nm1,
    int nm2,
    int na1,
    double sigma
) {
    // Ensure converted arrays stay alive
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto x2 = py::array_t<double, py::array::f_style | py::array::forcecast>(x2_in);
    auto q1 = py::array_t<int, py::array::f_style | py::array::forcecast>(q1_in);
    auto q2 = py::array_t<int, py::array::f_style | py::array::forcecast>(q2_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto n2 = py::array_t<int, py::array::f_style | py::array::forcecast>(n2_in);
    
    auto bufX1 = x1.request();
    auto bufX2 = x2.request();
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    auto bufN1 = n1.request();
    auto bufN2 = n2.request();
    
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    int max_atoms2 = static_cast<int>(bufX2.shape[1]);
    
    // Create output array (nm2, na1) - Fortran column-major
    // Note: Fortran expects kernel(nm2, na1)
    std::vector<ssize_t> shape = {nm2, na1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nm2};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    fatomic_local_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufX2.ptr),
        static_cast<const int*>(bufQ1.ptr),
        static_cast<const int*>(bufQ2.ptr),
        static_cast<const int*>(bufN1.ptr),
        static_cast<const int*>(bufN2.ptr),
        nm1, nm2, na1, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, max_atoms2, rep_size
    );
    
    return kernel;
}

// Wrapper for fatomic_local_gradient_kernel
py::array_t<double> atomic_local_gradient_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> dx2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2_in,
    int nm1,
    int nm2,
    int na1,
    int naq2,
    double sigma
) {
    // Ensure converted arrays stay alive
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto x2 = py::array_t<double, py::array::f_style | py::array::forcecast>(x2_in);
    auto dx2 = py::array_t<double, py::array::f_style | py::array::forcecast>(dx2_in);
    auto q1 = py::array_t<int, py::array::f_style | py::array::forcecast>(q1_in);
    auto q2 = py::array_t<int, py::array::f_style | py::array::forcecast>(q2_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto n2 = py::array_t<int, py::array::f_style | py::array::forcecast>(n2_in);
    
    auto bufX1 = x1.request();
    auto bufX2 = x2.request();
    auto bufDX2 = dx2.request();
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    auto bufN1 = n1.request();
    auto bufN2 = n2.request();
    
    if (bufDX2.ndim != 5) {
        throw std::runtime_error("DX2 must be a 5D array");
    }
    
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    int max_atoms2 = static_cast<int>(bufX2.shape[1]);
    
    // Create output array (naq2, na1) - Fortran column-major
    // Note: Fortran expects kernel(naq2, na1)
    std::vector<ssize_t> shape = {naq2, na1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * naq2};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    fatomic_local_gradient_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufX2.ptr),
        static_cast<const double*>(bufDX2.ptr),
        static_cast<const int*>(bufQ1.ptr),
        static_cast<const int*>(bufQ2.ptr),
        static_cast<const int*>(bufN1.ptr),
        static_cast<const int*>(bufN2.ptr),
        nm1, nm2, na1, naq2, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, max_atoms2, rep_size
    );
    
    return kernel;
}

// Wrapper for flocal_gradient_kernel
py::array_t<double> local_gradient_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2,
    py::array_t<double, py::array::f_style | py::array::forcecast> dx2,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1,
    py::array_t<int, py::array::f_style | py::array::forcecast> q2,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2,
    int nm1,
    int nm2,
    int naq2,
    double sigma
) {
    auto bufX1 = x1.request();
    auto bufX2 = x2.request();
    auto bufDX2 = dx2.request();
    
    if (bufDX2.ndim != 5) {
        throw std::runtime_error("DX2 must be a 5D array");
    }
    
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    int max_atoms2 = static_cast<int>(bufX2.shape[1]);
    
    // Create output array (naq2, nm1) - Fortran column-major
    std::vector<ssize_t> shape = {naq2, nm1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * naq2};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    flocal_gradient_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufX2.ptr),
        static_cast<const double*>(bufDX2.ptr),
        static_cast<const int*>(q1.request().ptr),
        static_cast<const int*>(q2.request().ptr),
        static_cast<const int*>(n1.request().ptr),
        static_cast<const int*>(n2.request().ptr),
        nm1, nm2, naq2, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, max_atoms2, rep_size
    );
    
    return kernel;
}

// Wrapper for fgdml_kernel
py::array_t<double> gdml_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> dx1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> dx2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2_in,
    int nm1,
    int nm2,
    int na1,
    int na2,
    double sigma
) {
    // Ensure converted arrays stay alive
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto x2 = py::array_t<double, py::array::f_style | py::array::forcecast>(x2_in);
    auto dx1 = py::array_t<double, py::array::f_style | py::array::forcecast>(dx1_in);
    auto dx2 = py::array_t<double, py::array::f_style | py::array::forcecast>(dx2_in);
    auto q1 = py::array_t<int, py::array::f_style | py::array::forcecast>(q1_in);
    auto q2 = py::array_t<int, py::array::f_style | py::array::forcecast>(q2_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto n2 = py::array_t<int, py::array::f_style | py::array::forcecast>(n2_in);
    
    auto bufX1 = x1.request();
    auto bufX2 = x2.request();
    auto bufDX1 = dx1.request();
    auto bufDX2 = dx2.request();
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    auto bufN1 = n1.request();
    auto bufN2 = n2.request();
    
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    int max_atoms2 = static_cast<int>(bufX2.shape[1]);
    
    // Create output array (na2*3, na1*3) - Fortran column-major
    // Note: Fortran expects kernel(na2*3, na1*3)
    int rows = na2 * 3;
    int cols = na1 * 3;
    std::vector<ssize_t> shape = {rows, cols};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * rows};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    fgdml_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufX2.ptr),
        static_cast<const double*>(bufDX1.ptr),
        static_cast<const double*>(bufDX2.ptr),
        static_cast<const int*>(bufQ1.ptr),
        static_cast<const int*>(bufQ2.ptr),
        static_cast<const int*>(bufN1.ptr),
        static_cast<const int*>(bufN2.ptr),
        nm1, nm2, na1, na2, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, max_atoms2, rep_size
    );
    
    return kernel;
}

// Wrapper for fsymmetric_gdml_kernel
py::array_t<double> symmetric_gdml_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> dx1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    int nm1,
    int na1,
    double sigma
) {
    // Ensure converted arrays stay alive
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto dx1 = py::array_t<double, py::array::f_style | py::array::forcecast>(dx1_in);
    auto q1 = py::array_t<int, py::array::f_style | py::array::forcecast>(q1_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    
    auto bufX1 = x1.request();
    auto bufDX1 = dx1.request();
    auto bufQ1 = q1.request();
    auto bufN1 = n1.request();
    
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    
    // Create output array (na1*3, na1*3) - Fortran column-major
    // Note: Fortran expects kernel(na1*3, na1*3)
    int size = na1 * 3;
    std::vector<ssize_t> shape = {size, size};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * size};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    fsymmetric_gdml_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufDX1.ptr),
        static_cast<const int*>(bufQ1.ptr),
        static_cast<const int*>(bufN1.ptr),
        nm1, na1, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, rep_size
    );
    
    return kernel;
}

// Wrapper for fgaussian_process_kernel
py::array_t<double> gaussian_process_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> dx1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> dx2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2_in,
    int nm1,
    int nm2,
    int na1,
    int na2,
    double sigma
) {
    // Ensure converted arrays stay alive
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto x2 = py::array_t<double, py::array::f_style | py::array::forcecast>(x2_in);
    auto dx1 = py::array_t<double, py::array::f_style | py::array::forcecast>(dx1_in);
    auto dx2 = py::array_t<double, py::array::f_style | py::array::forcecast>(dx2_in);
    auto q1 = py::array_t<int, py::array::f_style | py::array::forcecast>(q1_in);
    auto q2 = py::array_t<int, py::array::f_style | py::array::forcecast>(q2_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto n2 = py::array_t<int, py::array::f_style | py::array::forcecast>(n2_in);
    
    auto bufX1 = x1.request();
    auto bufX2 = x2.request();
    auto bufDX1 = dx1.request();
    auto bufDX2 = dx2.request();
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    auto bufN1 = n1.request();
    auto bufN2 = n2.request();
    
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    int max_atoms2 = static_cast<int>(bufX2.shape[1]);
    
    // Create output array (na2*3+nm2, na1*3+nm1) - Fortran column-major
    // Note: Fortran expects kernel(na2*3+nm2, na1*3+nm1)
    int rows = na2 * 3 + nm2;
    int cols = na1 * 3 + nm1;
    std::vector<ssize_t> shape = {rows, cols};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * rows};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    fgaussian_process_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufX2.ptr),
        static_cast<const double*>(bufDX1.ptr),
        static_cast<const double*>(bufDX2.ptr),
        static_cast<const int*>(bufQ1.ptr),
        static_cast<const int*>(bufQ2.ptr),
        static_cast<const int*>(bufN1.ptr),
        static_cast<const int*>(bufN2.ptr),
        nm1, nm2, na1, na2, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, max_atoms2, rep_size
    );
    
    return kernel;
}

// Wrapper for fsymmetric_gaussian_process_kernel
py::array_t<double> symmetric_gaussian_process_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> dx1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> q1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    int nm1,
    int na1,
    double sigma
) {
    // Ensure converted arrays stay alive
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto dx1 = py::array_t<double, py::array::f_style | py::array::forcecast>(dx1_in);
    auto q1 = py::array_t<int, py::array::f_style | py::array::forcecast>(q1_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    
    auto bufX1 = x1.request();
    auto bufDX1 = dx1.request();
    auto bufQ1 = q1.request();
    auto bufN1 = n1.request();
    
    int max_atoms1 = static_cast<int>(bufX1.shape[1]);
    int rep_size = static_cast<int>(bufX1.shape[2]);
    
    // Create output array (na1*3+nm1, na1*3+nm1) - Fortran column-major
    // Note: Fortran expects kernel(na1*3+nm1, na1*3+nm1)
    int size = na1 * 3 + nm1;
    std::vector<ssize_t> shape = {size, size};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * size};
    auto kernel = py::array_t<double>(shape, strides);
    auto bufK = kernel.request();
    
    fsymmetric_gaussian_process_kernel(
        static_cast<const double*>(bufX1.ptr),
        static_cast<const double*>(bufDX1.ptr),
        static_cast<const int*>(bufQ1.ptr),
        static_cast<const int*>(bufN1.ptr),
        nm1, na1, sigma,
        static_cast<double*>(bufK.ptr),
        max_atoms1, rep_size
    );
    
    return kernel;
}

PYBIND11_MODULE(_fgradient_kernels, m) {
    m.doc() = "QMLlib gradient kernel functions";

    m.def("fglobal_kernel", &global_kernel_wrapper,
        py::arg("x1"), py::arg("x2"), py::arg("q1"), py::arg("q2"),
        py::arg("n1"), py::arg("n2"), py::arg("nm1"), py::arg("nm2"),
        py::arg("sigma"),
        "Global kernel");

    m.def("flocal_kernels", &local_kernels_wrapper,
        py::arg("x1"), py::arg("x2"), py::arg("q1"), py::arg("q2"),
        py::arg("n1"), py::arg("n2"), py::arg("nm1"), py::arg("nm2"),
        py::arg("sigmas"), py::arg("nsigmas"),
        "Local kernels with multiple sigmas");

    m.def("fsymmetric_local_kernels", &symmetric_local_kernels_wrapper,
        py::arg("x1"), py::arg("q1"), py::arg("n1"), py::arg("nm1"),
        py::arg("sigmas"), py::arg("nsigmas"),
        "Symmetric local kernels");

    m.def("flocal_kernel", &local_kernel_wrapper,
        py::arg("x1"), py::arg("x2"), py::arg("q1"), py::arg("q2"),
        py::arg("n1"), py::arg("n2"), py::arg("nm1"), py::arg("nm2"),
        py::arg("sigma"),
        "Local kernel");

    m.def("fsymmetric_local_kernel", &symmetric_local_kernel_wrapper,
        py::arg("x1"), py::arg("q1"), py::arg("n1"), py::arg("nm1"),
        py::arg("sigma"),
        "Symmetric local kernel");

    m.def("fatomic_local_kernel", &atomic_local_kernel_wrapper,
        py::arg("x1"), py::arg("x2"), py::arg("q1"), py::arg("q2"),
        py::arg("n1"), py::arg("n2"), py::arg("nm1"), py::arg("nm2"),
        py::arg("na1"), py::arg("sigma"),
        "Atomic local kernel");

    m.def("fatomic_local_gradient_kernel", &atomic_local_gradient_kernel_wrapper,
        py::arg("x1"), py::arg("x2"), py::arg("dx2"),
        py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
        py::arg("nm1"), py::arg("nm2"), py::arg("na1"), py::arg("naq2"),
        py::arg("sigma"),
        "Atomic local gradient kernel");

    m.def("flocal_gradient_kernel", &local_gradient_kernel_wrapper,
        py::arg("x1"), py::arg("x2"), py::arg("dx2"),
        py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
        py::arg("nm1"), py::arg("nm2"), py::arg("naq2"), py::arg("sigma"),
        "Local gradient kernel");

    m.def("fgdml_kernel", &gdml_kernel_wrapper,
        py::arg("x1"), py::arg("x2"), py::arg("dx1"), py::arg("dx2"),
        py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
        py::arg("nm1"), py::arg("nm2"), py::arg("na1"), py::arg("na2"),
        py::arg("sigma"),
        "GDML kernel");

    m.def("fsymmetric_gdml_kernel", &symmetric_gdml_kernel_wrapper,
        py::arg("x1"), py::arg("dx1"), py::arg("q1"), py::arg("n1"),
        py::arg("nm1"), py::arg("na1"), py::arg("sigma"),
        "Symmetric GDML kernel");

    m.def("fgaussian_process_kernel", &gaussian_process_kernel_wrapper,
        py::arg("x1"), py::arg("x2"), py::arg("dx1"), py::arg("dx2"),
        py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
        py::arg("nm1"), py::arg("nm2"), py::arg("na1"), py::arg("na2"),
        py::arg("sigma"),
        "Gaussian process kernel");

    m.def("fsymmetric_gaussian_process_kernel", &symmetric_gaussian_process_kernel_wrapper,
        py::arg("x1"), py::arg("dx1"), py::arg("q1"), py::arg("n1"),
        py::arg("nm1"), py::arg("na1"), py::arg("sigma"),
        "Symmetric Gaussian process kernel");
}
