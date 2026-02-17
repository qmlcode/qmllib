#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>

namespace py = pybind11;

// Declare C ABI Fortran functions
extern "C" {
    void fkpca(const double* k, int n, int centering, double* kpca);
    void fwasserstein_kernel(const double* a, int rep_size, int na,
                            const double* b, int nb,
                            double* k, double sigma, int p, int q);
    
    // Basic kernel functions (2D arrays)
    void fgaussian_kernel(const double* a, int na, const double* b, int nb,
                         double* k, double sigma, int rep_size);
    void fgaussian_kernel_symmetric(const double* x, int n, double* k,
                                   double sigma, int rep_size);
    void flaplacian_kernel(const double* a, int na, const double* b, int nb,
                          double* k, double sigma, int rep_size);
    void flaplacian_kernel_symmetric(const double* x, int n, double* k,
                                    double sigma, int rep_size);
    void flinear_kernel(const double* a, int na, const double* b, int nb,
                       double* k, int rep_size);
    void fmatern_kernel_l2(const double* a, int na, const double* b, int nb,
                          double* k, double sigma, int order, int rep_size);
    void fsargan_kernel(const double* a, int na, const double* b, int nb,
                       double* k, double sigma, const double* gammas,
                       int ng, int rep_size);
    
    // Local kernel functions (2D arrays with molecule counts)
    void fget_local_kernels_gaussian(int rep_size, const double* q1, const double* q2,
                                    const int* n1, const int* n2,
                                    const double* sigmas,
                                    int nm1, int nm2, int nsigmas,
                                    int nq1, int nq2, double* kernels);
    void fget_local_kernels_laplacian(int rep_size, const double* q1, const double* q2,
                                     const int* n1, const int* n2,
                                     const double* sigmas,
                                     int nm1, int nm2, int nsigmas,
                                     int nq1, int nq2, double* kernels);
    
    // Vector kernel functions (3D arrays)
    void fget_vector_kernels_gaussian(const double* q1, const double* q2,
                                     const int* n1, const int* n2,
                                     const double* sigmas,
                                     int nm1, int nm2, int nsigmas,
                                     int rep_size, int max_atoms, double* kernels);
    void fget_vector_kernels_laplacian(const double* q1, const double* q2,
                                      const int* n1, const int* n2,
                                      const double* sigmas,
                                      int nm1, int nm2, int nsigmas,
                                      int rep_size, int max_atoms, double* kernels);
    void fget_vector_kernels_gaussian_symmetric(const double* q, const int* n,
                                               const double* sigmas,
                                               int nm, int nsigmas,
                                               int rep_size, int max_atoms,
                                               double* kernels);
    void fget_vector_kernels_laplacian_symmetric(const double* q, const int* n,
                                                const double* sigmas,
                                                int nm, int nsigmas,
                                                int rep_size, int max_atoms,
                                                double* kernels);
}

// Wrapper for fkpca
py::array_t<double> kpca_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> k,
    int n,
    bool centering
) {
    auto bufK = k.request();
    
    if (bufK.ndim != 2) {
        throw std::runtime_error("K must be a 2D array");
    }
    
    int size = static_cast<int>(bufK.shape[0]);
    
    if (bufK.shape[0] != bufK.shape[1]) {
        throw std::runtime_error("K must be a square matrix");
    }
    
    if (size != n) {
        throw std::runtime_error("K dimensions must match n parameter");
    }
    
    // Create Fortran-style (column-major) output array
    std::vector<ssize_t> shape = {n, n};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * n};
    auto kpca = py::array_t<double>(shape, strides);
    auto bufKPCA = kpca.request();
    
    // Call Fortran function (0=false, 1=true for centering)
    fkpca(
        static_cast<const double*>(bufK.ptr),
        n,
        centering ? 1 : 0,
        static_cast<double*>(bufKPCA.ptr)
    );
    
    return kpca;
}

// Wrapper for fwasserstein_kernel
py::array_t<double> wasserstein_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> a,
    int na,
    py::array_t<double, py::array::f_style | py::array::forcecast> b,
    int nb,
    double sigma,
    int p,
    int q
) {
    auto bufA = a.request();
    auto bufB = b.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    int rep_size = static_cast<int>(bufA.shape[0]);
    
    if (bufA.shape[0] != bufB.shape[0]) {
        throw std::runtime_error("A and B must have same representation size");
    }
    
    if (bufA.shape[1] != na) {
        throw std::runtime_error("A second dimension must match na");
    }
    
    if (bufB.shape[1] != nb) {
        throw std::runtime_error("B second dimension must match nb");
    }
    
    // Create Fortran-style (column-major) output array
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto k = py::array_t<double>(shape, strides);
    auto bufK = k.request();
    
    // Initialize to zero
    std::memset(bufK.ptr, 0, na * nb * sizeof(double));
    
    fwasserstein_kernel(
        static_cast<const double*>(bufA.ptr),
        rep_size, na,
        static_cast<const double*>(bufB.ptr),
        nb,
        static_cast<double*>(bufK.ptr),
        sigma, p, q
    );
    
    return k;
}

// Wrapper for fgaussian_kernel
py::array_t<double> gaussian_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> a,
    py::array_t<double, py::array::f_style | py::array::forcecast> b,
    double sigma
) {
    auto bufA = a.request();
    auto bufB = b.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    int rep_size = static_cast<int>(bufA.shape[0]);
    int na = static_cast<int>(bufA.shape[1]);
    int nb = static_cast<int>(bufB.shape[1]);
    
    if (bufA.shape[0] != bufB.shape[0]) {
        throw std::runtime_error("A and B must have same first dimension");
    }
    
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto k = py::array_t<double>(shape, strides);
    auto bufK = k.request();
    
    fgaussian_kernel(
        static_cast<const double*>(bufA.ptr), na,
        static_cast<const double*>(bufB.ptr), nb,
        static_cast<double*>(bufK.ptr), sigma, rep_size
    );
    
    return k;
}

// Wrapper for fgaussian_kernel_symmetric
py::array_t<double> gaussian_kernel_symmetric_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x,
    double sigma
) {
    auto bufX = x.request();
    
    if (bufX.ndim != 2) {
        throw std::runtime_error("X must be a 2D array");
    }
    
    int rep_size = static_cast<int>(bufX.shape[0]);
    int n = static_cast<int>(bufX.shape[1]);
    
    std::vector<ssize_t> shape = {n, n};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * n};
    auto k = py::array_t<double>(shape, strides);
    auto bufK = k.request();
    
    fgaussian_kernel_symmetric(
        static_cast<const double*>(bufX.ptr), n,
        static_cast<double*>(bufK.ptr), sigma, rep_size
    );
    
    return k;
}

// Wrapper for flaplacian_kernel
py::array_t<double> laplacian_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> a,
    py::array_t<double, py::array::f_style | py::array::forcecast> b,
    double sigma
) {
    auto bufA = a.request();
    auto bufB = b.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    int rep_size = static_cast<int>(bufA.shape[0]);
    int na = static_cast<int>(bufA.shape[1]);
    int nb = static_cast<int>(bufB.shape[1]);
    
    if (bufA.shape[0] != bufB.shape[0]) {
        throw std::runtime_error("A and B must have same first dimension");
    }
    
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto k = py::array_t<double>(shape, strides);
    auto bufK = k.request();
    
    flaplacian_kernel(
        static_cast<const double*>(bufA.ptr), na,
        static_cast<const double*>(bufB.ptr), nb,
        static_cast<double*>(bufK.ptr), sigma, rep_size
    );
    
    return k;
}

// Wrapper for flaplacian_kernel_symmetric
py::array_t<double> laplacian_kernel_symmetric_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> x,
    double sigma
) {
    auto bufX = x.request();
    
    if (bufX.ndim != 2) {
        throw std::runtime_error("X must be a 2D array");
    }
    
    int rep_size = static_cast<int>(bufX.shape[0]);
    int n = static_cast<int>(bufX.shape[1]);
    
    std::vector<ssize_t> shape = {n, n};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * n};
    auto k = py::array_t<double>(shape, strides);
    auto bufK = k.request();
    
    flaplacian_kernel_symmetric(
        static_cast<const double*>(bufX.ptr), n,
        static_cast<double*>(bufK.ptr), sigma, rep_size
    );
    
    return k;
}

// Wrapper for flinear_kernel
py::array_t<double> linear_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> a,
    py::array_t<double, py::array::f_style | py::array::forcecast> b
) {
    auto bufA = a.request();
    auto bufB = b.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    int rep_size = static_cast<int>(bufA.shape[0]);
    int na = static_cast<int>(bufA.shape[1]);
    int nb = static_cast<int>(bufB.shape[1]);
    
    if (bufA.shape[0] != bufB.shape[0]) {
        throw std::runtime_error("A and B must have same first dimension");
    }
    
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto k = py::array_t<double>(shape, strides);
    auto bufK = k.request();
    
    flinear_kernel(
        static_cast<const double*>(bufA.ptr), na,
        static_cast<const double*>(bufB.ptr), nb,
        static_cast<double*>(bufK.ptr), rep_size
    );
    
    return k;
}

// Wrapper for fmatern_kernel_l2
py::array_t<double> matern_kernel_l2_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> a,
    py::array_t<double, py::array::f_style | py::array::forcecast> b,
    double sigma,
    int order
) {
    auto bufA = a.request();
    auto bufB = b.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    int rep_size = static_cast<int>(bufA.shape[0]);
    int na = static_cast<int>(bufA.shape[1]);
    int nb = static_cast<int>(bufB.shape[1]);
    
    if (bufA.shape[0] != bufB.shape[0]) {
        throw std::runtime_error("A and B must have same first dimension");
    }
    
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto k = py::array_t<double>(shape, strides);
    auto bufK = k.request();
    
    fmatern_kernel_l2(
        static_cast<const double*>(bufA.ptr), na,
        static_cast<const double*>(bufB.ptr), nb,
        static_cast<double*>(bufK.ptr), sigma, order, rep_size
    );
    
    return k;
}

// Wrapper for fsargan_kernel
py::array_t<double> sargan_kernel_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> a,
    py::array_t<double, py::array::f_style | py::array::forcecast> b,
    double sigma,
    py::array_t<double, py::array::f_style | py::array::forcecast> gammas
) {
    auto bufA = a.request();
    auto bufB = b.request();
    auto bufG = gammas.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    if (bufG.ndim != 1) {
        throw std::runtime_error("Gammas must be a 1D array");
    }
    
    int rep_size = static_cast<int>(bufA.shape[0]);
    int na = static_cast<int>(bufA.shape[1]);
    int nb = static_cast<int>(bufB.shape[1]);
    int ng = static_cast<int>(bufG.shape[0]);
    
    if (bufA.shape[0] != bufB.shape[0]) {
        throw std::runtime_error("A and B must have same first dimension");
    }
    
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto k = py::array_t<double>(shape, strides);
    auto bufK = k.request();
    
    fsargan_kernel(
        static_cast<const double*>(bufA.ptr), na,
        static_cast<const double*>(bufB.ptr), nb,
        static_cast<double*>(bufK.ptr), sigma,
        static_cast<const double*>(bufG.ptr), ng, rep_size
    );
    
    return k;
}

// Wrapper for fget_local_kernels_gaussian
py::array_t<double> get_local_kernels_gaussian_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> q1,
    py::array_t<double, py::array::f_style | py::array::forcecast> q2,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2,
    py::array_t<double, py::array::f_style | py::array::forcecast> sigmas
) {
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    auto bufN1 = n1.request();
    auto bufN2 = n2.request();
    auto bufS = sigmas.request();
    
    if (bufQ1.ndim != 2 || bufQ2.ndim != 2) {
        throw std::runtime_error("Q1 and Q2 must be 2D arrays");
    }
    
    if (bufN1.ndim != 1 || bufN2.ndim != 1 || bufS.ndim != 1) {
        throw std::runtime_error("N1, N2, and sigmas must be 1D arrays");
    }
    
    int rep_size = static_cast<int>(bufQ1.shape[0]);
    int nq1 = static_cast<int>(bufQ1.shape[1]);
    int nq2 = static_cast<int>(bufQ2.shape[1]);
    int nm1 = static_cast<int>(bufN1.shape[0]);
    int nm2 = static_cast<int>(bufN2.shape[0]);
    int nsigmas = static_cast<int>(bufS.shape[0]);
    
    // Create output array (nsigmas, nm1, nm2)
    std::vector<ssize_t> shape = {nsigmas, nm1, nm2};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto kernels = py::array_t<double>(shape, strides);
    auto bufK = kernels.request();
    
    fget_local_kernels_gaussian(
        rep_size,
        static_cast<const double*>(bufQ1.ptr),
        static_cast<const double*>(bufQ2.ptr),
        static_cast<const int*>(bufN1.ptr),
        static_cast<const int*>(bufN2.ptr),
        static_cast<const double*>(bufS.ptr),
        nm1, nm2, nsigmas, nq1, nq2,
        static_cast<double*>(bufK.ptr)
    );
    
    return kernels;
}

// Wrapper for fget_local_kernels_laplacian
py::array_t<double> get_local_kernels_laplacian_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> q1,
    py::array_t<double, py::array::f_style | py::array::forcecast> q2,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2,
    py::array_t<double, py::array::f_style | py::array::forcecast> sigmas
) {
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    auto bufN1 = n1.request();
    auto bufN2 = n2.request();
    auto bufS = sigmas.request();
    
    if (bufQ1.ndim != 2 || bufQ2.ndim != 2) {
        throw std::runtime_error("Q1 and Q2 must be 2D arrays");
    }
    
    if (bufN1.ndim != 1 || bufN2.ndim != 1 || bufS.ndim != 1) {
        throw std::runtime_error("N1, N2, and sigmas must be 1D arrays");
    }
    
    int rep_size = static_cast<int>(bufQ1.shape[0]);
    int nq1 = static_cast<int>(bufQ1.shape[1]);
    int nq2 = static_cast<int>(bufQ2.shape[1]);
    int nm1 = static_cast<int>(bufN1.shape[0]);
    int nm2 = static_cast<int>(bufN2.shape[0]);
    int nsigmas = static_cast<int>(bufS.shape[0]);
    
    // Create output array (nsigmas, nm1, nm2)
    std::vector<ssize_t> shape = {nsigmas, nm1, nm2};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto kernels = py::array_t<double>(shape, strides);
    auto bufK = kernels.request();
    
    fget_local_kernels_laplacian(
        rep_size,
        static_cast<const double*>(bufQ1.ptr),
        static_cast<const double*>(bufQ2.ptr),
        static_cast<const int*>(bufN1.ptr),
        static_cast<const int*>(bufN2.ptr),
        static_cast<const double*>(bufS.ptr),
        nm1, nm2, nsigmas, nq1, nq2,
        static_cast<double*>(bufK.ptr)
    );
    
    return kernels;
}

// Wrapper for fget_vector_kernels_gaussian
py::array_t<double> get_vector_kernels_gaussian_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> q1,
    py::array_t<double, py::array::f_style | py::array::forcecast> q2,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2,
    py::array_t<double, py::array::f_style | py::array::forcecast> sigmas
) {
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    auto bufN1 = n1.request();
    auto bufN2 = n2.request();
    auto bufS = sigmas.request();
    
    if (bufQ1.ndim != 3 || bufQ2.ndim != 3) {
        throw std::runtime_error("Q1 and Q2 must be 3D arrays");
    }
    
    if (bufN1.ndim != 1 || bufN2.ndim != 1 || bufS.ndim != 1) {
        throw std::runtime_error("N1, N2, and sigmas must be 1D arrays");
    }
    
    int rep_size = static_cast<int>(bufQ1.shape[0]);
    int max_atoms = static_cast<int>(bufQ1.shape[1]);
    int nm1 = static_cast<int>(bufQ1.shape[2]);
    int nm2 = static_cast<int>(bufQ2.shape[2]);
    int nsigmas = static_cast<int>(bufS.shape[0]);
    
    // Create output array (nsigmas, nm1, nm2)
    std::vector<ssize_t> shape = {nsigmas, nm1, nm2};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto kernels = py::array_t<double>(shape, strides);
    auto bufK = kernels.request();
    
    fget_vector_kernels_gaussian(
        static_cast<const double*>(bufQ1.ptr),
        static_cast<const double*>(bufQ2.ptr),
        static_cast<const int*>(bufN1.ptr),
        static_cast<const int*>(bufN2.ptr),
        static_cast<const double*>(bufS.ptr),
        nm1, nm2, nsigmas, rep_size, max_atoms,
        static_cast<double*>(bufK.ptr)
    );
    
    return kernels;
}

// Wrapper for fget_vector_kernels_laplacian
py::array_t<double> get_vector_kernels_laplacian_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> q1,
    py::array_t<double, py::array::f_style | py::array::forcecast> q2,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2,
    py::array_t<double, py::array::f_style | py::array::forcecast> sigmas
) {
    auto bufQ1 = q1.request();
    auto bufQ2 = q2.request();
    auto bufN1 = n1.request();
    auto bufN2 = n2.request();
    auto bufS = sigmas.request();
    
    if (bufQ1.ndim != 3 || bufQ2.ndim != 3) {
        throw std::runtime_error("Q1 and Q2 must be 3D arrays");
    }
    
    if (bufN1.ndim != 1 || bufN2.ndim != 1 || bufS.ndim != 1) {
        throw std::runtime_error("N1, N2, and sigmas must be 1D arrays");
    }
    
    int rep_size = static_cast<int>(bufQ1.shape[0]);
    int max_atoms = static_cast<int>(bufQ1.shape[1]);
    int nm1 = static_cast<int>(bufQ1.shape[2]);
    int nm2 = static_cast<int>(bufQ2.shape[2]);
    int nsigmas = static_cast<int>(bufS.shape[0]);
    
    // Create output array (nsigmas, nm1, nm2)
    std::vector<ssize_t> shape = {nsigmas, nm1, nm2};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto kernels = py::array_t<double>(shape, strides);
    auto bufK = kernels.request();
    
    fget_vector_kernels_laplacian(
        static_cast<const double*>(bufQ1.ptr),
        static_cast<const double*>(bufQ2.ptr),
        static_cast<const int*>(bufN1.ptr),
        static_cast<const int*>(bufN2.ptr),
        static_cast<const double*>(bufS.ptr),
        nm1, nm2, nsigmas, rep_size, max_atoms,
        static_cast<double*>(bufK.ptr)
    );
    
    return kernels;
}

// Wrapper for fget_vector_kernels_gaussian_symmetric
py::array_t<double> get_vector_kernels_gaussian_symmetric_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> q,
    py::array_t<int, py::array::f_style | py::array::forcecast> n,
    py::array_t<double, py::array::f_style | py::array::forcecast> sigmas
) {
    auto bufQ = q.request();
    auto bufN = n.request();
    auto bufS = sigmas.request();
    
    if (bufQ.ndim != 3) {
        throw std::runtime_error("Q must be a 3D array");
    }
    
    if (bufN.ndim != 1 || bufS.ndim != 1) {
        throw std::runtime_error("N and sigmas must be 1D arrays");
    }
    
    int rep_size = static_cast<int>(bufQ.shape[0]);
    int max_atoms = static_cast<int>(bufQ.shape[1]);
    int nm = static_cast<int>(bufQ.shape[2]);
    int nsigmas = static_cast<int>(bufS.shape[0]);
    
    // Create output array (nsigmas, nm, nm)
    std::vector<ssize_t> shape = {nsigmas, nm, nm};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm};
    auto kernels = py::array_t<double>(shape, strides);
    auto bufK = kernels.request();
    
    fget_vector_kernels_gaussian_symmetric(
        static_cast<const double*>(bufQ.ptr),
        static_cast<const int*>(bufN.ptr),
        static_cast<const double*>(bufS.ptr),
        nm, nsigmas, rep_size, max_atoms,
        static_cast<double*>(bufK.ptr)
    );
    
    return kernels;
}

// Wrapper for fget_vector_kernels_laplacian_symmetric
py::array_t<double> get_vector_kernels_laplacian_symmetric_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> q,
    py::array_t<int, py::array::f_style | py::array::forcecast> n,
    py::array_t<double, py::array::f_style | py::array::forcecast> sigmas
) {
    auto bufQ = q.request();
    auto bufN = n.request();
    auto bufS = sigmas.request();
    
    if (bufQ.ndim != 3) {
        throw std::runtime_error("Q must be a 3D array");
    }
    
    if (bufN.ndim != 1 || bufS.ndim != 1) {
        throw std::runtime_error("N and sigmas must be 1D arrays");
    }
    
    int rep_size = static_cast<int>(bufQ.shape[0]);
    int max_atoms = static_cast<int>(bufQ.shape[1]);
    int nm = static_cast<int>(bufQ.shape[2]);
    int nsigmas = static_cast<int>(bufS.shape[0]);
    
    // Create output array (nsigmas, nm, nm)
    std::vector<ssize_t> shape = {nsigmas, nm, nm};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm};
    auto kernels = py::array_t<double>(shape, strides);
    auto bufK = kernels.request();
    
    fget_vector_kernels_laplacian_symmetric(
        static_cast<const double*>(bufQ.ptr),
        static_cast<const int*>(bufN.ptr),
        static_cast<const double*>(bufS.ptr),
        nm, nsigmas, rep_size, max_atoms,
        static_cast<double*>(bufK.ptr)
    );
    
    return kernels;
}

PYBIND11_MODULE(_fkernels, m) {
    m.doc() = "QMLlib kernel functions";

    m.def("fkpca", &kpca_wrapper,
        py::arg("k"), py::arg("n"), py::arg("centering"),
        "Kernel PCA decomposition");

    m.def("fwasserstein_kernel", &wasserstein_kernel_wrapper,
        py::arg("a"), py::arg("na"), py::arg("b"), py::arg("nb"),
        py::arg("sigma"), py::arg("p"), py::arg("q"),
        "Wasserstein kernel computation");
    
    m.def("fgaussian_kernel", &gaussian_kernel_wrapper,
        py::arg("a"), py::arg("b"), py::arg("sigma"),
        "Gaussian kernel");
    
    m.def("fgaussian_kernel_symmetric", &gaussian_kernel_symmetric_wrapper,
        py::arg("x"), py::arg("sigma"),
        "Symmetric Gaussian kernel");
    
    m.def("flaplacian_kernel", &laplacian_kernel_wrapper,
        py::arg("a"), py::arg("b"), py::arg("sigma"),
        "Laplacian kernel");
    
    m.def("flaplacian_kernel_symmetric", &laplacian_kernel_symmetric_wrapper,
        py::arg("x"), py::arg("sigma"),
        "Symmetric Laplacian kernel");
    
    m.def("flinear_kernel", &linear_kernel_wrapper,
        py::arg("a"), py::arg("b"),
        "Linear kernel");
    
    m.def("fmatern_kernel_l2", &matern_kernel_l2_wrapper,
        py::arg("a"), py::arg("b"), py::arg("sigma"), py::arg("order"),
        "Matern kernel with L2 distance");
    
    m.def("fsargan_kernel", &sargan_kernel_wrapper,
        py::arg("a"), py::arg("b"), py::arg("sigma"), py::arg("gammas"),
        "Sargan kernel");
    
    m.def("fget_local_kernels_gaussian", &get_local_kernels_gaussian_wrapper,
        py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
        py::arg("sigmas"),
        "Local Gaussian kernels");
    
    m.def("fget_local_kernels_laplacian", &get_local_kernels_laplacian_wrapper,
        py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
        py::arg("sigmas"),
        "Local Laplacian kernels");
    
    m.def("fget_vector_kernels_gaussian", &get_vector_kernels_gaussian_wrapper,
        py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
        py::arg("sigmas"),
        "Vector Gaussian kernels");
    
    m.def("fget_vector_kernels_laplacian", &get_vector_kernels_laplacian_wrapper,
        py::arg("q1"), py::arg("q2"), py::arg("n1"), py::arg("n2"),
        py::arg("sigmas"),
        "Vector Laplacian kernels");
    
    m.def("fget_vector_kernels_gaussian_symmetric", &get_vector_kernels_gaussian_symmetric_wrapper,
        py::arg("q"), py::arg("n"), py::arg("sigmas"),
        "Symmetric vector Gaussian kernels");
    
    m.def("fget_vector_kernels_laplacian_symmetric", &get_vector_kernels_laplacian_symmetric_wrapper,
        py::arg("q"), py::arg("n"), py::arg("sigmas"),
        "Symmetric vector Laplacian kernels");
}
