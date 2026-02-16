#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>

namespace py = pybind11;

// Declare C ABI Fortran functions
extern "C" {
    void fmanhattan_distance(const double* A, int nv, int na,
                            const double* B, int nb, double* D);
    void fl2_distance(const double* A, int nv, int na,
                     const double* B, int nb, double* D);
    void fp_distance_double(const double* A, int nv, int na,
                           const double* B, int nb, double* D, double p);
    void fp_distance_integer(const double* A, int nv, int na,
                            const double* B, int nb, double* D, int p);
}

// Wrapper for fmanhattan_distance
py::array_t<double> manhattan_distance_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> A,
    py::array_t<double, py::array::f_style | py::array::forcecast> B
) {
    auto bufA = A.request();
    auto bufB = B.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    int nv = static_cast<int>(bufA.shape[0]);
    int na = static_cast<int>(bufA.shape[1]);
    int nb = static_cast<int>(bufB.shape[1]);
    
    if (bufB.shape[0] != nv) {
        throw std::runtime_error("A and B must have same first dimension");
    }
    
    // Create Fortran-style (column-major) output array
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto D = py::array_t<double>(shape, strides);
    auto bufD = D.request();
    
    // Initialize to zero
    std::memset(bufD.ptr, 0, na * nb * sizeof(double));
    
    fmanhattan_distance(
        static_cast<const double*>(bufA.ptr), nv, na,
        static_cast<const double*>(bufB.ptr), nb,
        static_cast<double*>(bufD.ptr)
    );
    
    return D;
}

// Wrapper for fl2_distance
py::array_t<double> l2_distance_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> A,
    py::array_t<double, py::array::f_style | py::array::forcecast> B
) {
    auto bufA = A.request();
    auto bufB = B.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    int nv = static_cast<int>(bufA.shape[0]);
    int na = static_cast<int>(bufA.shape[1]);
    int nb = static_cast<int>(bufB.shape[1]);
    
    if (bufB.shape[0] != nv) {
        throw std::runtime_error("A and B must have same first dimension");
    }
    
    // Create Fortran-style (column-major) output array
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto D = py::array_t<double>(shape, strides);
    auto bufD = D.request();
    
    // Initialize to zero
    std::memset(bufD.ptr, 0, na * nb * sizeof(double));
    
    fl2_distance(
        static_cast<const double*>(bufA.ptr), nv, na,
        static_cast<const double*>(bufB.ptr), nb,
        static_cast<double*>(bufD.ptr)
    );
    
    return D;
}

// Wrapper for fp_distance_double
py::array_t<double> p_distance_double_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> A,
    py::array_t<double, py::array::f_style | py::array::forcecast> B,
    double p
) {
    auto bufA = A.request();
    auto bufB = B.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    int nv = static_cast<int>(bufA.shape[0]);
    int na = static_cast<int>(bufA.shape[1]);
    int nb = static_cast<int>(bufB.shape[1]);
    
    if (bufB.shape[0] != nv) {
        throw std::runtime_error("A and B must have same first dimension");
    }
    
    // Create Fortran-style (column-major) output array
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto D = py::array_t<double>(shape, strides);
    auto bufD = D.request();
    
    // Initialize to zero
    std::memset(bufD.ptr, 0, na * nb * sizeof(double));
    
    fp_distance_double(
        static_cast<const double*>(bufA.ptr), nv, na,
        static_cast<const double*>(bufB.ptr), nb,
        static_cast<double*>(bufD.ptr), p
    );
    
    return D;
}

// Wrapper for fp_distance_integer
py::array_t<double> p_distance_integer_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> A,
    py::array_t<double, py::array::f_style | py::array::forcecast> B,
    int p
) {
    auto bufA = A.request();
    auto bufB = B.request();
    
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("A and B must be 2D arrays");
    }
    
    int nv = static_cast<int>(bufA.shape[0]);
    int na = static_cast<int>(bufA.shape[1]);
    int nb = static_cast<int>(bufB.shape[1]);
    
    if (bufB.shape[0] != nv) {
        throw std::runtime_error("A and B must have same first dimension");
    }
    
    // Create Fortran-style (column-major) output array
    std::vector<ssize_t> shape = {na, nb};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * na};
    auto D = py::array_t<double>(shape, strides);
    auto bufD = D.request();
    
    // Initialize to zero
    std::memset(bufD.ptr, 0, na * nb * sizeof(double));
    
    fp_distance_integer(
        static_cast<const double*>(bufA.ptr), nv, na,
        static_cast<const double*>(bufB.ptr), nb,
        static_cast<double*>(bufD.ptr), p
    );
    
    return D;
}

PYBIND11_MODULE(_fdistance, m) {
    m.doc() = "QMLlib distance functions (Manhattan, L2, Lp)";

    m.def("fmanhattan_distance", &manhattan_distance_wrapper,
        py::arg("a"), py::arg("b"),
        "Compute Manhattan (L1) distance matrix");

    m.def("fl2_distance", &l2_distance_wrapper,
        py::arg("a"), py::arg("b"),
        "Compute L2 (Euclidean) distance matrix");

    m.def("fp_distance_double", &p_distance_double_wrapper,
        py::arg("a"), py::arg("b"), py::arg("p"),
        "Compute Lp distance matrix (double precision p)");

    m.def("fp_distance_integer", &p_distance_integer_wrapper,
        py::arg("a"), py::arg("b"), py::arg("p"),
        "Compute Lp distance matrix (integer p)");
}
