#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdlib>
extern "C" {
  void compute_inverse_distance(const double* x_3_by_n, int n, double* d_packed);
  void kernel_symm_simple(const double* x, int lda, int n, double* k, int ldk, double alpha);
  void kernel_symm_blas(const double* x, int lda, int n, double* k, int ldk, double alpha);
}

namespace py = pybind11;


inline double* aligned_alloc_64(size_t nelems) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, nelems * sizeof(double)) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<double*>(p);
}

inline void aligned_free_64(void* p) {
    std::free(p);
}

py::array_t<double> inverse_distance(py::array_t<double, py::array::c_style | py::array::forcecast> X) {
    auto buf = X.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("X must have shape (N,3)");
    }
    const int n = static_cast<int>(buf.shape[0]);

    // D packed length
    const ssize_t m = static_cast<ssize_t>(n) * (n - 1) / 2;
    auto D = py::array_t<double>(m);

    // Pass row-major (N,3) as transposed view (3,N) to Fortran without copy:
    // NumPy will give a view; pybind11 exposes data pointer for the view.
    py::array_t<double> XT({3, n}, {buf.strides[1], buf.strides[0]}, static_cast<double*>(buf.ptr), X);

    compute_inverse_distance(static_cast<const double*>(XT.request().ptr), n,
                                      static_cast<double*>(D.request().ptr));
    return D;
}

py::array_t<double> kernel_symm_simple_py(
    py::array_t<double, py::array::forcecast | py::array::f_style> X,
    double alpha
) {
    // Require (rep_size, n) in Fortran order; forcecast|f_style will copy if needed.
    auto xb = X.request();
    if (xb.ndim != 2) {
        throw std::runtime_error("X must be 2D with shape (rep_size, n) in column-major (Fortran) order");
    }
    const int lda = static_cast<int>(xb.shape[0]);
    const int n   = static_cast<int>(xb.shape[1]);

    // Allocate K as Fortran-order (n x n): stride0 = 8, stride1 = n*8
    auto K = py::array_t<double>({n, n}, {sizeof(double), static_cast<ssize_t>(n)*sizeof(double)});

    kernel_symm_simple(static_cast<const double*>(xb.ptr),
                  lda, n,
                  static_cast<double*>(K.request().ptr),
                  /*ldk=*/n, alpha);

    return K;
}


py::array_t<double> kernel_symm_blas_py(
    py::array_t<double, py::array::forcecast | py::array::f_style> X,
    double alpha
) {
    // Require (rep_size, n) in Fortran order; forcecast|f_style will copy if needed.
    auto xb = X.request();
    if (xb.ndim != 2) {
        throw std::runtime_error("X must be 2D with shape (rep_size, n) in column-major (Fortran) order");
    }
    const int lda = static_cast<int>(xb.shape[0]);
    const int n   = static_cast<int>(xb.shape[1]);

    // Allocate K as Fortran-order (n x n): stride0 = 8, stride1 = n*8
    // auto K = py::array_t<double>({n, n}, {sizeof(double), static_cast<ssize_t>(n)*sizeof(double)});
    auto ptr = aligned_alloc_64(static_cast<size_t>(n) * static_cast<size_t>(n));

    auto capsule = py::capsule(ptr, [](void *p) {
        aligned_free_64(p);
    });
    
    auto K = py::array_t<double>(
        {n, n},
        {static_cast<ssize_t>(n) * sizeof(double), sizeof(double)}, // row-major
        ptr,
        capsule
    );

    kernel_symm_blas(static_cast<const double*>(xb.ptr),
                  lda, n,
                  static_cast<double*>(K.request().ptr),
                  /*ldk=*/n, alpha);

    return K;
}

PYBIND11_MODULE(_qmllib, m) {
    m.doc() = "qmllib: Fortran kernels with C ABI and Python bindings";
    m.def("inverse_distance", &inverse_distance, "Compute packed inverse distance matrix from (N,3) coordinates");
    m.def("kernel_symm_simple", &kernel_symm_simple_py,
      "Compute K (upper triangle) with Gaussian-like exp(alpha * ||xi-xj||^2). "
      "X must be shape (rep_size, n), Fortran-order.");
    m.def("kernel_symm_blas", &kernel_symm_blas_py,
      "Compute K (upper triangle) with Gaussian-like exp(alpha * ||xi-xj||^2). "
      "X must be shape (rep_size, n), Fortran-order.");

}
