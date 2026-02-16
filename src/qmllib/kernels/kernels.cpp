#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cblas.h>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <vector>

namespace py = pybind11;

void ckernel_symm_blas(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                      py::array_t<double, py::array::c_style | py::array::forcecast> K,
                      double alpha) {
    // Request buffers
    auto bufX = X.request();
    auto bufK = K.request();

    if (bufX.ndim != 2 || bufK.ndim != 2) {
        throw std::runtime_error("X and K must be 2D arrays");
    }

    int n = bufX.shape[0];   // rows of X
    int rep_size  = bufX.shape[1];   // cols of X
    // int ldk = bufK.shape[1];   // leading dimension for row-major

    double* Xptr = static_cast<double*>(bufX.ptr);
    double* Kptr = static_cast<double*>(bufK.ptr);

    double t0 = omp_get_wtime();

    // Equivalent to: K = -2*alpha * X * X^T (symmetric, row-major)
    // SYRK in row-major: C = alpha*A*A^T + beta*C
    // Better to use Lower triangle in C
    std::cout << "sizes: n=" << n << ", rep_size=" << rep_size << "\n";
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, rep_size, -2.0 * alpha, Xptr, rep_size, 0.0, Kptr, n);
    double t1 = omp_get_wtime();
    std::cout << "dsyrk took " << (t1 - t0) << " seconds\n";

    // // Extract diagonal of K
    std::vector<double> diag(n);
    for (int i = 0; i < n; i++) {
        diag[i] = -0.5 * Kptr[i * n + i];
    }

    // Add diag + diag^T using dsyr2 with onevec = 1
    std::vector<double> onevec(n, 1.0);
    cblas_dsyr2(CblasRowMajor, CblasLower, n, 1.0,
                onevec.data(), 1, diag.data(), 1, Kptr, n);

    // Exponentiate lower triangle
    #pragma omp parallel for  shared(Kptr, n) schedule(guided)
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            Kptr[j * n + i] = std::exp(Kptr[j * n + i]);
        }
    }
}

namespace py = pybind11;

void ckernel_syrk_test(py::array_t<double> X,
                       py::array_t<double> K,
                       double alpha) {
    auto bufX = X.request();
    auto bufK = K.request();

    std::cout << "X: shape=(" << bufX.shape[0] << "," << bufX.shape[1] << ") "
              << "strides=(" << bufX.strides[0] << "," << bufX.strides[1] << ") "
              << "c_contig=" << (X.flags() & py::array::c_style ? "true" : "false") << " "
              << "f_contig=" << (X.flags() & py::array::f_style ? "true" : "false") << " "
              << "owndata=" << (X.owndata() ? "true" : "false") << std::endl;

    std::cout << "K: shape=(" << bufK.shape[0] << "," << bufK.shape[1] << ") "
              << "strides=(" << bufK.strides[0] << "," << bufK.strides[1] << ") "
              << "c_contig=" << (K.flags() & py::array::c_style ? "true" : "false") << " "
              << "f_contig=" << (K.flags() & py::array::f_style ? "true" : "false") << " "
              << "owndata=" << (K.owndata() ? "true" : "false") << std::endl;

    // Time DSYRK only
    py::gil_scoped_release release;
    double t0 = omp_get_wtime();
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                bufX.shape[0], bufX.shape[1], -2.0 * alpha,
                static_cast<double*>(bufX.ptr), bufX.shape[1],
                0.0, static_cast<double*>(bufK.ptr), bufK.shape[1]);
    double t1 = omp_get_wtime();

    std::cout << "dsyrk took " << (t1 - t0) << " s\n";
}

void bench_dsyrk(int n, int rep_size, double alpha) {
    std::vector<double> X(n * rep_size);
    std::vector<double> K(n * n);

    for (int i = 0; i < n * rep_size; i++) X[i] = std::sin(0.001 * i);

    double t0 = omp_get_wtime();

        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                    n, rep_size, -2.0 * alpha,
                    X.data(), rep_size,
                    0.0, K.data(), n);
    
double t1 = omp_get_wtime();
    std::cout << "dsyrk took " << (t1 - t0) << " seconds\n";

}

void bench_dsyrk_Xinternal(py::array_t<double> K, double alpha) {
    auto bufK = K.request();
    int n = bufK.shape[0], rep_size = 512;

    std::vector<double> X(n * rep_size);
    for (int i = 0; i < n * rep_size; i++) X[i] = std::sin(0.001 * i);

    uintptr_t addr = reinterpret_cast<uintptr_t>(bufK.ptr);
    std::cout << "K base address = " << (void*)bufK.ptr
          << " (mod 64 = " << (addr % 64) << ")\n";

    double* Kptr = static_cast<double*>(bufK.ptr);

    double t0 = omp_get_wtime();
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, rep_size, -2.0*alpha,
                X.data(), rep_size,
                0.0, Kptr, n);
    double t1 = omp_get_wtime();
    std::cout << "bench_dsyrk X internal, K from Python took " << (t1 - t0) << " s\n";
}

void bench_dsyrk_Kinternal(py::array_t<double> X, double alpha) {
    auto bufX = X.request();
    int n = bufX.shape[0], rep_size = bufX.shape[1];
    double* Xptr = static_cast<double*>(bufX.ptr);

    std::vector<double> K(n * n);

    double t0 = omp_get_wtime();
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, rep_size, -2.0*alpha,
                Xptr, rep_size,
                0.0, K.data(), n);
    double t1 = omp_get_wtime();
    std::cout << "bench_dsyrk K internal, X from Python took " << (t1 - t0) << " s\n";
}


// Simple aligned alloc (POSIX, 64-byte)
inline double* aligned_alloc_64(size_t nelems) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, nelems * sizeof(double)) != 0)
        throw std::bad_alloc();
    return static_cast<double*>(p);
}
inline void aligned_free_64(void* p) { std::free(p); }

py::array_t<double> cfkernel_symm_blas(
    py::array_t<double, py::array::c_style> X,
    double alpha
) {
    auto bufX = X.request();

    if (bufX.ndim != 2)
        throw std::runtime_error("X must be 2D");

    int n        = static_cast<int>(bufX.shape[0]);
    int rep_size = static_cast<int>(bufX.shape[1]);
    double* Xptr = static_cast<double*>(bufX.ptr);

    // Allocate aligned K (row-major)
    size_t nelems = static_cast<size_t>(n) * static_cast<size_t>(n);
    double* Kptr = aligned_alloc_64(nelems);

    auto capsule = py::capsule(Kptr, [](void* p){ aligned_free_64(p); });

    auto K = py::array_t<double>(
        {n, n},
        {static_cast<ssize_t>(n) * sizeof(double), sizeof(double)}, // row-major strides
        Kptr,
        capsule
    );

    // === Compute ===
    double t0 = omp_get_wtime();

    // SYRK in row-major (lower triangle)
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
                n, rep_size, -2.0 * alpha,
                Xptr, rep_size,
                0.0, Kptr, n);

    double t1 = omp_get_wtime();
    std::cout << "dsyrk took " << (t1 - t0) << " seconds\n";

    // Extract diagonal
    std::vector<double> diag(n);
    for (int i = 0; i < n; i++) {
        diag[i] = -0.5 * Kptr[i*n + i];  // row-major diag
    }

    // Add diag + diag^T
    std::vector<double> onevec(n, 1.0);
    cblas_dsyr2(CblasRowMajor, CblasLower, n, 1.0,
                onevec.data(), 1, diag.data(), 1, Kptr, n);

    // Exponentiate lower triangle
    #pragma omp parallel for schedule(guided)
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            Kptr[j*n + i] = std::exp(Kptr[j*n + i]);
        }
    }

    return K;
}
