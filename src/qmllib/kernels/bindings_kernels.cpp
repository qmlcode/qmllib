#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// declare the kernel function implemented in kernels.cpp
void ckernel_symm_blas(py::array_t<double, py::array::c_style | py::array::forcecast>,
                       py::array_t<double, py::array::c_style | py::array::forcecast>,
                       double);

// declare the kernel function implemented in kernels.cpp
// void ckernel_syrk_test(py::array_t<double, py::array::c_style | py::array::forcecast>,
//                        py::array_t<double, py::array::c_style | py::array::forcecast>,
//                        double);

void ckernel_syrk_test(py::array_t<double> X,
                       py::array_t<double> K,
                       double alpha);

void bench_dsyrk(int n, int rep_size, double alpha);

// Case 1: X internal, K from Python
void bench_dsyrk_Xinternal(py::array_t<double> K, double alpha);

// Case 2: K internal, X from Python
void bench_dsyrk_Kinternal(py::array_t<double> X, double alpha);

py::array_t<double> cfkernel_symm_blas(
    py::array_t<double, py::array::c_style> X,
    double alpha
);


PYBIND11_MODULE(_kernels, m) {
    m.def("cfkernel_symm_blas", &cfkernel_symm_blas,
          py::arg("X"), py::arg("alpha"),
          "Compute symmetric kernel matrix (C++/BLAS, NumPy C-order)");
    m.doc() = "Symmetric kernel construction (C++ + BLAS, NumPy-compatible)";
    m.def("ckernel_symm_blas", &ckernel_symm_blas,
          py::arg("X"), py::arg("K"), py::arg("alpha"),
          "Compute symmetric kernel matrix (NumPy C-order).");
    m.def("ckernel_syrk_test", &ckernel_syrk_test,
      py::arg("X"), py::arg("K"), py::arg("alpha"),
      "Compute symmetric kernel matrix (handles C and F order arrays).");
    m.def("bench_dsyrk", &bench_dsyrk,
          py::arg("n"), py::arg("rep_size"), py::arg("alpha"),
          "Benchmark dsyrk performance."
         );
    m.def("bench_dsyrk_Xinternal", &bench_dsyrk_Xinternal,
          py::arg("K"), py::arg("alpha"),
          "Benchmark DSYRK with X allocated inside C++ and K provided by Python.");

    m.def("bench_dsyrk_Kinternal", &bench_dsyrk_Kinternal,
          py::arg("X"), py::arg("alpha"),
          "Benchmark DSYRK with K allocated inside C++ and X provided by Python.");
}
