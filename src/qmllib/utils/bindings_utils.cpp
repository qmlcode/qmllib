#include <pybind11/pybind11.h>

namespace py = pybind11;

extern "C" {
    void check_openmp(int* compiled_with_openmp);
    int get_threads();
}

PYBIND11_MODULE(_utils, m) {
    m.doc() = "QMLlib utilities module";

    m.def("check_openmp", []() -> bool {
        int result;
        check_openmp(&result);
        return result != 0;
    }, "Check if compiled with OpenMP support");

    m.def("get_threads", &get_threads,
        "Get the maximum number of OpenMP threads");
}
