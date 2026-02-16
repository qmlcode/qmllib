#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Fortran function declarations
extern "C" {
    void fget_sbot(const double* coordinates, const double* nuclear_charges,
                   int z1, int z2, int z3, double rcut, int nx, double dgrid,
                   double sigma, double coeff, double* ys, int natoms);
    
    void fget_sbot_local(const double* coordinates, const double* nuclear_charges,
                        int ia_python, int z1, int z2, int z3, double rcut, int nx,
                        double dgrid, double sigma, double coeff, double* ys, int natoms);
    
    void fget_sbop(const double* coordinates, const double* nuclear_charges,
                   int z1, int z2, double rcut, int nx, double dgrid, double sigma,
                   double coeff, double rpower, double* ys, int natoms);
    
    void fget_sbop_local(const double* coordinates, const double* nuclear_charges,
                        int ia_python, int z1, int z2, double rcut, int nx,
                        double dgrid, double sigma, double coeff, double rpower,
                        double* ys, int natoms);
}

// Wrapper for fget_sbot
py::array_t<double> get_sbot_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> nuclear_charges_in,
    int z1, int z2, int z3, double rcut, int nx, double dgrid,
    double sigma, double coeff
) {
    // Ensure converted arrays stay alive
    auto coordinates = py::array_t<double, py::array::f_style | py::array::forcecast>(coordinates_in);
    auto nuclear_charges = py::array_t<double, py::array::f_style | py::array::forcecast>(nuclear_charges_in);
    
    auto bufCoords = coordinates.request();
    auto bufCharges = nuclear_charges.request();
    
    int natoms = static_cast<int>(bufCoords.shape[0]);
    
    // Create output array - Fortran column-major
    std::vector<ssize_t> shape = {nx};
    std::vector<ssize_t> strides = {sizeof(double)};
    auto ys = py::array_t<double>(shape, strides);
    auto bufYs = ys.request();
    
    fget_sbot(
        static_cast<const double*>(bufCoords.ptr),
        static_cast<const double*>(bufCharges.ptr),
        z1, z2, z3, rcut, nx, dgrid, sigma, coeff,
        static_cast<double*>(bufYs.ptr),
        natoms
    );
    
    return ys;
}

// Wrapper for fget_sbot_local
py::array_t<double> get_sbot_local_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> nuclear_charges_in,
    int ia_python, int z1, int z2, int z3, double rcut, int nx, double dgrid,
    double sigma, double coeff
) {
    // Ensure converted arrays stay alive
    auto coordinates = py::array_t<double, py::array::f_style | py::array::forcecast>(coordinates_in);
    auto nuclear_charges = py::array_t<double, py::array::f_style | py::array::forcecast>(nuclear_charges_in);
    
    auto bufCoords = coordinates.request();
    auto bufCharges = nuclear_charges.request();
    
    int natoms = static_cast<int>(bufCoords.shape[0]);
    
    // Create output array - Fortran column-major
    std::vector<ssize_t> shape = {nx};
    std::vector<ssize_t> strides = {sizeof(double)};
    auto ys = py::array_t<double>(shape, strides);
    auto bufYs = ys.request();
    
    fget_sbot_local(
        static_cast<const double*>(bufCoords.ptr),
        static_cast<const double*>(bufCharges.ptr),
        ia_python, z1, z2, z3, rcut, nx, dgrid, sigma, coeff,
        static_cast<double*>(bufYs.ptr),
        natoms
    );
    
    return ys;
}

// Wrapper for fget_sbop
py::array_t<double> get_sbop_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> nuclear_charges_in,
    int z1, int z2, double rcut, int nx, double dgrid, double sigma,
    double coeff, double rpower
) {
    // Ensure converted arrays stay alive
    auto coordinates = py::array_t<double, py::array::f_style | py::array::forcecast>(coordinates_in);
    auto nuclear_charges = py::array_t<double, py::array::f_style | py::array::forcecast>(nuclear_charges_in);
    
    auto bufCoords = coordinates.request();
    auto bufCharges = nuclear_charges.request();
    
    int natoms = static_cast<int>(bufCoords.shape[0]);
    
    // Create output array - Fortran column-major
    std::vector<ssize_t> shape = {nx};
    std::vector<ssize_t> strides = {sizeof(double)};
    auto ys = py::array_t<double>(shape, strides);
    auto bufYs = ys.request();
    
    fget_sbop(
        static_cast<const double*>(bufCoords.ptr),
        static_cast<const double*>(bufCharges.ptr),
        z1, z2, rcut, nx, dgrid, sigma, coeff, rpower,
        static_cast<double*>(bufYs.ptr),
        natoms
    );
    
    return ys;
}

// Wrapper for fget_sbop_local
py::array_t<double> get_sbop_local_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> nuclear_charges_in,
    int ia_python, int z1, int z2, double rcut, int nx, double dgrid,
    double sigma, double coeff, double rpower
) {
    // Ensure converted arrays stay alive
    auto coordinates = py::array_t<double, py::array::f_style | py::array::forcecast>(coordinates_in);
    auto nuclear_charges = py::array_t<double, py::array::f_style | py::array::forcecast>(nuclear_charges_in);
    
    auto bufCoords = coordinates.request();
    auto bufCharges = nuclear_charges.request();
    
    int natoms = static_cast<int>(bufCoords.shape[0]);
    
    // Create output array - Fortran column-major
    std::vector<ssize_t> shape = {nx};
    std::vector<ssize_t> strides = {sizeof(double)};
    auto ys = py::array_t<double>(shape, strides);
    auto bufYs = ys.request();
    
    fget_sbop_local(
        static_cast<const double*>(bufCoords.ptr),
        static_cast<const double*>(bufCharges.ptr),
        ia_python, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower,
        static_cast<double*>(bufYs.ptr),
        natoms
    );
    
    return ys;
}

PYBIND11_MODULE(_fslatm, m) {
    m.doc() = "QMLlib SLATM representation functions";

    m.def("fget_sbot", &get_sbot_wrapper,
        py::arg("coordinates"), py::arg("nuclear_charges"),
        py::arg("z1"), py::arg("z2"), py::arg("z3"),
        py::arg("rcut"), py::arg("nx"), py::arg("dgrid"),
        py::arg("sigma"), py::arg("coeff"),
        "SBOT three-body representation");

    m.def("fget_sbot_local", &get_sbot_local_wrapper,
        py::arg("coordinates"), py::arg("nuclear_charges"),
        py::arg("ia_python"), py::arg("z1"), py::arg("z2"), py::arg("z3"),
        py::arg("rcut"), py::arg("nx"), py::arg("dgrid"),
        py::arg("sigma"), py::arg("coeff"),
        "SBOT local three-body representation");

    m.def("fget_sbop", &get_sbop_wrapper,
        py::arg("coordinates"), py::arg("nuclear_charges"),
        py::arg("z1"), py::arg("z2"),
        py::arg("rcut"), py::arg("nx"), py::arg("dgrid"),
        py::arg("sigma"), py::arg("coeff"), py::arg("rpower"),
        "SBOP two-body representation");

    m.def("fget_sbop_local", &get_sbop_local_wrapper,
        py::arg("coordinates"), py::arg("nuclear_charges"),
        py::arg("ia_python"), py::arg("z1"), py::arg("z2"),
        py::arg("rcut"), py::arg("nx"), py::arg("dgrid"),
        py::arg("sigma"), py::arg("coeff"), py::arg("rpower"),
        "SBOP local two-body representation");
}
