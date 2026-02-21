#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Declare C ABI Fortran functions
extern "C" {
    void fgenerate_acsf(const double* coordinates, const int* nuclear_charges,
                       const int* elements, const double* Rs2, const double* Rs3,
                       const double* Ts, double eta2, double eta3, double zeta,
                       double rcut, double acut, int natoms, int rep_size,
                       double* rep, int n_elements, int n_Rs2, int n_Rs3, int n_Ts);
    
    void fgenerate_acsf_and_gradients(const double* coordinates, const int* nuclear_charges,
                                     const int* elements, const double* Rs2, const double* Rs3,
                                     const double* Ts, double eta2, double eta3, double zeta,
                                     double rcut, double acut, int natoms, int rep_size,
                                     double* rep, double* grad, int n_elements, int n_Rs2,
                                     int n_Rs3, int n_Ts);
    
    void fgenerate_fchl_acsf(const double* coordinates, const int* nuclear_charges,
                            const int* elements, const double* Rs2, const double* Rs3,
                            const double* Ts, double eta2, double eta3, double zeta,
                            double rcut, double acut, int natoms, int rep_size,
                            double two_body_decay, double three_body_decay,
                            double three_body_weight, double* rep, int n_elements,
                            int n_Rs2, int n_Rs3, int n_Ts);
    
    void fgenerate_fchl_acsf_and_gradients(const double* coordinates, const int* nuclear_charges,
                                          const int* elements, const double* Rs2, const double* Rs3,
                                          const double* Ts, double eta2, double eta3, double zeta,
                                          double rcut, double acut, int natoms, int rep_size,
                                          double two_body_decay, double three_body_decay,
                                          double three_body_weight, double* rep, double* grad,
                                          int n_elements, int n_Rs2, int n_Rs3, int n_Ts);
}

// Wrapper for fgenerate_acsf
py::array_t<double> generate_acsf_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    py::array_t<int, py::array::f_style | py::array::forcecast> nuclear_charges,
    py::array_t<int, py::array::f_style | py::array::forcecast> elements,
    py::array_t<double, py::array::f_style | py::array::forcecast> Rs2,
    py::array_t<double, py::array::f_style | py::array::forcecast> Rs3,
    py::array_t<double, py::array::f_style | py::array::forcecast> Ts,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    int natoms,
    int rep_size
) {
    auto bufCoords = coordinates.request();
    auto bufCharges = nuclear_charges.request();
    auto bufElements = elements.request();
    
    int n_elements = static_cast<int>(bufElements.size);
    int n_Rs2 = static_cast<int>(Rs2.request().size);
    int n_Rs3 = static_cast<int>(Rs3.request().size);
    int n_Ts = static_cast<int>(Ts.request().size);
    
    // Create output array (natoms, rep_size) - Fortran column-major
    std::vector<ssize_t> shape = {natoms, rep_size};
    std::vector<ssize_t> strides = {static_cast<ssize_t>(sizeof(double)), static_cast<ssize_t>(sizeof(double) * natoms)};
    auto rep = py::array_t<double>(shape, strides);
    auto bufRep = rep.request();
    
    fgenerate_acsf(
        static_cast<const double*>(bufCoords.ptr),
        static_cast<const int*>(bufCharges.ptr),
        static_cast<const int*>(bufElements.ptr),
        static_cast<const double*>(Rs2.request().ptr),
        static_cast<const double*>(Rs3.request().ptr),
        static_cast<const double*>(Ts.request().ptr),
        eta2, eta3, zeta, rcut, acut, natoms, rep_size,
        static_cast<double*>(bufRep.ptr),
        n_elements, n_Rs2, n_Rs3, n_Ts
    );
    
    return rep;
}

// Wrapper for fgenerate_acsf_and_gradients
std::tuple<py::array_t<double>, py::array_t<double>> generate_acsf_and_gradients_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    py::array_t<int, py::array::f_style | py::array::forcecast> nuclear_charges,
    py::array_t<int, py::array::f_style | py::array::forcecast> elements,
    py::array_t<double, py::array::f_style | py::array::forcecast> Rs2,
    py::array_t<double, py::array::f_style | py::array::forcecast> Rs3,
    py::array_t<double, py::array::f_style | py::array::forcecast> Ts,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    int natoms,
    int rep_size
) {
    auto bufCoords = coordinates.request();
    auto bufCharges = nuclear_charges.request();
    auto bufElements = elements.request();
    
    int n_elements = static_cast<int>(bufElements.size);
    int n_Rs2 = static_cast<int>(Rs2.request().size);
    int n_Rs3 = static_cast<int>(Rs3.request().size);
    int n_Ts = static_cast<int>(Ts.request().size);
    
    // Create output array (natoms, rep_size) - Fortran column-major
    std::vector<ssize_t> rep_shape = {natoms, rep_size};
    std::vector<ssize_t> rep_strides = {sizeof(double), static_cast<ssize_t>(sizeof(double) * natoms)};
    auto rep = py::array_t<double>(rep_shape, rep_strides);
    auto bufRep = rep.request();
    
    // Create output array (natoms, rep_size, natoms, 3) - Fortran column-major
    std::vector<ssize_t> grad_shape = {natoms, rep_size, natoms, 3};
    std::vector<ssize_t> grad_strides = {
        sizeof(double),
        static_cast<ssize_t>(sizeof(double) * natoms),
        static_cast<ssize_t>(sizeof(double) * natoms * rep_size),
        static_cast<ssize_t>(sizeof(double) * natoms * rep_size) * natoms
    };
    auto grad = py::array_t<double>(grad_shape, grad_strides);
    auto bufGrad = grad.request();
    
    fgenerate_acsf_and_gradients(
        static_cast<const double*>(bufCoords.ptr),
        static_cast<const int*>(bufCharges.ptr),
        static_cast<const int*>(bufElements.ptr),
        static_cast<const double*>(Rs2.request().ptr),
        static_cast<const double*>(Rs3.request().ptr),
        static_cast<const double*>(Ts.request().ptr),
        eta2, eta3, zeta, rcut, acut, natoms, rep_size,
        static_cast<double*>(bufRep.ptr),
        static_cast<double*>(bufGrad.ptr),
        n_elements, n_Rs2, n_Rs3, n_Ts
    );
    
    return std::make_tuple(rep, grad);
}

// Wrapper for fgenerate_fchl_acsf
py::array_t<double> generate_fchl_acsf_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    py::array_t<int, py::array::f_style | py::array::forcecast> nuclear_charges,
    py::array_t<int, py::array::f_style | py::array::forcecast> elements,
    py::array_t<double, py::array::f_style | py::array::forcecast> Rs2,
    py::array_t<double, py::array::f_style | py::array::forcecast> Rs3,
    py::array_t<double, py::array::f_style | py::array::forcecast> Ts,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    int natoms,
    int rep_size,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight
) {
    auto bufCoords = coordinates.request();
    auto bufCharges = nuclear_charges.request();
    auto bufElements = elements.request();
    
    int n_elements = static_cast<int>(bufElements.size);
    int n_Rs2 = static_cast<int>(Rs2.request().size);
    int n_Rs3 = static_cast<int>(Rs3.request().size);
    int n_Ts = static_cast<int>(Ts.request().size);
    
    // Create output array (natoms, rep_size) - Fortran column-major
    std::vector<ssize_t> shape = {natoms, rep_size};
    std::vector<ssize_t> strides = {static_cast<ssize_t>(sizeof(double)), static_cast<ssize_t>(sizeof(double) * natoms)};
    auto rep = py::array_t<double>(shape, strides);
    auto bufRep = rep.request();
    
    fgenerate_fchl_acsf(
        static_cast<const double*>(bufCoords.ptr),
        static_cast<const int*>(bufCharges.ptr),
        static_cast<const int*>(bufElements.ptr),
        static_cast<const double*>(Rs2.request().ptr),
        static_cast<const double*>(Rs3.request().ptr),
        static_cast<const double*>(Ts.request().ptr),
        eta2, eta3, zeta, rcut, acut, natoms, rep_size,
        two_body_decay, three_body_decay, three_body_weight,
        static_cast<double*>(bufRep.ptr),
        n_elements, n_Rs2, n_Rs3, n_Ts
    );
    
    return rep;
}

// Wrapper for fgenerate_fchl_acsf_and_gradients
std::tuple<py::array_t<double>, py::array_t<double>> generate_fchl_acsf_and_gradients_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    py::array_t<int, py::array::f_style | py::array::forcecast> nuclear_charges,
    py::array_t<int, py::array::f_style | py::array::forcecast> elements,
    py::array_t<double, py::array::f_style | py::array::forcecast> Rs2,
    py::array_t<double, py::array::f_style | py::array::forcecast> Rs3,
    py::array_t<double, py::array::f_style | py::array::forcecast> Ts,
    double eta2,
    double eta3,
    double zeta,
    double rcut,
    double acut,
    int natoms,
    int rep_size,
    double two_body_decay,
    double three_body_decay,
    double three_body_weight
) {
    auto bufCoords = coordinates.request();
    auto bufCharges = nuclear_charges.request();
    auto bufElements = elements.request();
    
    int n_elements = static_cast<int>(bufElements.size);
    int n_Rs2 = static_cast<int>(Rs2.request().size);
    int n_Rs3 = static_cast<int>(Rs3.request().size);
    int n_Ts = static_cast<int>(Ts.request().size);
    
    // Create output array (natoms, rep_size) - Fortran column-major
    std::vector<ssize_t> rep_shape = {natoms, rep_size};
    std::vector<ssize_t> rep_strides = {sizeof(double), static_cast<ssize_t>(sizeof(double) * natoms)};
    auto rep = py::array_t<double>(rep_shape, rep_strides);
    auto bufRep = rep.request();
    
    // Create output array (natoms, rep_size, natoms, 3) - Fortran column-major
    std::vector<ssize_t> grad_shape = {natoms, rep_size, natoms, 3};
    std::vector<ssize_t> grad_strides = {
        sizeof(double),
        static_cast<ssize_t>(sizeof(double) * natoms),
        static_cast<ssize_t>(sizeof(double) * natoms * rep_size),
        static_cast<ssize_t>(sizeof(double) * natoms * rep_size) * natoms
    };
    auto grad = py::array_t<double>(grad_shape, grad_strides);
    auto bufGrad = grad.request();
    
    fgenerate_fchl_acsf_and_gradients(
        static_cast<const double*>(bufCoords.ptr),
        static_cast<const int*>(bufCharges.ptr),
        static_cast<const int*>(bufElements.ptr),
        static_cast<const double*>(Rs2.request().ptr),
        static_cast<const double*>(Rs3.request().ptr),
        static_cast<const double*>(Ts.request().ptr),
        eta2, eta3, zeta, rcut, acut, natoms, rep_size,
        two_body_decay, three_body_decay, three_body_weight,
        static_cast<double*>(bufRep.ptr),
        static_cast<double*>(bufGrad.ptr),
        n_elements, n_Rs2, n_Rs3, n_Ts
    );
    
    return std::make_tuple(rep, grad);
}

PYBIND11_MODULE(_facsf, m) {
    m.doc() = "QMLlib ACSF/FCHL representation functions";

    m.def("fgenerate_acsf", &generate_acsf_wrapper,
        py::arg("coordinates"), py::arg("nuclear_charges"), py::arg("elements"),
        py::arg("Rs2"), py::arg("Rs3"), py::arg("Ts"),
        py::arg("eta2"), py::arg("eta3"), py::arg("zeta"),
        py::arg("rcut"), py::arg("acut"), py::arg("natoms"), py::arg("rep_size"),
        "Generate ACSF representation");

    m.def("fgenerate_acsf_and_gradients", &generate_acsf_and_gradients_wrapper,
        py::arg("coordinates"), py::arg("nuclear_charges"), py::arg("elements"),
        py::arg("Rs2"), py::arg("Rs3"), py::arg("Ts"),
        py::arg("eta2"), py::arg("eta3"), py::arg("zeta"),
        py::arg("rcut"), py::arg("acut"), py::arg("natoms"), py::arg("rep_size"),
        "Generate ACSF representation and gradients");

    m.def("fgenerate_fchl_acsf", &generate_fchl_acsf_wrapper,
        py::arg("coordinates"), py::arg("nuclear_charges"), py::arg("elements"),
        py::arg("Rs2"), py::arg("Rs3"), py::arg("Ts"),
        py::arg("eta2"), py::arg("eta3"), py::arg("zeta"),
        py::arg("rcut"), py::arg("acut"), py::arg("natoms"), py::arg("rep_size"),
        py::arg("two_body_decay"), py::arg("three_body_decay"), py::arg("three_body_weight"),
        "Generate FCHL-ACSF representation");

    m.def("fgenerate_fchl_acsf_and_gradients", &generate_fchl_acsf_and_gradients_wrapper,
        py::arg("coordinates"), py::arg("nuclear_charges"), py::arg("elements"),
        py::arg("Rs2"), py::arg("Rs3"), py::arg("Ts"),
        py::arg("eta2"), py::arg("eta3"), py::arg("zeta"),
        py::arg("rcut"), py::arg("acut"), py::arg("natoms"), py::arg("rep_size"),
        py::arg("two_body_decay"), py::arg("three_body_decay"), py::arg("three_body_weight"),
        "Generate FCHL-ACSF representation and gradients");
}
