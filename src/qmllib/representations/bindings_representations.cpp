#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <vector>

namespace py = pybind11;

// Declare C ABI Fortran functions
extern "C" {
    void fgenerate_coulomb_matrix(const double* atomic_charges, const double* coordinates, 
                                  int natoms, int nmax, double* cm);
    void fgenerate_unsorted_coulomb_matrix(const double* atomic_charges, const double* coordinates,
                                          int natoms, int nmax, double* cm);
    void fgenerate_eigenvalue_coulomb_matrix(const double* atomic_charges, const double* coordinates,
                                            int natoms, int nmax, double* sorted_eigenvalues);
    void fgenerate_local_coulomb_matrix(const int* central_atom_indices, int central_natoms,
                                       const double* atomic_charges, const double* coordinates,
                                       int natoms, int nmax, double* cent_cutoff, double* cent_decay,
                                       double* int_cutoff, double* int_decay, double* cm);
    void fgenerate_atomic_coulomb_matrix(const int* central_atom_indices, int central_natoms,
                                        const double* atomic_charges, const double* coordinates,
                                        int natoms, int nmax, double* cent_cutoff, double* cent_decay,
                                        double* int_cutoff, double* int_decay, double* cm);
    void fgenerate_bob(const double* atomic_charges, const double* coordinates,
                      const int* nuclear_charges, const int* id, const int* nmax,
                      int nid, int ncm, int natoms, double* cm);
}

// Wrapper for fgenerate_coulomb_matrix
py::array_t<double> generate_coulomb_matrix_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> atomic_charges,
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    int nmax
) {
    auto bufAC = atomic_charges.request();
    auto bufCoord = coordinates.request();

    if (bufAC.ndim != 1) {
        throw std::runtime_error("atomic_charges must be 1D array");
    }
    if (bufCoord.ndim != 2 || bufCoord.shape[1] != 3) {
        throw std::runtime_error("coordinates must be (N,3) array");
    }

    int natoms = static_cast<int>(bufAC.shape[0]);
    int cm_size = (nmax + 1) * nmax / 2;

    auto cm = py::array_t<double>(cm_size);
    auto bufCM = cm.request();

    fgenerate_coulomb_matrix(
        static_cast<const double*>(bufAC.ptr),
        static_cast<const double*>(bufCoord.ptr),
        natoms, nmax,
        static_cast<double*>(bufCM.ptr)
    );

    return cm;
}

// Wrapper for fgenerate_unsorted_coulomb_matrix
py::array_t<double> generate_unsorted_coulomb_matrix_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> atomic_charges,
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    int nmax
) {
    auto bufAC = atomic_charges.request();
    auto bufCoord = coordinates.request();

    if (bufAC.ndim != 1) {
        throw std::runtime_error("atomic_charges must be 1D array");
    }
    if (bufCoord.ndim != 2 || bufCoord.shape[1] != 3) {
        throw std::runtime_error("coordinates must be (N,3) array");
    }

    int natoms = static_cast<int>(bufAC.shape[0]);
    int cm_size = (nmax + 1) * nmax / 2;

    auto cm = py::array_t<double>(cm_size);
    auto bufCM = cm.request();

    fgenerate_unsorted_coulomb_matrix(
        static_cast<const double*>(bufAC.ptr),
        static_cast<const double*>(bufCoord.ptr),
        natoms, nmax,
        static_cast<double*>(bufCM.ptr)
    );

    return cm;
}

// Wrapper for fgenerate_eigenvalue_coulomb_matrix
py::array_t<double> generate_eigenvalue_coulomb_matrix_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> atomic_charges,
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    int nmax
) {
    auto bufAC = atomic_charges.request();
    auto bufCoord = coordinates.request();

    if (bufAC.ndim != 1) {
        throw std::runtime_error("atomic_charges must be 1D array");
    }
    if (bufCoord.ndim != 2 || bufCoord.shape[1] != 3) {
        throw std::runtime_error("coordinates must be (N,3) array");
    }

    int natoms = static_cast<int>(bufAC.shape[0]);

    auto eigenvalues = py::array_t<double>(nmax);
    auto bufEV = eigenvalues.request();

    fgenerate_eigenvalue_coulomb_matrix(
        static_cast<const double*>(bufAC.ptr),
        static_cast<const double*>(bufCoord.ptr),
        natoms, nmax,
        static_cast<double*>(bufEV.ptr)
    );

    return eigenvalues;
}

// Wrapper for fgenerate_local_coulomb_matrix
py::array_t<double> generate_local_coulomb_matrix_wrapper(
    py::array_t<int, py::array::f_style | py::array::forcecast> central_atom_indices,
    int central_natoms,
    py::array_t<double, py::array::f_style | py::array::forcecast> atomic_charges,
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    int natoms,
    int nmax,
    double cent_cutoff,
    double cent_decay,
    double int_cutoff,
    double int_decay
) {
    auto bufIndices = central_atom_indices.request();
    auto bufAC = atomic_charges.request();
    auto bufCoord = coordinates.request();

    int cm_size = (nmax + 1) * nmax / 2;
    // Create Fortran-style (column-major) array with proper strides
    std::vector<ssize_t> shape = {central_natoms, cm_size};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * central_natoms};
    auto cm = py::array_t<double>(shape, strides);
    auto bufCM = cm.request();

    // Make copies of cutoff parameters since Fortran modifies them
    double cent_cutoff_copy = cent_cutoff;
    double cent_decay_copy = cent_decay;
    double int_cutoff_copy = int_cutoff;
    double int_decay_copy = int_decay;

    fgenerate_local_coulomb_matrix(
        static_cast<const int*>(bufIndices.ptr),
        central_natoms,
        static_cast<const double*>(bufAC.ptr),
        static_cast<const double*>(bufCoord.ptr),
        natoms, nmax,
        &cent_cutoff_copy, &cent_decay_copy,
        &int_cutoff_copy, &int_decay_copy,
        static_cast<double*>(bufCM.ptr)
    );

    return cm;
}

// Wrapper for fgenerate_atomic_coulomb_matrix
py::array_t<double> generate_atomic_coulomb_matrix_wrapper(
    py::array_t<int, py::array::f_style | py::array::forcecast> central_atom_indices,
    int central_natoms,
    py::array_t<double, py::array::f_style | py::array::forcecast> atomic_charges,
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    int natoms,
    int nmax,
    double cent_cutoff,
    double cent_decay,
    double int_cutoff,
    double int_decay
) {
    auto bufIndices = central_atom_indices.request();
    auto bufAC = atomic_charges.request();
    auto bufCoord = coordinates.request();

    int cm_size = (nmax + 1) * nmax / 2;
    // Create Fortran-style (column-major) array with proper strides
    std::vector<ssize_t> shape = {central_natoms, cm_size};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * central_natoms};
    auto cm = py::array_t<double>(shape, strides);
    auto bufCM = cm.request();

    // Make copies of cutoff parameters since Fortran modifies them
    double cent_cutoff_copy = cent_cutoff;
    double cent_decay_copy = cent_decay;
    double int_cutoff_copy = int_cutoff;
    double int_decay_copy = int_decay;

    fgenerate_atomic_coulomb_matrix(
        static_cast<const int*>(bufIndices.ptr),
        central_natoms,
        static_cast<const double*>(bufAC.ptr),
        static_cast<const double*>(bufCoord.ptr),
        natoms, nmax,
        &cent_cutoff_copy, &cent_decay_copy,
        &int_cutoff_copy, &int_decay_copy,
        static_cast<double*>(bufCM.ptr)
    );

    return cm;
}

// Wrapper for fgenerate_bob
py::array_t<double> generate_bob_wrapper(
    py::array_t<double, py::array::f_style | py::array::forcecast> atomic_charges,
    py::array_t<double, py::array::f_style | py::array::forcecast> coordinates,
    py::array_t<int, py::array::f_style | py::array::forcecast> nuclear_charges,
    py::array_t<int, py::array::f_style | py::array::forcecast> id,
    py::array_t<int, py::array::f_style | py::array::forcecast> nmax,
    int ncm
) {
    auto bufAC = atomic_charges.request();
    auto bufCoord = coordinates.request();
    auto bufNC = nuclear_charges.request();
    auto bufID = id.request();
    auto bufNmax = nmax.request();

    int natoms = static_cast<int>(bufAC.shape[0]);
    int nid = static_cast<int>(bufID.shape[0]);

    auto cm = py::array_t<double>(ncm);
    auto bufCM = cm.request();

    fgenerate_bob(
        static_cast<const double*>(bufAC.ptr),
        static_cast<const double*>(bufCoord.ptr),
        static_cast<const int*>(bufNC.ptr),
        static_cast<const int*>(bufID.ptr),
        static_cast<const int*>(bufNmax.ptr),
        nid, ncm, natoms,
        static_cast<double*>(bufCM.ptr)
    );

    return cm;
}

PYBIND11_MODULE(_representations, m) {
    m.doc() = "qmllib: Fortran representation routines with pybind11 bindings";
    
    m.def("fgenerate_coulomb_matrix", &generate_coulomb_matrix_wrapper,
          py::arg("atomic_charges"), py::arg("coordinates"), py::arg("nmax"),
          "Generate Coulomb Matrix representation");
    
    m.def("fgenerate_unsorted_coulomb_matrix", &generate_unsorted_coulomb_matrix_wrapper,
          py::arg("atomic_charges"), py::arg("coordinates"), py::arg("nmax"),
          "Generate unsorted Coulomb Matrix representation");
    
    m.def("fgenerate_eigenvalue_coulomb_matrix", &generate_eigenvalue_coulomb_matrix_wrapper,
          py::arg("atomic_charges"), py::arg("coordinates"), py::arg("nmax"),
          "Generate eigenvalue Coulomb Matrix representation");
    
    m.def("fgenerate_local_coulomb_matrix", &generate_local_coulomb_matrix_wrapper,
          py::arg("central_atom_indices"), py::arg("central_natoms"),
          py::arg("atomic_charges"), py::arg("coordinates"),
          py::arg("natoms"), py::arg("nmax"),
          py::arg("cent_cutoff"), py::arg("cent_decay"),
          py::arg("int_cutoff"), py::arg("int_decay"),
          "Generate local Coulomb Matrix representation");
    
    m.def("fgenerate_atomic_coulomb_matrix", &generate_atomic_coulomb_matrix_wrapper,
          py::arg("central_atom_indices"), py::arg("central_natoms"),
          py::arg("atomic_charges"), py::arg("coordinates"),
          py::arg("natoms"), py::arg("nmax"),
          py::arg("cent_cutoff"), py::arg("cent_decay"),
          py::arg("int_cutoff"), py::arg("int_decay"),
          "Generate atomic Coulomb Matrix representation");
    
    m.def("fgenerate_bob", &generate_bob_wrapper,
          py::arg("atomic_charges"), py::arg("coordinates"),
          py::arg("nuclear_charges"), py::arg("id"),
          py::arg("nmax"), py::arg("ncm"),
          "Generate Bag of Bonds representation");
}
