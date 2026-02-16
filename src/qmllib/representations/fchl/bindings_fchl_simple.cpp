#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declarations for C-compatible Fortran functions
extern "C" {
    void fget_kernels_fchl(
        int nm1, int nm2, int na1, int nf1, int nn1, int na2, int nf2, int nn2,
        int np1, int np2, int npd1, int npd2, int npar1, int npar2,
        const double* x1, const double* x2, int verbose, const int* n1, const int* n2,
        const int* nneigh1, const int* nneigh2, int nsigmas,
        double t_width, double d_width, double cut_start, double cut_distance,
        int order, const double* pd, double distance_scale, double angular_scale,
        int alchemy, double two_body_power, double three_body_power,
        int kernel_idx, const double* parameters, double* kernels);
    
    void fget_symmetric_kernels_fchl(
        int nm1, int na1, int nf1, int nn1, int np1, int npd1, int npd2, int npar1, int npar2,
        const double* x1, int verbose, const int* n1, const int* nneigh1, int nsigmas,
        double t_width, double d_width, double cut_start, double cut_distance,
        int order, const double* pd, double distance_scale, double angular_scale,
        int alchemy, double two_body_power, double three_body_power,
        int kernel_idx, const double* parameters, double* kernels);
    
    void fget_global_symmetric_kernels_fchl(
        int nm1, int na1, int nf1, int nn1, int np1, int npd1, int npd2, int npar1, int npar2,
        const double* x1, int verbose, const int* n1, const int* nneigh1, int nsigmas,
        double t_width, double d_width, double cut_start, double cut_distance,
        int order, const double* pd, double distance_scale, double angular_scale,
        int alchemy, double two_body_power, double three_body_power,
        int kernel_idx, const double* parameters, double* kernels);
    
    void fget_global_kernels_fchl(
        int nm1, int nm2, int na1, int nf1, int nn1, int na2, int nf2, int nn2,
        int np1, int np2, int npd1, int npd2, int npar1, int npar2,
        const double* x1, const double* x2, int verbose, const int* n1, const int* n2,
        const int* nneigh1, const int* nneigh2, int nsigmas,
        double t_width, double d_width, double cut_start, double cut_distance,
        int order, const double* pd, double distance_scale, double angular_scale,
        int alchemy, double two_body_power, double three_body_power,
        int kernel_idx, const double* parameters, double* kernels);
    
    void fget_atomic_kernels_fchl(
        int na1, int nf1, int nn1, int na2, int nf2, int nn2,
        int np1, int np2, int npd1, int npd2, int npar1, int npar2,
        const double* x1, const double* x2, int verbose,
        const int* nneigh1, const int* nneigh2, int nsigmas,
        double t_width, double d_width, double cut_start, double cut_distance,
        int order, const double* pd, double distance_scale, double angular_scale,
        int alchemy, double two_body_power, double three_body_power,
        int kernel_idx, const double* parameters, double* kernels);
    
    void fget_atomic_symmetric_kernels_fchl(
        int na1, int nf1, int nn1, int np1, int npd1, int npd2, int npar1, int npar2,
        const double* x1, int verbose, const int* nneigh1, int nsigmas,
        double t_width, double d_width, double cut_start, double cut_distance,
        int order, const double* pd, double distance_scale, double angular_scale,
        int alchemy, double two_body_power, double three_body_power,
        int kernel_idx, const double* parameters, double* kernels);
}

py::array_t<double> fget_kernels_fchl_py(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2_in,
    bool verbose,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh2_in,
    int nm1, int nm2, int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double, py::array::f_style | py::array::forcecast> pd_in,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double, py::array::f_style | py::array::forcecast> parameters_in) {
    
    // Ensure Fortran-style arrays
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto x2 = py::array_t<double, py::array::f_style | py::array::forcecast>(x2_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto n2 = py::array_t<int, py::array::f_style | py::array::forcecast>(n2_in);
    auto nneigh1 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh1_in);
    auto nneigh2 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh2_in);
    auto pd = py::array_t<double, py::array::f_style | py::array::forcecast>(pd_in);
    auto parameters = py::array_t<double, py::array::f_style | py::array::forcecast>(parameters_in);
    
    auto b1 = x1.request(), b2 = x2.request();
    auto bn1 = n1.request(), bn2 = n2.request();
    auto bnn1 = nneigh1.request(), bnn2 = nneigh2.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    
    // Create output array - Fortran-style
    std::vector<ssize_t> shape = {nsigmas, nm1, nm2};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto result = py::array_t<double>(shape, strides);
    auto br = result.request();
    
    fget_kernels_fchl(
        nm1, nm2, 
        (int)b1.shape[1], (int)b1.shape[2], (int)b1.shape[3],  // na1, nf1, nn1
        (int)b2.shape[1], (int)b2.shape[2], (int)b2.shape[3],  // na2, nf2, nn2
        (int)bn1.shape[0], (int)bn2.shape[0],                   // np1, np2
        (int)bpd.shape[0], (int)bpd.shape[1],                   // npd1, npd2
        (int)bpar.shape[0], (int)bpar.shape[1],                 // npar1, npar2
        (double*)b1.ptr, (double*)b2.ptr, v,
        (int*)bn1.ptr, (int*)bn2.ptr,
        (int*)bnn1.ptr, (int*)bnn2.ptr, nsigmas,
        t_width, d_width, cut_start, cut_distance, order,
        (double*)bpd.ptr, distance_scale, angular_scale, a,
        two_body_power, three_body_power, kernel_idx,
        (double*)bpar.ptr, (double*)br.ptr);
    
    return result;
}

py::array_t<double> fget_symmetric_kernels_fchl_py(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    bool verbose,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh1_in,
    int nm1, int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double, py::array::f_style | py::array::forcecast> pd_in,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double, py::array::f_style | py::array::forcecast> parameters_in) {
    
    // Ensure Fortran-style arrays
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto nneigh1 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh1_in);
    auto pd = py::array_t<double, py::array::f_style | py::array::forcecast>(pd_in);
    auto parameters = py::array_t<double, py::array::f_style | py::array::forcecast>(parameters_in);
    
    auto b1 = x1.request();
    auto bn1 = n1.request(), bnn1 = nneigh1.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    
    // Create output array - Fortran-style
    std::vector<ssize_t> shape = {nsigmas, nm1, nm1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto result = py::array_t<double>(shape, strides);
    auto br = result.request();
    
    fget_symmetric_kernels_fchl(
        nm1,
        (int)b1.shape[1], (int)b1.shape[2], (int)b1.shape[3],  // na1, nf1, nn1
        (int)bn1.shape[0],                                       // np1
        (int)bpd.shape[0], (int)bpd.shape[1],                   // npd1, npd2
        (int)bpar.shape[0], (int)bpar.shape[1],                 // npar1, npar2
        (double*)b1.ptr, v,
        (int*)bn1.ptr, (int*)bnn1.ptr, nsigmas,
        t_width, d_width, cut_start, cut_distance, order,
        (double*)bpd.ptr, distance_scale, angular_scale, a,
        two_body_power, three_body_power, kernel_idx,
        (double*)bpar.ptr, (double*)br.ptr);
    
    return result;
}

py::array_t<double> fget_global_symmetric_kernels_fchl_py(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    bool verbose,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh1_in,
    int nm1, int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double, py::array::f_style | py::array::forcecast> pd_in,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double, py::array::f_style | py::array::forcecast> parameters_in) {
    
    // Ensure Fortran-style arrays
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto nneigh1 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh1_in);
    auto pd = py::array_t<double, py::array::f_style | py::array::forcecast>(pd_in);
    auto parameters = py::array_t<double, py::array::f_style | py::array::forcecast>(parameters_in);
    
    auto b1 = x1.request();
    auto bn1 = n1.request(), bnn1 = nneigh1.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    
    // Create output array - Fortran-style
    std::vector<ssize_t> shape = {nsigmas, nm1, nm1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto result = py::array_t<double>(shape, strides);
    auto br = result.request();
    
    fget_global_symmetric_kernels_fchl(
        nm1,
        (int)b1.shape[1], (int)b1.shape[2], (int)b1.shape[3],  // na1, nf1, nn1
        (int)bn1.shape[0],                                       // np1
        (int)bpd.shape[0], (int)bpd.shape[1],                   // npd1, npd2
        (int)bpar.shape[0], (int)bpar.shape[1],                 // npar1, npar2
        (double*)b1.ptr, v,
        (int*)bn1.ptr, (int*)bnn1.ptr, nsigmas,
        t_width, d_width, cut_start, cut_distance, order,
        (double*)bpd.ptr, distance_scale, angular_scale, a,
        two_body_power, three_body_power, kernel_idx,
        (double*)bpar.ptr, (double*)br.ptr);
    
    return result;
}

py::array_t<double> fget_global_kernels_fchl_py(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2_in,
    bool verbose,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> n2_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh2_in,
    int nm1, int nm2, int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double, py::array::f_style | py::array::forcecast> pd_in,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double, py::array::f_style | py::array::forcecast> parameters_in) {
    
    // Ensure Fortran-style arrays
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto x2 = py::array_t<double, py::array::f_style | py::array::forcecast>(x2_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto n2 = py::array_t<int, py::array::f_style | py::array::forcecast>(n2_in);
    auto nneigh1 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh1_in);
    auto nneigh2 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh2_in);
    auto pd = py::array_t<double, py::array::f_style | py::array::forcecast>(pd_in);
    auto parameters = py::array_t<double, py::array::f_style | py::array::forcecast>(parameters_in);
    
    auto b1 = x1.request(), b2 = x2.request();
    auto bn1 = n1.request(), bn2 = n2.request();
    auto bnn1 = nneigh1.request(), bnn2 = nneigh2.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    
    // Create output array - Fortran-style
    std::vector<ssize_t> shape = {nsigmas, nm1, nm2};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto result = py::array_t<double>(shape, strides);
    auto br = result.request();
    
    fget_global_kernels_fchl(
        nm1, nm2, 
        (int)b1.shape[1], (int)b1.shape[2], (int)b1.shape[3],  // na1, nf1, nn1
        (int)b2.shape[1], (int)b2.shape[2], (int)b2.shape[3],  // na2, nf2, nn2
        (int)bn1.shape[0], (int)bn2.shape[0],                   // np1, np2
        (int)bpd.shape[0], (int)bpd.shape[1],                   // npd1, npd2
        (int)bpar.shape[0], (int)bpar.shape[1],                 // npar1, npar2
        (double*)b1.ptr, (double*)b2.ptr, v,
        (int*)bn1.ptr, (int*)bn2.ptr,
        (int*)bnn1.ptr, (int*)bnn2.ptr, nsigmas,
        t_width, d_width, cut_start, cut_distance, order,
        (double*)bpd.ptr, distance_scale, angular_scale, a,
        two_body_power, three_body_power, kernel_idx,
        (double*)bpar.ptr, (double*)br.ptr);
    
    return result;
}

py::array_t<double> fget_atomic_kernels_fchl_py(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    py::array_t<double, py::array::f_style | py::array::forcecast> x2_in,
    bool verbose,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh2_in,
    int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double, py::array::f_style | py::array::forcecast> pd_in,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double, py::array::f_style | py::array::forcecast> parameters_in) {
    
    // Ensure Fortran-style arrays
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto x2 = py::array_t<double, py::array::f_style | py::array::forcecast>(x2_in);
    auto nneigh1 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh1_in);
    auto nneigh2 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh2_in);
    auto pd = py::array_t<double, py::array::f_style | py::array::forcecast>(pd_in);
    auto parameters = py::array_t<double, py::array::f_style | py::array::forcecast>(parameters_in);
    
    auto b1 = x1.request(), b2 = x2.request();
    auto bnn1 = nneigh1.request(), bnn2 = nneigh2.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    
    // Get dimensions - atomic kernels use 3D arrays (na, nf, nn)
    int na1 = (int)b1.shape[0];  // natoms1
    int na2 = (int)b2.shape[0];  // natoms2
    
    // Create output array - Fortran-style
    std::vector<ssize_t> shape = {nsigmas, na1, na2};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * na1};
    auto result = py::array_t<double>(shape, strides);
    auto br = result.request();
    
    fget_atomic_kernels_fchl(
        (int)b1.shape[0], (int)b1.shape[1], (int)b1.shape[2],  // na1, nf1, nn1
        (int)b2.shape[0], (int)b2.shape[1], (int)b2.shape[2],  // na2, nf2, nn2
        (int)bnn1.shape[0], (int)bnn2.shape[0],                 // np1, np2
        (int)bpd.shape[0], (int)bpd.shape[1],                   // npd1, npd2
        (int)bpar.shape[0], (int)bpar.shape[1],                 // npar1, npar2
        (double*)b1.ptr, (double*)b2.ptr, v,
        (int*)bnn1.ptr, (int*)bnn2.ptr, nsigmas,
        t_width, d_width, cut_start, cut_distance, order,
        (double*)bpd.ptr, distance_scale, angular_scale, a,
        two_body_power, three_body_power, kernel_idx,
        (double*)bpar.ptr, (double*)br.ptr);
    
    return result;
}

py::array_t<double> fget_atomic_symmetric_kernels_fchl_py(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in,
    bool verbose,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh1_in,
    int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double, py::array::f_style | py::array::forcecast> pd_in,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double, py::array::f_style | py::array::forcecast> parameters_in) {
    
    // Ensure Fortran-style arrays
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto nneigh1 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh1_in);
    auto pd = py::array_t<double, py::array::f_style | py::array::forcecast>(pd_in);
    auto parameters = py::array_t<double, py::array::f_style | py::array::forcecast>(parameters_in);
    
    auto b1 = x1.request();
    auto bnn1 = nneigh1.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    
    // Get dimensions - atomic kernels use 3D arrays (na, nf, nn)
    int na1 = (int)b1.shape[0];  // natoms1
    
    // Create output array - Fortran-style
    std::vector<ssize_t> shape = {nsigmas, na1, na1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * na1};
    auto result = py::array_t<double>(shape, strides);
    auto br = result.request();
    
    fget_atomic_symmetric_kernels_fchl(
        (int)b1.shape[0], (int)b1.shape[1], (int)b1.shape[2],  // na1, nf1, nn1
        (int)bnn1.shape[0],                                      // np1
        (int)bpd.shape[0], (int)bpd.shape[1],                   // npd1, npd2
        (int)bpar.shape[0], (int)bpar.shape[1],                 // npar1, npar2
        (double*)b1.ptr, v, (int*)bnn1.ptr, nsigmas,
        t_width, d_width, cut_start, cut_distance, order,
        (double*)bpd.ptr, distance_scale, angular_scale, a,
        two_body_power, three_body_power, kernel_idx,
        (double*)bpar.ptr, (double*)br.ptr);
    
    return result;
}

PYBIND11_MODULE(ffchl_module, m) {
    m.doc() = "QMLlib FCHL representation functions (simplified)";

    py::module_ kt = m.def_submodule("ffchl_kernel_types", "Kernel type constants");
    kt.attr("GAUSSIAN") = 1;
    kt.attr("LINEAR") = 2;
    kt.attr("POLYNOMIAL") = 3;
    kt.attr("SIGMOID") = 4;
    kt.attr("MULTIQUADRATIC") = 5;
    kt.attr("INV_MULTIQUADRATIC") = 6;
    kt.attr("BESSEL") = 7;
    kt.attr("L2") = 8;
    kt.attr("MATERN") = 9;
    kt.attr("CAUCHY") = 10;
    kt.attr("POLYNOMIAL2") = 11;
    
    // Lowercase aliases
    kt.attr("gaussian") = 1;
    kt.attr("linear") = 2;
    kt.attr("polynomial") = 3;
    kt.attr("sigmoid") = 4;
    kt.attr("multiquadratic") = 5;
    kt.attr("inv_multiquadratic") = 6;
    kt.attr("bessel") = 7;
    kt.attr("l2") = 8;
    kt.attr("matern") = 9;
    kt.attr("cauchy") = 10;
    kt.attr("polynomial2") = 11;

    m.def("fget_kernels_fchl", &fget_kernels_fchl_py,
        py::arg("x1"), py::arg("x2"), py::arg("verbose"),
        py::arg("n1"), py::arg("n2"), py::arg("nneigh1"), py::arg("nneigh2"),
        py::arg("nm1"), py::arg("nm2"), py::arg("nsigmas"),
        py::arg("t_width"), py::arg("d_width"), py::arg("cut_start"), py::arg("cut_distance"),
        py::arg("order"), py::arg("pd"), py::arg("distance_scale"), py::arg("angular_scale"),
        py::arg("alchemy"), py::arg("two_body_power"), py::arg("three_body_power"),
        py::arg("kernel_idx"), py::arg("parameters"));
    
    m.def("fget_symmetric_kernels_fchl", &fget_symmetric_kernels_fchl_py,
        py::arg("x1"), py::arg("verbose"), py::arg("n1"), py::arg("nneigh1"),
        py::arg("nm1"), py::arg("nsigmas"),
        py::arg("t_width"), py::arg("d_width"), py::arg("cut_start"), py::arg("cut_distance"),
        py::arg("order"), py::arg("pd"), py::arg("distance_scale"), py::arg("angular_scale"),
        py::arg("alchemy"), py::arg("two_body_power"), py::arg("three_body_power"),
        py::arg("kernel_idx"), py::arg("parameters"));
    
    m.def("fget_global_symmetric_kernels_fchl", &fget_global_symmetric_kernels_fchl_py,
        py::arg("x1"), py::arg("verbose"), py::arg("n1"), py::arg("nneigh1"),
        py::arg("nm1"), py::arg("nsigmas"),
        py::arg("t_width"), py::arg("d_width"), py::arg("cut_start"), py::arg("cut_distance"),
        py::arg("order"), py::arg("pd"), py::arg("distance_scale"), py::arg("angular_scale"),
        py::arg("alchemy"), py::arg("two_body_power"), py::arg("three_body_power"),
        py::arg("kernel_idx"), py::arg("parameters"));
    
    m.def("fget_global_kernels_fchl", &fget_global_kernels_fchl_py,
        py::arg("x1"), py::arg("x2"), py::arg("verbose"),
        py::arg("n1"), py::arg("n2"), py::arg("nneigh1"), py::arg("nneigh2"),
        py::arg("nm1"), py::arg("nm2"), py::arg("nsigmas"),
        py::arg("t_width"), py::arg("d_width"), py::arg("cut_start"), py::arg("cut_distance"),
        py::arg("order"), py::arg("pd"), py::arg("distance_scale"), py::arg("angular_scale"),
        py::arg("alchemy"), py::arg("two_body_power"), py::arg("three_body_power"),
        py::arg("kernel_idx"), py::arg("parameters"));
    
    m.def("fget_atomic_kernels_fchl", &fget_atomic_kernels_fchl_py,
        py::arg("x1"), py::arg("x2"), py::arg("verbose"),
        py::arg("nneigh1"), py::arg("nneigh2"), py::arg("nsigmas"),
        py::arg("t_width"), py::arg("d_width"), py::arg("cut_start"), py::arg("cut_distance"),
        py::arg("order"), py::arg("pd"), py::arg("distance_scale"), py::arg("angular_scale"),
        py::arg("alchemy"), py::arg("two_body_power"), py::arg("three_body_power"),
        py::arg("kernel_idx"), py::arg("parameters"));
    
    m.def("fget_atomic_symmetric_kernels_fchl", &fget_atomic_symmetric_kernels_fchl_py,
        py::arg("x1"), py::arg("verbose"), py::arg("nneigh1"), py::arg("nsigmas"),
        py::arg("t_width"), py::arg("d_width"), py::arg("cut_start"), py::arg("cut_distance"),
        py::arg("order"), py::arg("pd"), py::arg("distance_scale"), py::arg("angular_scale"),
        py::arg("alchemy"), py::arg("two_body_power"), py::arg("three_body_power"),
        py::arg("kernel_idx"), py::arg("parameters"));
}
