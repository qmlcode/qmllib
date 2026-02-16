#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Fortran function declarations
extern "C" {
    void fget_kernels_fchl_wrapper(
        int, int, int, int, int, int, int, int,
        int, int, int, int, int, int,
        int, int, int, int,
        const double*, const double*, int, const int*, const int*,
        const int*, const int*, int, int, int,
        double, double, double, double,
        int, const double*, double, double,
        int, double, double, int, const double*,
        double*);
    
    void fget_symmetric_kernels_fchl_wrapper(
        const double*, const int*, const int*, const int*, const int*, const int*,
        const double*, const double*, const double*, const double*,
        const int*, const double*, const double*, const double*,
        const int*, const double*, const double*, const int*, const double*,
        double*, const int*, const int*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*, const int*, const int*);
    
    void fget_global_symmetric_kernels_fchl_wrapper(
        const double*, const int*, const int*, const int*, const int*, const int*,
        const double*, const double*, const double*, const double*,
        const int*, const double*, const double*, const double*,
        const int*, const double*, const double*, const int*, const double*,
        double*, const int*, const int*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*, const int*, const int*);
    
    void fget_global_kernels_fchl_wrapper(
        const double*, const double*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*, const int*,
        const double*, const double*, const double*, const double*,
        const int*, const double*, const double*, const double*,
        const int*, const double*, const double*, const int*, const double*,
        double*, const int*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*, const int*, const int*, const int*, const int*);
        
    void fget_atomic_kernels_fchl_wrapper(
        const double*, const double*, const int*, const int*, const int*,
        const int*, const int*, const int*,
        const double*, const double*, const double*, const double*,
        const int*, const double*, const double*, const double*,
        const int*, const double*, const double*, const int*, const double*,
        double*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*);
    
    void fget_atomic_symmetric_kernels_fchl_wrapper(
        const double*, const int*, const int*, const int*, const int*,
        const double*, const double*, const double*, const double*,
        const int*, const double*, const double*, const double*,
        const int*, const double*, const double*, const int*, const double*,
        double*, const int*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*);
    
    void fget_atomic_local_kernels_fchl_wrapper(
        const double*, const double*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*, const int*,
        const int*, const double*, const double*, const double*, const double*,
        const int*, const double*, const double*, const double*,
        const int*, const double*, const double*, const int*, const double*,
        double*, const int*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*, const int*, const int*,
        const int*, const int*, const int*, const int*, const int*, const int*, const int*, const int*);
}

// Minimal wrapper - delegates to existing wrapper
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
    int d1[4] = {(int)b1.shape[0], (int)b1.shape[1], (int)b1.shape[2], (int)b1.shape[3]};
    int d2[4] = {(int)b2.shape[0], (int)b2.shape[1], (int)b2.shape[2], (int)b2.shape[3]};
    int dn1 = bn1.shape[0], dn2 = bn2.shape[0];
    int dnn1[2] = {(int)bnn1.shape[0], (int)bnn1.shape[1]};
    int dnn2[2] = {(int)bnn2.shape[0], (int)bnn2.shape[1]};
    int dpd[2] = {(int)bpd.shape[0], (int)bpd.shape[1]};
    int dpar[2] = {(int)bpar.shape[0], (int)bpar.shape[1]};
    
    // Create output array - Fortran-style
    std::vector<ssize_t> shape = {nsigmas, nm1, nm2};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto result = py::array_t<double>(shape, strides);
    auto br = result.request();
    
    fget_kernels_fchl_wrapper(
        d1[0], d1[1], d1[2], d1[3], d2[0], d2[1], d2[2], d2[3],
        dn1, dn2, dnn1[0], dnn1[1], dnn2[0], dnn2[1], dpd[0], dpd[1], dpar[0], dpar[1],
        (double*)b1.ptr, (double*)b2.ptr, v, (int*)bn1.ptr, (int*)bn2.ptr,
        (int*)bnn1.ptr, (int*)bnn2.ptr, nm1, nm2, nsigmas,
        t_width, d_width, cut_start, cut_distance,
        order, (double*)bpd.ptr, distance_scale, angular_scale,
        a, two_body_power, three_body_power, kernel_idx, (double*)bpar.ptr,
        (double*)br.ptr);
    
    return result;
}

// Symmetric version
py::array_t<double> fget_symmetric_kernels_fchl_py(
    py::array_t<double, py::array::f_style | py::array::forcecast> x1_in, bool verbose,
    py::array_t<int, py::array::f_style | py::array::forcecast> n1_in,
    py::array_t<int, py::array::f_style | py::array::forcecast> nneigh1_in,
    int nm1, int nsigmas, double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double, py::array::f_style | py::array::forcecast> pd_in,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power, int kernel_idx,
    py::array_t<double, py::array::f_style | py::array::forcecast> parameters_in) {
    
    // Ensure Fortran-style arrays
    auto x1 = py::array_t<double, py::array::f_style | py::array::forcecast>(x1_in);
    auto n1 = py::array_t<int, py::array::f_style | py::array::forcecast>(n1_in);
    auto nneigh1 = py::array_t<int, py::array::f_style | py::array::forcecast>(nneigh1_in);
    auto pd = py::array_t<double, py::array::f_style | py::array::forcecast>(pd_in);
    auto parameters = py::array_t<double, py::array::f_style | py::array::forcecast>(parameters_in);
    
    auto b1 = x1.request(), bn1 = n1.request(), bnn1 = nneigh1.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    int d1[4] = {(int)b1.shape[0], (int)b1.shape[1], (int)b1.shape[2], (int)b1.shape[3]};
    int dn1 = bn1.shape[0], dnn1[2] = {(int)bnn1.shape[0], (int)bnn1.shape[1]};
    int dpd[2] = {(int)bpd.shape[0], (int)bpd.shape[1]}, dpar[2] = {(int)bpar.shape[0], (int)bpar.shape[1]};
    
    // Create output array - Fortran-style
    std::vector<ssize_t> shape = {nsigmas, nm1, nm1};
    std::vector<ssize_t> strides = {sizeof(double), sizeof(double) * nsigmas, sizeof(double) * nsigmas * nm1};
    auto result = py::array_t<double>(shape, strides);
    auto br = result.request();
    
    fget_symmetric_kernels_fchl_wrapper(
        (double*)b1.ptr, &v, (int*)bn1.ptr, (int*)bnn1.ptr, &nm1, &nsigmas,
        &t_width, &d_width, &cut_start, &cut_distance, &order, (double*)bpd.ptr,
        &distance_scale, &angular_scale, &a, &two_body_power, &three_body_power,
        &kernel_idx, (double*)bpar.ptr, (double*)br.ptr,
        d1, d1+1, d1+2, d1+3, &dn1, dnn1, dnn1+1, dpd, dpd+1, dpar, dpar+1);
    
    return result;
}

// Global symmetric (same signature as symmetric)
py::array_t<double> fget_global_symmetric_kernels_fchl_py(
    py::array_t<double> x1, bool verbose, py::array_t<int> n1, py::array_t<int> nneigh1,
    int nm1, int nsigmas, double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double> pd, double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power, int kernel_idx, py::array_t<double> parameters) {
    
    auto b1 = x1.request(), bn1 = n1.request(), bnn1 = nneigh1.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    int d1[4] = {(int)b1.shape[0], (int)b1.shape[1], (int)b1.shape[2], (int)b1.shape[3]};
    int dn1 = bn1.shape[0], dnn1[2] = {(int)bnn1.shape[0], (int)bnn1.shape[1]};
    int dpd[2] = {(int)bpd.shape[0], (int)bpd.shape[1]}, dpar[2] = {(int)bpar.shape[0], (int)bpar.shape[1]};
    
    auto result = py::array_t<double>({nsigmas, nm1, nm1});
    auto br = result.request();
    
    fget_global_symmetric_kernels_fchl_wrapper(
        (double*)b1.ptr, &v, (int*)bn1.ptr, (int*)bnn1.ptr, &nm1, &nsigmas,
        &t_width, &d_width, &cut_start, &cut_distance, &order, (double*)bpd.ptr,
        &distance_scale, &angular_scale, &a, &two_body_power, &three_body_power,
        &kernel_idx, (double*)bpar.ptr, (double*)br.ptr,
        d1, d1+1, d1+2, d1+3, &dn1, dnn1, dnn1+1, dpd, dpd+1, dpar, dpar+1);
    
    return result;
}

// Global (same signature as regular)
py::array_t<double> fget_global_kernels_fchl_py(
    py::array_t<double> x1, py::array_t<double> x2, bool verbose,
    py::array_t<int> n1, py::array_t<int> n2,
    py::array_t<int> nneigh1, py::array_t<int> nneigh2,
    int nm1, int nm2, int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double> pd,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double> parameters) {
    
    auto b1 = x1.request(), b2 = x2.request();
    auto bn1 = n1.request(), bn2 = n2.request();
    auto bnn1 = nneigh1.request(), bnn2 = nneigh2.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    int d1[4] = {(int)b1.shape[0], (int)b1.shape[1], (int)b1.shape[2], (int)b1.shape[3]};
    int d2[4] = {(int)b2.shape[0], (int)b2.shape[1], (int)b2.shape[2], (int)b2.shape[3]};
    int dn1 = bn1.shape[0], dn2 = bn2.shape[0];
    int dnn1[2] = {(int)bnn1.shape[0], (int)bnn1.shape[1]};
    int dnn2[2] = {(int)bnn2.shape[0], (int)bnn2.shape[1]};
    int dpd[2] = {(int)bpd.shape[0], (int)bpd.shape[1]};
    int dpar[2] = {(int)bpar.shape[0], (int)bpar.shape[1]};
    
    auto result = py::array_t<double>({nsigmas, nm1, nm2});
    auto br = result.request();
    
    fget_global_kernels_fchl_wrapper(
        (double*)b1.ptr, (double*)b2.ptr, &v, (int*)bn1.ptr, (int*)bn2.ptr,
        (int*)bnn1.ptr, (int*)bnn2.ptr, &nm1, &nm2, &nsigmas,
        &t_width, &d_width, &cut_start, &cut_distance, &order, (double*)bpd.ptr,
        &distance_scale, &angular_scale, &a, &two_body_power, &three_body_power,
        &kernel_idx, (double*)bpar.ptr, (double*)br.ptr,
        d1, d1+1, d1+2, d1+3, d2, d2+1, d2+2, d2+3,
        &dn1, &dn2, dnn1, dnn1+1, dnn2, dnn2+1, dpd, dpd+1, dpar, dpar+1);
    
    return result;
}

// Atomic (3D arrays)
py::array_t<double> fget_atomic_kernels_fchl_py(
    py::array_t<double> x1, py::array_t<double> x2, bool verbose,
    py::array_t<int> nneigh1, py::array_t<int> nneigh2,
    int na1, int na2, int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double> pd,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double> parameters) {
    
    auto b1 = x1.request(), b2 = x2.request();
    auto bnn1 = nneigh1.request(), bnn2 = nneigh2.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    int d1[3] = {(int)b1.shape[0], (int)b1.shape[1], (int)b1.shape[2]};
    int d2[3] = {(int)b2.shape[0], (int)b2.shape[1], (int)b2.shape[2]};
    int dnn1 = bnn1.shape[0], dnn2 = bnn2.shape[0];
    int dpd[2] = {(int)bpd.shape[0], (int)bpd.shape[1]};
    int dpar[2] = {(int)bpar.shape[0], (int)bpar.shape[1]};
    
    auto result = py::array_t<double>({nsigmas, na1, na2});
    auto br = result.request();
    
    fget_atomic_kernels_fchl_wrapper(
        (double*)b1.ptr, (double*)b2.ptr, &v, (int*)bnn1.ptr, (int*)bnn2.ptr,
        &na1, &na2, &nsigmas,
        &t_width, &d_width, &cut_start, &cut_distance, &order, (double*)bpd.ptr,
        &distance_scale, &angular_scale, &a, &two_body_power, &three_body_power,
        &kernel_idx, (double*)bpar.ptr, (double*)br.ptr,
        d1, d1+1, d1+2, d2, d2+1, d2+2, &dnn1, &dnn2, dpd, dpd+1, dpar, dpar+1);
    
    return result;
}

// Atomic symmetric
py::array_t<double> fget_atomic_symmetric_kernels_fchl_py(
    py::array_t<double> x1, bool verbose, py::array_t<int> nneigh1,
    int na1, int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double> pd,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double> parameters) {
    
    auto b1 = x1.request(), bnn1 = nneigh1.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    int d1[3] = {(int)b1.shape[0], (int)b1.shape[1], (int)b1.shape[2]};
    int dnn1 = bnn1.shape[0];
    int dpd[2] = {(int)bpd.shape[0], (int)bpd.shape[1]};
    int dpar[2] = {(int)bpar.shape[0], (int)bpar.shape[1]};
    
    auto result = py::array_t<double>({nsigmas, na1, na1});
    auto br = result.request();
    
    fget_atomic_symmetric_kernels_fchl_wrapper(
        (double*)b1.ptr, &v, (int*)bnn1.ptr, &na1, &nsigmas,
        &t_width, &d_width, &cut_start, &cut_distance, &order, (double*)bpd.ptr,
        &distance_scale, &angular_scale, &a, &two_body_power, &three_body_power,
        &kernel_idx, (double*)bpar.ptr, (double*)br.ptr,
        d1, d1+1, d1+2, &dnn1, dpd, dpd+1, dpar, dpar+1);
    
    return result;
}

// Atomic local
py::array_t<double> fget_atomic_local_kernels_fchl_py(
    py::array_t<double> x1, py::array_t<double> x2, bool verbose,
    py::array_t<int> n1, py::array_t<int> n2,
    py::array_t<int> nneigh1, py::array_t<int> nneigh2,
    int nm1, int nm2, int na1, int nsigmas,
    double t_width, double d_width, double cut_start, double cut_distance,
    int order, py::array_t<double> pd,
    double distance_scale, double angular_scale, bool alchemy,
    double two_body_power, double three_body_power,
    int kernel_idx, py::array_t<double> parameters) {
    
    auto b1 = x1.request(), b2 = x2.request();
    auto bn1 = n1.request(), bn2 = n2.request();
    auto bnn1 = nneigh1.request(), bnn2 = nneigh2.request();
    auto bpd = pd.request(), bpar = parameters.request();
    
    int v = verbose ? 1 : 0, a = alchemy ? 1 : 0;
    int d1[4] = {(int)b1.shape[0], (int)b1.shape[1], (int)b1.shape[2], (int)b1.shape[3]};
    int d2[4] = {(int)b2.shape[0], (int)b2.shape[1], (int)b2.shape[2], (int)b2.shape[3]};
    int dn1 = bn1.shape[0], dn2 = bn2.shape[0];
    int dnn1[2] = {(int)bnn1.shape[0], (int)bnn1.shape[1]};
    int dnn2[2] = {(int)bnn2.shape[0], (int)bnn2.shape[1]};
    int dpd[2] = {(int)bpd.shape[0], (int)bpd.shape[1]};
    int dpar[2] = {(int)bpar.shape[0], (int)bpar.shape[1]};
    
    auto result = py::array_t<double>({nsigmas, nm1, na1});
    auto br = result.request();
    
    fget_atomic_local_kernels_fchl_wrapper(
        (double*)b1.ptr, (double*)b2.ptr, &v, (int*)bn1.ptr, (int*)bn2.ptr,
        (int*)bnn1.ptr, (int*)bnn2.ptr, &nm1, &nm2, &na1, &nsigmas,
        &t_width, &d_width, &cut_start, &cut_distance, &order, (double*)bpd.ptr,
        &distance_scale, &angular_scale, &a, &two_body_power, &three_body_power,
        &kernel_idx, (double*)bpar.ptr, (double*)br.ptr,
        d1, d1+1, d1+2, d1+3, d2, d2+1, d2+2, d2+3,
        &dn1, &dn2, dnn1, dnn1+1, dnn2, dnn2+1, dpd, dpd+1, dpar, dpar+1);
    
    return result;
}

PYBIND11_MODULE(ffchl_module, m) {
    m.doc() = "QMLlib FCHL representation functions";

    py::module_ kt = m.def_submodule("ffchl_kernel_types", "Kernel type constants");
    // Uppercase (for backward compatibility if needed)
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
    // Lowercase (as used in Python code)
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

#define PY_ARGS(f) py::arg(#f)
#define PY_ARGS2(f,g) py::arg(#f), py::arg(#g)

    m.def("fget_kernels_fchl", &fget_kernels_fchl_py,
        PY_ARGS2(x1,x2), PY_ARGS(verbose), PY_ARGS2(n1,n2), PY_ARGS2(nneigh1,nneigh2),
        PY_ARGS2(nm1,nm2), PY_ARGS(nsigmas), PY_ARGS2(t_width,d_width), 
        PY_ARGS2(cut_start,cut_distance), PY_ARGS2(order,pd),
        PY_ARGS2(distance_scale,angular_scale), PY_ARGS(alchemy),
        PY_ARGS2(two_body_power,three_body_power), PY_ARGS2(kernel_idx,parameters));

    m.def("fget_symmetric_kernels_fchl", &fget_symmetric_kernels_fchl_py,
        PY_ARGS(x1), PY_ARGS2(verbose,n1), PY_ARGS2(nneigh1,nm1), PY_ARGS(nsigmas),
        PY_ARGS2(t_width,d_width), PY_ARGS2(cut_start,cut_distance), PY_ARGS2(order,pd),
        PY_ARGS2(distance_scale,angular_scale), PY_ARGS(alchemy),
        PY_ARGS2(two_body_power,three_body_power), PY_ARGS2(kernel_idx,parameters));

    m.def("fget_global_symmetric_kernels_fchl", &fget_global_symmetric_kernels_fchl_py,
        PY_ARGS(x1), PY_ARGS2(verbose,n1), PY_ARGS2(nneigh1,nm1), PY_ARGS(nsigmas),
        PY_ARGS2(t_width,d_width), PY_ARGS2(cut_start,cut_distance), PY_ARGS2(order,pd),
        PY_ARGS2(distance_scale,angular_scale), PY_ARGS(alchemy),
        PY_ARGS2(two_body_power,three_body_power), PY_ARGS2(kernel_idx,parameters));

    m.def("fget_global_kernels_fchl", &fget_global_kernels_fchl_py,
        PY_ARGS2(x1,x2), PY_ARGS(verbose), PY_ARGS2(n1,n2), PY_ARGS2(nneigh1,nneigh2),
        PY_ARGS2(nm1,nm2), PY_ARGS(nsigmas), PY_ARGS2(t_width,d_width), 
        PY_ARGS2(cut_start,cut_distance), PY_ARGS2(order,pd),
        PY_ARGS2(distance_scale,angular_scale), PY_ARGS(alchemy),
        PY_ARGS2(two_body_power,three_body_power), PY_ARGS2(kernel_idx,parameters));

    m.def("fget_atomic_kernels_fchl", &fget_atomic_kernels_fchl_py,
        PY_ARGS2(x1,x2), PY_ARGS(verbose), PY_ARGS2(nneigh1,nneigh2),
        PY_ARGS2(na1,na2), PY_ARGS(nsigmas), PY_ARGS2(t_width,d_width), 
        PY_ARGS2(cut_start,cut_distance), PY_ARGS2(order,pd),
        PY_ARGS2(distance_scale,angular_scale), PY_ARGS(alchemy),
        PY_ARGS2(two_body_power,three_body_power), PY_ARGS2(kernel_idx,parameters));

    m.def("fget_atomic_symmetric_kernels_fchl", &fget_atomic_symmetric_kernels_fchl_py,
        PY_ARGS(x1), PY_ARGS2(verbose,nneigh1), PY_ARGS2(na1,nsigmas),
        PY_ARGS2(t_width,d_width), PY_ARGS2(cut_start,cut_distance), PY_ARGS2(order,pd),
        PY_ARGS2(distance_scale,angular_scale), PY_ARGS(alchemy),
        PY_ARGS2(two_body_power,three_body_power), PY_ARGS2(kernel_idx,parameters));

    m.def("fget_atomic_local_kernels_fchl", &fget_atomic_local_kernels_fchl_py,
        PY_ARGS2(x1,x2), PY_ARGS(verbose), PY_ARGS2(n1,n2), PY_ARGS2(nneigh1,nneigh2),
        PY_ARGS2(nm1,nm2), PY_ARGS2(na1,nsigmas), PY_ARGS2(t_width,d_width), 
        PY_ARGS2(cut_start,cut_distance), PY_ARGS2(order,pd),
        PY_ARGS2(distance_scale,angular_scale), PY_ARGS(alchemy),
        PY_ARGS2(two_body_power,three_body_power), PY_ARGS2(kernel_idx,parameters));
}
