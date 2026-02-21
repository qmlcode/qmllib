subroutine fget_local_gradient_kernels_fchl(nm1, na1, nf1, nn1, nm2, nxyz2, npm2, na2i, na2j, nf2, nn2, &
       & np1, np2, nngh1_1, nngh1_2, nngh2_1, nngh2_2, nngh2_3, nngh2_4, nngh2_5, &
       & npd1, npd2, npar1, npar2, &
       & x1, x2, verbose, n1, n2, nneigh1, nneigh2, &
       & naq2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
       & kernel_idx, parameters, kernels) bind(C, name="fget_local_gradient_kernels_fchl")

   use iso_c_binding
   use ffchl_module, only: scalar, get_angular_norm2, &
       & get_pmax, get_ksi, init_cosp_sinp, get_selfscalar, &
       & get_pmax_displaced, get_ksi_displaced, init_cosp_sinp_displaced, get_selfscalar_displaced

   use ffchl_kernels, only: kernel

   implicit none

   ! Dimensions (must come first for bind(C))
   integer(c_int), intent(in), value :: nm1, na1, nf1, nn1      ! x1 dimensions: nmol, natoms, nfeatures, nneigh
   integer(c_int), intent(in), value :: nm2, nxyz2, npm2, na2i, na2j, nf2, nn2  ! x2 dimensions: nmol, 3, 2, natoms_i, natoms_j, nfeatures, nneigh
   integer(c_int), intent(in), value :: np1, np2                 ! n1, n2 dimensions
   integer(c_int), intent(in), value :: nngh1_1, nngh1_2         ! nneigh1 dimensions
   integer(c_int), intent(in), value :: nngh2_1, nngh2_2, nngh2_3, nngh2_4, nngh2_5  ! nneigh2 dimensions (5D)
   integer(c_int), intent(in), value :: npd1, npd2               ! pd dimensions
   integer(c_int), intent(in), value :: npar1, npar2             ! parameters dimensions
   integer(c_int), intent(in), value :: naq2                     ! Total number of force components
   integer(c_int), intent(in), value :: nsigmas                  ! Number of kernels
   integer(c_int), intent(in), value :: order                    ! Truncation order
   integer(c_int), intent(in), value :: kernel_idx               ! Kernel ID

   ! fchl descriptors for the training set, format (nm1,maxatoms,5,maxneighbors)
   real(c_double), dimension(nm1, na1, nf1, nn1), intent(in) :: x1

   ! fchl descriptors for the training set, format (nm2,3,2,maxatoms,maxatoms,5,maxneighbors)
   real(c_double), dimension(nm2, nxyz2, npm2, na2i, na2j, nf2, nn2), intent(in) :: x2

   ! Whether to be verbose with output
   integer(c_int), intent(in), value :: verbose

   ! List of numbers of atoms in each molecule
   integer(c_int), dimension(np1), intent(in) :: n1
   integer(c_int), dimension(np2), intent(in) :: n2

   ! Number of neighbors for each atom in each compound
   integer(c_int), dimension(nngh1_1, nngh1_2), intent(in) :: nneigh1
   integer(c_int), dimension(nngh2_1, nngh2_2, nngh2_3, nngh2_4, nngh2_5), intent(in) :: nneigh2

   real(c_double), intent(in), value :: t_width
   real(c_double), intent(in), value :: d_width
   real(c_double), intent(in), value :: cut_start
   real(c_double), intent(in), value :: cut_distance
   real(c_double), intent(in), value :: distance_scale
   real(c_double), intent(in), value :: angular_scale
   real(c_double), intent(in), value :: two_body_power
   real(c_double), intent(in), value :: three_body_power
   real(c_double), intent(in), value :: dx

   ! Switch alchemy on or off
   integer(c_int), intent(in), value :: alchemy

   ! Periodic table distance matrix
   real(c_double), dimension(npd1, npd2), intent(in) :: pd

   ! Kernel parameters
   real(c_double), dimension(npar1, npar2), intent(in) :: parameters

   ! Resulting kernel matrix
   real(c_double), dimension(nsigmas, nm1, naq2), intent(out) :: kernels

   ! Internal counters
   integer :: i2, j1, j2
   integer :: na, nb
   integer :: a, b

   ! Convert C int to Fortran logical
   logical :: verbose_logical, alchemy_logical

   ! Temporary variables necessary for parallelization
   double precision :: s12

   ! Pre-computed terms in the full distance matrix
   double precision, allocatable, dimension(:, :) :: self_scalar1
   double precision, allocatable, dimension(:, :, :, :, :) :: self_scalar2

   ! Pre-computed two-body weights
   double precision, allocatable, dimension(:, :, :) :: ksi1
   double precision, allocatable, dimension(:, :, :, :, :, :) :: ksi2

   ! Pre-computed terms for the Fourier expansion of the three-body term
   double precision, allocatable, dimension(:, :, :, :, :) :: sinp1
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp1

   ! Pre-computed terms for the Fourier expansion of the three-body term
   double precision, allocatable, dimension(:, :, :, :, :, :, :) :: sinp2
   double precision, allocatable, dimension(:, :, :, :, :, :, :) :: cosp2

   ! Indexes for numerical differentiation
   integer :: idx1, idx2
   integer :: xyz_pm2
   integer :: xyz2, pm2

   ! Max index in the periodic table
   integer :: pmax1
   integer :: pmax2

   ! Angular normalization constant
   double precision :: ang_norm2

   ! Max number of neighbors
   integer :: maxneigh1
   integer :: maxneigh2

   ! Work kernel
   double precision, allocatable, dimension(:) :: ktmp
   allocate (ktmp(size(parameters, dim=1)))

   kernels = 0.0d0

   ! Angular normalization constant
   ang_norm2 = get_angular_norm2(t_width)

   ! Max number of neighbors in the representations
   maxneigh1 = maxval(nneigh1)
   maxneigh2 = maxval(nneigh2)

   ! pmax = max nuclear charge
   pmax1 = get_pmax(x1, n1)
   pmax2 = get_pmax_displaced(x2, n2)

   ! Get two-body weight function
   allocate (ksi1(size(x1, dim=1), maxval(n1), maxval(nneigh1)))
   allocate (ksi2(size(x2, dim=1), 3, size(x2, dim=3), maxval(n2), maxval(n2), maxval(nneigh2)))

   call get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose_logical, ksi1)
   call get_ksi_displaced(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose_logical, ksi2)

   ! ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
   ! ksi2 = get_ksi_displaced(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)

   ! Allocate three-body Fourier terms
   allocate (cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
   allocate (sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))

   ! Initialize and pre-calculate three-body Fourier terms
   call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start, cut_distance, &
       & cosp1, sinp1, verbose_logical)

   ! Allocate three-body Fourier terms
   allocate (cosp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))
   allocate (sinp2(nm2, 3*2, maxval(n2), maxval(n2), pmax2, order, maxneigh2))

   ! Initialize and pre-calculate three-body Fourier terms
   call init_cosp_sinp_displaced(x2, n2, nneigh2, three_body_power, order, cut_start, &
       & cut_distance, cosp2, sinp2, verbose_logical)

   ! Pre-calculate self-scalar terms
   allocate (self_scalar1(nm1, maxval(n1)))
   allocate (self_scalar2(nm2, 3, size(x2, dim=3), maxval(n2), maxval(n2)))
   call get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
        & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy_logical, verbose_logical, self_scalar1)
   call get_selfscalar_displaced(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, &
   & d_width, cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy_logical, verbose_logical, self_scalar2)

   ! Pre-calculate self-scalar terms
   ! self_scalar2 = get_selfscalar_displaced(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, &
   ! & d_width, cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose)
   ! self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
   !      & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose)

    !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,s12,ktmp,j1,j2,i2,a,b),&
    !$OMP& PRIVATE(idx1,idx2,xyz2,pm2)
   do a = 1, nm1
      na = n1(a)
      idx1 = a
      do j1 = 1, na

         do b = 1, nm2
            nb = n2(b)
            do xyz2 = 1, 3
            do pm2 = 1, 2
               xyz_pm2 = 2*xyz2 + pm2 - 2
               do i2 = 1, nb
                  idx2 = (sum(n2(:b)) - n2(b))*3 + (i2 - 1)*3 + xyz2
                  do j2 = 1, nb

                     s12 = scalar(x1(a, j1, :, :), x2(b, xyz2, pm2, i2, j2, :, :), &
                         & nneigh1(a, j1), nneigh2(b, xyz2, pm2, i2, j2), &
                         & ksi1(a, j1, :), ksi2(b, xyz2, pm2, i2, j2, :), &
                         & sinp1(a, j1, :, :, :), sinp2(b, xyz_pm2, i2, j2, :, :, :), &
                         & cosp1(a, j1, :, :, :), cosp2(b, xyz_pm2, i2, j2, :, :, :), &
                         & t_width, d_width, cut_distance, order, &
                         & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)

                     ktmp = 0.0d0
                     call kernel(self_scalar1(a, j1), self_scalar2(b, xyz2, pm2, i2, j2), s12, &
                     & kernel_idx, parameters, ktmp)

                     !$OMP CRITICAL
                     if (pm2 == 2) then

                        kernels(:, idx1, idx2) = kernels(:, idx1, idx2) + ktmp

                        ! kernels(:, idx1, idx2) = kernels(:, idx1, idx2) &
                        !     & + kernel(self_scalar1(a, j1), self_scalar2(b, xyz2, pm2, i2, j2), s12, &
                        !     & kernel_idx, parameters)
                     else
                        kernels(:, idx1, idx2) = kernels(:, idx1, idx2) - ktmp

                        ! kernels(:, idx1, idx2) = kernels(:, idx1, idx2) &
                        !     & - kernel(self_scalar1(a, j1), self_scalar2(b, xyz2, pm2, i2, j2), s12, &
                        !     & kernel_idx, parameters)
                     end if
                     !$OMP END CRITICAL

                  end do
               end do
            end do
            end do
         end do
      end do
   end do
   !$OMP END PARALLEL do

   kernels = kernels/(2*dx)

   deallocate (ktmp)
   deallocate (ksi1)
   deallocate (ksi2)
   deallocate (cosp1)
   deallocate (sinp1)
   deallocate (cosp2)
   deallocate (sinp2)
   deallocate (self_scalar1)
   deallocate (self_scalar2)

end subroutine fget_local_gradient_kernels_fchl
