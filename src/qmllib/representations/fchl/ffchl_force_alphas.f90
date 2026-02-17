subroutine fget_force_alphas_fchl(nm1, nm2, na1, nsigmas, &
       & n1_size, n2_size, nneigh1_size1, nneigh1_size2, &
       & nneigh2_size1, nneigh2_size2, nneigh2_size3, nneigh2_size4, nneigh2_size5, &
       & x1_size1, x1_size2, x1_size3, x1_size4, &
       & x2_size1, x2_size2, x2_size3, x2_size4, x2_size5, x2_size6, x2_size7, &
       & forces_size1, forces_size2, energies_size, &
       & pd_size1, pd_size2, parameters_size1, parameters_size2, &
       & x1, x2, verbose, forces, energies, n1, n2, nneigh1, nneigh2, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, dx, &
       & kernel_idx, parameters, llambda, alphas) bind(C, name="fget_force_alphas_fchl")

   use iso_c_binding
   use ffchl_module, only: scalar, get_angular_norm2, &
       & get_pmax, get_ksi, init_cosp_sinp, get_selfscalar, &
       & get_pmax_displaced, get_ksi_displaced, init_cosp_sinp_displaced, get_selfscalar_displaced
   use ffchl_kernels, only: kernel

   implicit none

   ! Dimensions (MUST be first with value attribute for bind(C))
   integer(c_int), intent(in), value :: nm1, nm2, na1, nsigmas
   integer(c_int), intent(in), value :: n1_size, n2_size
   integer(c_int), intent(in), value :: nneigh1_size1, nneigh1_size2
   integer(c_int), intent(in), value :: nneigh2_size1, nneigh2_size2, nneigh2_size3, nneigh2_size4, nneigh2_size5
   integer(c_int), intent(in), value :: x1_size1, x1_size2, x1_size3, x1_size4
   integer(c_int), intent(in), value :: x2_size1, x2_size2, x2_size3, x2_size4, x2_size5, x2_size6, x2_size7
   integer(c_int), intent(in), value :: forces_size1, forces_size2, energies_size
   integer(c_int), intent(in), value :: pd_size1, pd_size2
   integer(c_int), intent(in), value :: parameters_size1, parameters_size2

   ! fchl descriptors
   real(c_double), dimension(x1_size1, x1_size2, x1_size3, x1_size4), intent(in) :: x1
   real(c_double), dimension(x2_size1, x2_size2, x2_size3, x2_size4, x2_size5, x2_size6, x2_size7), intent(in) :: x2

   ! Whether to be verbose with output (C int, not logical)
   integer(c_int), intent(in), value :: verbose

   real(c_double), dimension(forces_size1, forces_size2), intent(in) :: forces
   real(c_double), dimension(energies_size), intent(in) :: energies

   ! List of numbers of atoms in each molecule
   integer(c_int), dimension(n1_size), intent(in) :: n1
   integer(c_int), dimension(n2_size), intent(in) :: n2

   ! Number of neighbors for each atom in each compound
   integer(c_int), dimension(nneigh1_size1, nneigh1_size2), intent(in) :: nneigh1
   integer(c_int), dimension(nneigh2_size1, nneigh2_size2, nneigh2_size3, nneigh2_size4, nneigh2_size5), intent(in) :: nneigh2

   ! Kernel parameters
   real(c_double), intent(in), value :: t_width, d_width, cut_start, cut_distance
   integer(c_int), intent(in), value :: order
   real(c_double), dimension(pd_size1, pd_size2), intent(in) :: pd
   real(c_double), intent(in), value :: distance_scale, angular_scale
   integer(c_int), intent(in), value :: alchemy
   real(c_double), intent(in), value :: two_body_power, three_body_power, dx
   integer(c_int), intent(in), value :: kernel_idx
   real(c_double), dimension(parameters_size1, parameters_size2), intent(in) :: parameters

   ! Regularization parameter
   real(c_double), intent(in), value :: llambda

   ! Resulting regression coefficients
   real(c_double), dimension(nsigmas, na1), intent(out) :: alphas

   ! Convert C integers to Fortran logicals
   logical :: verbose_logical, alchemy_logical

   ! Internal counters
   integer :: i, j, i2, j1, j2
   integer :: na, nb, ni, nj
   integer :: a, b, k

   ! Temporary variables necessary for parallelization
   double precision :: s12

   ! Pre-computed terms in the full distance matrix
   double precision, allocatable, dimension(:, :) :: self_scalar1
   double precision, allocatable, dimension(:, :, :, :, :) :: self_scalar2

   ! Pre-computed terms
   double precision, allocatable, dimension(:, :, :) :: ksi1
   double precision, allocatable, dimension(:, :, :, :, :, :) :: ksi2

   double precision, allocatable, dimension(:, :, :, :, :) :: sinp1
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp1

   double precision, allocatable, dimension(:, :, :, :, :, :, :) :: sinp2
   double precision, allocatable, dimension(:, :, :, :, :, :, :) :: cosp2

   ! Indexes for numerical differentiation
   integer :: xyz_pm2
   integer :: xyz2, pm2
   integer :: idx1, idx2
   integer :: idx1_start, idx2_start

   ! 1/(2*dx)
   double precision :: inv_2dx

   ! Max index in the periodic table
   integer :: pmax1
   integer :: pmax2

   ! Angular normalization constant
   double precision :: ang_norm2

   ! Max number of neighbors
   integer :: maxneigh1
   integer :: maxneigh2

   ! Info variable for BLAS/LAPACK calls
   integer :: info

   ! Feature vector multiplied by the kernel derivatives
   double precision, allocatable, dimension(:, :) :: y

   ! Numerical derivatives of kernel
   double precision, allocatable, dimension(:, :, :)  :: kernel_delta

   ! Scratch space for products of the kernel derivatives
   double precision, allocatable, dimension(:, :, :)  :: kernel_scratch

   ! Kernel between molecules and atom
   double precision, allocatable, dimension(:, :, :) :: kernel_ma

   ! Work kernel
   double precision, allocatable, dimension(:) :: ktmp

   ! Convert C integers to Fortran logicals
   verbose_logical = (verbose /= 0)
   alchemy_logical = (alchemy /= 0)

   allocate (ktmp(size(parameters, dim=1)))

   alphas = 0.0d0
   inv_2dx = 1.0d0/(2.0d0*dx)

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

   allocate (kernel_delta(na1, na1, nsigmas))
   allocate (y(na1, nsigmas))
   y = 0.0d0

   allocate (kernel_scratch(na1, na1, nsigmas))
   kernel_scratch = 0.0d0

   ! Calculate kernel derivatives and add to kernel matrix
   do xyz2 = 1, 3

      kernel_delta = 0.0d0

      !$OMP PARALLEL DO schedule(dynamic) PRIVATE(na,nb,xyz_pm2,s12), &
      !$OMP& PRIVATE(idx1,idx2,idx1_start,idx2_start)
      do a = 1, nm1
         na = n1(a)
         idx1_start = sum(n1(:a)) - na
         do j1 = 1, na
            idx1 = idx1_start + j1

            do b = 1, nm2
               nb = n2(b)
               idx2_start = (sum(n2(:b)) - nb)

               do pm2 = 1, 2
                  xyz_pm2 = 2*xyz2 + pm2 - 2
                  do i2 = 1, nb
                     idx2 = idx2_start + i2
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
                                    kernel_idx, parameters, ktmp)

                        if (pm2 == 2) then
                           kernel_delta(idx1, idx2, :) = kernel_delta(idx1, idx2, :) + ktmp*inv_2dx
                        else
                           kernel_delta(idx1, idx2, :) = kernel_delta(idx1, idx2, :) - ktmp*inv_2dx
                        end if

                     end do
                  end do
               end do
            end do
         end do
      end do
      !$OMP END PARALLEL do

      do k = 1, nsigmas
         call dsyrk("U", "N", na1, na1, 1.0d0, kernel_delta(1, 1, k), na1, &
            & 1.0d0, kernel_scratch(1, 1, k), na1)

         call dgemv("N", na1, na1, 1.0d0, kernel_delta(:, :, k), na1, &
             & forces(:, xyz2), 1, 1.0d0, y(:, k), 1)
      end do

   end do

   deallocate (kernel_delta)
   deallocate (self_scalar2)
   deallocate (ksi2)
   deallocate (cosp2)
   deallocate (sinp2)

   allocate (kernel_MA(nm1, na1, nsigmas))
   kernel_MA = 0.0d0

   !$OMP PARALLEL DO schedule(dynamic) PRIVATE(ni,nj,idx1,s12,idx1_start)
   do a = 1, nm1
      ni = n1(a)
      idx1_start = sum(n1(:a)) - ni
      do i = 1, ni

         idx1 = idx1_start + i

         do b = 1, nm1
            nj = n1(b)
            do j = 1, nj

               s12 = scalar(x1(a, i, :, :), x1(b, j, :, :), &
                   & nneigh1(a, i), nneigh1(b, j), ksi1(a, i, :), ksi1(b, j, :), &
                   & sinp1(a, i, :, :, :), sinp1(b, j, :, :, :), &
                   & cosp1(a, i, :, :, :), cosp1(b, j, :, :, :), &
                   & t_width, d_width, cut_distance, order, &
                   & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)

               ktmp = 0.0d0
               call kernel(self_scalar1(a, i), self_scalar1(b, j), s12, &
                           kernel_idx, parameters, ktmp)

               kernel_MA(b, idx1, :) = kernel_MA(b, idx1, :) + ktmp

            end do
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (self_scalar1)
   deallocate (ksi1)
   deallocate (cosp1)
   deallocate (sinp1)

   do k = 1, nsigmas
      call dsyrk("U", "T", na1, nm1, 1.0d0, kernel_MA(:, :, k), nm1, &
          & 1.0d0, kernel_scratch(:, :, k), na1)

      call dgemv("T", nm1, na1, 1.0d0, kernel_ma(:, :, k), nm1, &
                    & energies(:), 1, 1.0d0, y(:, k), 1)
   end do

   deallocate (kernel_ma)

   ! Add regularization
   do k = 1, nsigmas
      do i = 1, na1
         kernel_scratch(i, i, k) = kernel_scratch(i, i, k) + llambda
      end do
   end do

   alphas = 0.0d0

   ! Solve alphas using Cholesky decomposition
   do k = 1, nsigmas
      call dpotrf("U", na1, kernel_scratch(:, :, k), na1, info)
      if (info > 0) then
         write (*, *) "WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
         write (*, *) "WARNING: The", info, "-th leading order is not positive definite."
      else if (info < 0) then
         write (*, *) "WARNING: Error in LAPACK Cholesky decomposition DPOTRF()."
         write (*, *) "WARNING: The", -info, "-th argument had an illegal value."
      end if

      call dpotrs("U", na1, 1, kernel_scratch(:, :, k), na1, y(:, k), na1, info)
      if (info < 0) then
         write (*, *) "WARNING: Error in LAPACK Cholesky solver DPOTRS()."
         write (*, *) "WARNING: The", -info, "-th argument had an illegal value."
      end if

      alphas(k, :) = y(:, k)
   end do

   deallocate (y)
   deallocate (kernel_scratch)
   deallocate (ktmp)

end subroutine fget_force_alphas_fchl
