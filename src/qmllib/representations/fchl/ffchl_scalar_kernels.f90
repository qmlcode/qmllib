subroutine fget_kernels_fchl(nm1, nm2, na1, nf1, nn1, na2, nf2, nn2, &
       & np1, np2, npd1, npd2, npar1, npar2, &
       & x1, x2, verbose, n1, n2, nneigh1, nneigh2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, &
       & kernel_idx, parameters, kernels) bind(C, name="fget_kernels_fchl")

   use iso_c_binding
   use ffchl_module, only: scalar, get_angular_norm2, get_pmax, get_ksi, init_cosp_sinp, get_selfscalar
   use ffchl_kernels, only: kernel

   implicit none

   ! Dimensions (must come first for bind(C))
   integer(c_int), intent(in), value :: nm1, nm2      ! Number of molecules
   integer(c_int), intent(in), value :: na1, nf1, nn1 ! x1 dimensions: natoms, nfeatures, nneighbors
   integer(c_int), intent(in), value :: na2, nf2, nn2 ! x2 dimensions
   integer(c_int), intent(in), value :: np1, np2      ! n1, n2 dimensions
   integer(c_int), intent(in), value :: npd1, npd2    ! pd dimensions
   integer(c_int), intent(in), value :: npar1, npar2  ! parameters dimensions
   integer(c_int), intent(in), value :: nsigmas       ! Number of sigmas
   integer(c_int), intent(in), value :: order         ! Truncation order for Fourier terms
   integer(c_int), intent(in), value :: kernel_idx    ! Kernel ID

   ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
   real(c_double), dimension(nm1, na1, nf1, nn1), intent(in) :: x1
   real(c_double), dimension(nm2, na2, nf2, nn2), intent(in) :: x2

   ! Whether to be verbose with output (int instead of logical for C compat)
   integer(c_int), intent(in), value :: verbose

   ! List of numbers of atoms in each molecule
   integer(c_int), dimension(np1), intent(in) :: n1
   integer(c_int), dimension(np2), intent(in) :: n2

   ! Number of neighbors for each atom in each compound
   integer(c_int), dimension(nm1, na1), intent(in) :: nneigh1
   integer(c_int), dimension(nm2, na2), intent(in) :: nneigh2

   ! Angular Gaussian width
   real(c_double), intent(in), value :: t_width

   ! Distance Gaussian width
   real(c_double), intent(in), value :: d_width

   ! Fraction of cut_distance at which cut-off starts
   real(c_double), intent(in), value :: cut_start
   real(c_double), intent(in), value :: cut_distance

   ! Periodic table distance matrix
   real(c_double), dimension(npd1, npd2), intent(in) :: pd

   ! Scaling for angular and distance terms
   real(c_double), intent(in), value :: distance_scale
   real(c_double), intent(in), value :: angular_scale

   ! Switch alchemy on or off (int instead of logical for C compat)
   integer(c_int), intent(in), value :: alchemy

   ! Decaying power laws for two- and three-body terms
   real(c_double), intent(in), value :: two_body_power
   real(c_double), intent(in), value :: three_body_power

   ! Kernel parameters
   real(c_double), dimension(npar1, npar2), intent(in) :: parameters

   ! Resulting alpha vector
   real(c_double), dimension(nsigmas, nm1, nm2), intent(out) :: kernels

   ! Internal counters
   integer :: i, j
   integer :: ni, nj
   integer :: a, b

   ! Temporary variables necessary for parallelization
   double precision :: s12

   ! Pre-computed terms in the full distance matrix
   double precision, allocatable, dimension(:, :) :: self_scalar1
   double precision, allocatable, dimension(:, :) :: self_scalar2

   ! Pre-computed two-body weights
   double precision, allocatable, dimension(:, :, :) :: ksi1
   double precision, allocatable, dimension(:, :, :) :: ksi2

   ! Pre-computed terms for the Fourier expansion of the three-body term
   double precision, allocatable, dimension(:, :, :, :, :) :: sinp1
   double precision, allocatable, dimension(:, :, :, :, :) :: sinp2
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp1
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp2

   ! Max index in the periodic table
   integer :: pmax1
   integer :: pmax2

   ! Angular normalization constant
   double precision :: ang_norm2

   ! Max number of neighbors
   integer :: maxneigh1
   integer :: maxneigh2

   ! Work kernel space
   integer :: n
   double precision, allocatable, dimension(:) :: ktmp

   ! Convert C int to Fortran logical
   logical :: verbose_logical, alchemy_logical

   verbose_logical = (verbose /= 0)
   alchemy_logical = (alchemy /= 0)

   kernels(:, :, :) = 0.0d0

   ! Get max number of neighbors
   maxneigh1 = maxval(nneigh1)
   maxneigh2 = maxval(nneigh2)

   ! Calculate angular normalization constant
   ang_norm2 = get_angular_norm2(t_width)

   ! pmax = max nuclear charge
   pmax1 = get_pmax(x1, n1)
   pmax2 = get_pmax(x2, n2)

   ! Get two-body weight function
   !ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
   !ksi2 = get_ksi(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)

   !  subroutine get_ksi(x, na, nneigh, two_body_power, cut_start, cut_distance, verbose, ksi)! result(ksi)
   ! maxneigh = maxval(nneigh)
   ! maxatoms = maxval(na)
   ! nm = size(x, dim=1)
   allocate (ksi1(size(x1, dim=1), maxval(n1), maxval(nneigh1)))
   allocate (ksi2(size(x2, dim=1), maxval(n2), maxval(nneigh2)))
   call get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose_logical, ksi1)
   call get_ksi(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose_logical, ksi2)

   n = size(parameters, dim=1)
   allocate (ktmp(n))

   ! Allocate three-body Fourier terms
   allocate (cosp1(nm1, maxval(n1), pmax1, order, maxneigh1))
   allocate (sinp1(nm1, maxval(n1), pmax1, order, maxneigh1))

   ! Initialize and pre-calculate three-body Fourier terms
   call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start, cut_distance, &
       & cosp1, sinp1, verbose_logical)

   ! Allocate three-body Fourier terms
   allocate (cosp2(nm2, maxval(n2), pmax2, order, maxneigh2))
   allocate (sinp2(nm2, maxval(n2), pmax2, order, maxneigh2))

   ! Initialize and pre-calculate three-body Fourier terms
   call init_cosp_sinp(x2, n2, nneigh2, three_body_power, order, cut_start, cut_distance, &
       & cosp2, sinp2, verbose_logical)

   ! Pre-calculate self-scalar terms
   !self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
   !     & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose)
   allocate (self_scalar1(nm1, maxval(n1)))
   call get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
        & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy_logical, verbose_logical,&
 &self_scalar1)

   ! Pre-calculate self-scalar terms
   !self_scalar2 = get_selfscalar(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, d_width, &
   !     & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose)
   allocate (self_scalar2(nm2, maxval(n2)))
   call get_selfscalar(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, d_width, &
        & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy_logical, verbose_logical,&
    &self_scalar2)

   !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,ktmp)
   do b = 1, nm2
      nj = n2(b)
      do a = 1, nm1
         ni = n1(a)

         do i = 1, ni
            do j = 1, nj

               s12 = scalar(x1(a, i, :, :), x2(b, j, :, :), &
                   & nneigh1(a, i), nneigh2(b, j), ksi1(a, i, :), ksi2(b, j, :), &
                   & sinp1(a, i, :, :, :), sinp2(b, j, :, :, :), &
                   & cosp1(a, i, :, :, :), cosp2(b, j, :, :, :), &
                   & t_width, d_width, cut_distance, order, &
                   & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)

               !kernels(:, a, b) = kernels(:, a, b) &
               !    & + kernel(self_scalar1(a, i), self_scalar2(b, j), s12, &
               !    & kernel_idx, parameters)

               ktmp = 0.0d0
               call kernel(self_scalar1(a, i), self_scalar2(b, j), s12, &
               & kernel_idx, parameters, ktmp)
               kernels(:, a, b) = kernels(:, a, b) + ktmp

            end do
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (ktmp)
   deallocate (self_scalar1)
   deallocate (self_scalar2)
   deallocate (ksi1)
   deallocate (ksi2)
   deallocate (cosp1)
   deallocate (cosp2)
   deallocate (sinp1)
   deallocate (sinp2)

end subroutine fget_kernels_fchl

subroutine fget_symmetric_kernels_fchl(nm1, na1, nf1, nn1, np1, npd1, npd2, npar1, npar2, &
       & x1, verbose, n1, nneigh1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, &
       & kernel_idx, parameters, kernels) bind(C, name="fget_symmetric_kernels_fchl")

   use iso_c_binding
   use ffchl_module, only: scalar, get_angular_norm2, get_pmax, get_ksi, init_cosp_sinp, get_selfscalar
   use ffchl_kernels, only: kernel

   implicit none

   ! Dimensions (must come first for bind(C))
   integer(c_int), intent(in), value :: nm1           ! Number of molecules
   integer(c_int), intent(in), value :: na1, nf1, nn1 ! x1 dimensions: natoms, nfeatures, nneighbors
   integer(c_int), intent(in), value :: np1           ! n1 dimension
   integer(c_int), intent(in), value :: npd1, npd2    ! pd dimensions
   integer(c_int), intent(in), value :: npar1, npar2  ! parameters dimensions
   integer(c_int), intent(in), value :: nsigmas       ! Number of sigmas
   integer(c_int), intent(in), value :: order         ! Truncation order for Fourier terms
   integer(c_int), intent(in), value :: kernel_idx    ! Kernel ID

   ! FCHL descriptors for the training set, format (i,j_1,5,m_1)
   real(c_double), dimension(nm1, na1, nf1, nn1), intent(in) :: x1

   ! Whether to be verbose with output (int instead of logical for C compat)
   integer(c_int), intent(in), value :: verbose

   ! List of numbers of atoms in each molecule
   integer(c_int), dimension(np1), intent(in) :: n1

   ! Number of neighbors for each atom in each compound
   integer(c_int), dimension(nm1, na1), intent(in) :: nneigh1

   real(c_double), intent(in), value :: two_body_power
   real(c_double), intent(in), value :: three_body_power

   real(c_double), intent(in), value :: t_width
   real(c_double), intent(in), value :: d_width
   real(c_double), intent(in), value :: cut_start
   real(c_double), intent(in), value :: cut_distance
   real(c_double), intent(in), value :: distance_scale
   real(c_double), intent(in), value :: angular_scale

   ! Switch alchemy on or off (int instead of logical for C compat)
   integer(c_int), intent(in), value :: alchemy
   real(c_double), dimension(npd1, npd2), intent(in) :: pd

   ! Resulting alpha vector
   real(c_double), dimension(nsigmas, nm1, nm1), intent(out) :: kernels

   ! Kernel parameters
   real(c_double), dimension(npar1, npar2), intent(in) :: parameters

   ! Internal counters
   integer :: i, j, ni, nj
   integer :: a, b

   ! Temporary variables necessary for parallelization
   double precision :: s12

   ! Pre-computed terms in the full distance matrix
   double precision, allocatable, dimension(:, :) :: self_scalar1

   ! Pre-computed terms
   double precision, allocatable, dimension(:, :, :) :: ksi1

   double precision, allocatable, dimension(:, :, :, :, :) :: sinp1
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp1

   ! counter for periodic distance
   integer :: pmax1

   double precision :: ang_norm2

   integer :: maxneigh1

   ! Work kernel
   double precision, allocatable, dimension(:) :: ktmp

   ! Convert C int to Fortran logical
   logical :: verbose_logical, alchemy_logical

   verbose_logical = (verbose /= 0)
   alchemy_logical = (alchemy /= 0)

   kernels(:, :, :) = 0.0d0

   ang_norm2 = get_angular_norm2(t_width)

   maxneigh1 = maxval(nneigh1)
   pmax1 = get_pmax(x1, n1)

   allocate (ksi1(size(x1, dim=1), maxval(n1), maxval(nneigh1)))
   call get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose_logical, ksi1)
   !ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)

   allocate (cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
   allocate (sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

   call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start, cut_distance, &
       & cosp1, sinp1, verbose_logical)

   allocate (self_scalar1(nm1, maxval(n1)))
   call get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
        & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy_logical, verbose_logical, self_scalar1)
   !self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
   !     & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose)

   allocate (ktmp(size(parameters, dim=1)))

   !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,ktmp)
   do b = 1, nm1
      nj = n1(b)
      do a = b, nm1
         ni = n1(a)

         do i = 1, ni
            do j = 1, nj

               s12 = scalar(x1(a, i, :, :), x1(b, j, :, :), &
                   & nneigh1(a, i), nneigh1(b, j), ksi1(a, i, :), ksi1(b, j, :), &
                   & sinp1(a, i, :, :, :), sinp1(b, j, :, :, :), &
                   & cosp1(a, i, :, :, :), cosp1(b, j, :, :, :), &
                   & t_width, d_width, cut_distance, order, &
                   & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)

               ktmp(:) = 0.0d0
               call kernel(self_scalar1(a, i), self_scalar1(b, j), s12, &
                   & kernel_idx, parameters, ktmp)

               kernels(:, a, b) = kernels(:, a, b) + ktmp
               !kernels(:, a, b) = kernels(:, a, b) &
               !    & + kernel(self_scalar1(a, i), self_scalar1(b, j), s12, &
               !    & kernel_idx, parameters)

               kernels(:, b, a) = kernels(:, a, b)

            end do
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (ktmp)
   deallocate (self_scalar1)
   deallocate (ksi1)
   deallocate (cosp1)
   deallocate (sinp1)

end subroutine fget_symmetric_kernels_fchl

subroutine fget_global_symmetric_kernels_fchl(nm1, na1, nf1, nn1, np1, npd1, npd2, npar1, npar2, &
       & x1, verbose, n1, nneigh1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, &
       & kernel_idx, parameters, kernels) bind(C, name="fget_global_symmetric_kernels_fchl")

   use iso_c_binding
   use ffchl_module, only: scalar, get_angular_norm2, get_pmax, get_ksi, init_cosp_sinp
   use ffchl_kernels, only: kernel

   implicit none

   ! Dimensions (must come first for bind(C))
   integer(c_int), intent(in), value :: nm1           ! Number of molecules
   integer(c_int), intent(in), value :: na1, nf1, nn1 ! x1 dimensions
   integer(c_int), intent(in), value :: np1           ! n1 dimension
   integer(c_int), intent(in), value :: npd1, npd2    ! pd dimensions
   integer(c_int), intent(in), value :: npar1, npar2  ! parameters dimensions
   integer(c_int), intent(in), value :: nsigmas       ! Number of sigmas
   integer(c_int), intent(in), value :: order         ! Truncation order
   integer(c_int), intent(in), value :: kernel_idx    ! Kernel ID

   ! FCHL descriptors for the training set
   real(c_double), dimension(nm1, na1, nf1, nn1), intent(in) :: x1

   ! Whether to be verbose with output
   integer(c_int), intent(in), value :: verbose

   ! List of numbers of atoms in each molecule
   integer(c_int), dimension(np1), intent(in) :: n1

   ! Number of neighbors for each atom in each compound
   integer(c_int), dimension(nm1, na1), intent(in) :: nneigh1

   real(c_double), intent(in), value :: two_body_power
   real(c_double), intent(in), value :: three_body_power
   real(c_double), intent(in), value :: t_width
   real(c_double), intent(in), value :: d_width
   real(c_double), intent(in), value :: cut_start
   real(c_double), intent(in), value :: cut_distance
   real(c_double), intent(in), value :: distance_scale
   real(c_double), intent(in), value :: angular_scale

   ! Switch alchemy on or off
   integer(c_int), intent(in), value :: alchemy
   real(c_double), dimension(npd1, npd2), intent(in) :: pd

   ! Resulting kernel matrix
   real(c_double), dimension(nsigmas, nm1, nm1), intent(out) :: kernels

   ! Kernel parameters
   real(c_double), dimension(npar1, npar2), intent(in) :: parameters

   ! Internal counters
   integer :: i, j, ni, nj
   integer :: a, b

   ! Temporary variables
   double precision :: s12
   double precision :: mol_dist

   ! Pre-computed terms
   double precision, allocatable, dimension(:) :: self_scalar1
   double precision, allocatable, dimension(:, :, :) :: ksi1
   double precision, allocatable, dimension(:, :, :, :, :) :: sinp1
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp1

   ! Helper variables
   integer :: pmax1
   double precision :: ang_norm2
   integer :: maxneigh1

   ! Work kernel
   double precision, allocatable, dimension(:) :: ktmp

   ! Convert C int to Fortran logical
   logical :: verbose_logical, alchemy_logical

   verbose_logical = (verbose /= 0)
   alchemy_logical = (alchemy /= 0)

   allocate (ktmp(size(parameters, dim=1)))

   maxneigh1 = maxval(nneigh1)

   ang_norm2 = get_angular_norm2(t_width)

   pmax1 = get_pmax(x1, n1)

   !ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
   allocate (ksi1(size(x1, dim=1), maxval(n1), maxval(nneigh1)))
   call get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose_logical, ksi1)

   allocate (cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
   allocate (sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

   call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start, cut_distance, &
       & cosp1, sinp1, verbose_logical)

   allocate (self_scalar1(nm1))

   self_scalar1 = 0.0d0

   !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:self_scalar1)
   do a = 1, nm1
      ni = n1(a)
      do i = 1, ni
         do j = 1, ni

            self_scalar1(a) = self_scalar1(a) + scalar(x1(a, i, :, :), x1(a, j, :, :), &
                & nneigh1(a, i), nneigh1(a, j), ksi1(a, i, :), ksi1(a, j, :), &
                & sinp1(a, i, :, :, :), sinp1(a, j, :, :, :), &
                & cosp1(a, i, :, :, :), cosp1(a, j, :, :, :), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)
         end do
      end do
   end do
   !$OMP END PARALLEL DO

   kernels(:, :, :) = 0.0d0

   !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,mol_dist,ktmp)
   do b = 1, nm1
      nj = n1(b)
      do a = b, nm1
         ni = n1(a)

         mol_dist = 0.0d0

         do i = 1, ni
            do j = 1, nj

             s12 = scalar(x1(a, i, :, :), x1(b, j, :, :), &
                 & nneigh1(a, i), nneigh1(b, j), ksi1(a, i, :), ksi1(b, j, :), &
                 & sinp1(a, i, :, :, :), sinp1(b, j, :, :, :), &
                 & cosp1(a, i, :, :, :), cosp1(b, j, :, :, :), &
                 & t_width, d_width, cut_distance, order, &
                 & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)

             mol_dist = mol_dist + s12

          end do
       end do

       ktmp = 0.0d0
       call kernel(self_scalar1(a), self_scalar1(b), mol_dist, &
           & kernel_idx, parameters, ktmp)
       kernels(:, a, b) = ktmp
       !kernels(:, a, b) = kernel(self_scalar1(a), self_scalar1(b), mol_dist, &
       !    & kernel_idx, parameters)

       kernels(:, b, a) = kernels(:, a, b)

    end do
 end do
 !$OMP END PARALLEL DO

 deallocate (ktmp)
 deallocate (self_scalar1)
 deallocate (ksi1)
 deallocate (cosp1)
 deallocate (sinp1)

end subroutine fget_global_symmetric_kernels_fchl

subroutine fget_global_kernels_fchl(nm1, nm2, na1, nf1, nn1, na2, nf2, nn2, &
       & np1, np2, npd1, npd2, npar1, npar2, &
       & x1, x2, verbose, n1, n2, nneigh1, nneigh2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, &
       & kernel_idx, parameters, kernels) bind(C, name="fget_global_kernels_fchl")

   use iso_c_binding
   use ffchl_module, only: scalar, get_angular_norm2, get_pmax, get_ksi, init_cosp_sinp
   use ffchl_kernels, only: kernel

   implicit none

   ! Dimensions (must come first for bind(C))
   integer(c_int), intent(in), value :: nm1, nm2      ! Number of molecules
   integer(c_int), intent(in), value :: na1, nf1, nn1 ! x1 dimensions
   integer(c_int), intent(in), value :: na2, nf2, nn2 ! x2 dimensions
   integer(c_int), intent(in), value :: np1, np2      ! n1, n2 dimensions
   integer(c_int), intent(in), value :: npd1, npd2    ! pd dimensions
   integer(c_int), intent(in), value :: npar1, npar2  ! parameters dimensions
   integer(c_int), intent(in), value :: nsigmas       ! Number of sigmas
   integer(c_int), intent(in), value :: order         ! Truncation order
   integer(c_int), intent(in), value :: kernel_idx    ! Kernel ID

   ! fchl descriptors
   real(c_double), dimension(nm1, na1, nf1, nn1), intent(in) :: x1
   real(c_double), dimension(nm2, na2, nf2, nn2), intent(in) :: x2

   ! Whether to be verbose with output
   integer(c_int), intent(in), value :: verbose

   ! List of numbers of atoms in each molecule
   integer(c_int), dimension(np1), intent(in) :: n1
   integer(c_int), dimension(np2), intent(in) :: n2

   ! Number of neighbors for each atom in each compound
   integer(c_int), dimension(nm1, na1), intent(in) :: nneigh1
   integer(c_int), dimension(nm2, na2), intent(in) :: nneigh2

   real(c_double), intent(in), value :: two_body_power
   real(c_double), intent(in), value :: three_body_power
   real(c_double), intent(in), value :: t_width
   real(c_double), intent(in), value :: d_width
   real(c_double), intent(in), value :: cut_start
   real(c_double), intent(in), value :: cut_distance
   real(c_double), intent(in), value :: distance_scale
   real(c_double), intent(in), value :: angular_scale

   ! Switch alchemy on or off
   integer(c_int), intent(in), value :: alchemy
   real(c_double), dimension(npd1, npd2), intent(in) :: pd

   ! Resulting kernel matrix
   real(c_double), dimension(nsigmas, nm1, nm2), intent(out) :: kernels

   ! Kernel parameters
   real(c_double), dimension(npar1, npar2), intent(in) :: parameters

   ! Internal counters
   integer :: i, j
   integer :: ni, nj
   integer :: a, b

   ! Temporary variables
   double precision :: s12
   double precision :: mol_dist

   ! Pre-computed terms
   double precision, allocatable, dimension(:) :: self_scalar1
   double precision, allocatable, dimension(:) :: self_scalar2
   double precision, allocatable, dimension(:, :, :) :: ksi1
   double precision, allocatable, dimension(:, :, :) :: ksi2
   double precision, allocatable, dimension(:, :, :, :, :) :: sinp1
   double precision, allocatable, dimension(:, :, :, :, :) :: sinp2
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp1
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp2

   ! Helper variables
   integer :: pmax1, pmax2
   double precision :: ang_norm2
   integer :: maxneigh1, maxneigh2

   ! Work kernel
   double precision, allocatable, dimension(:) :: ktmp

   ! Convert C int to Fortran logical
   logical :: verbose_logical, alchemy_logical

   verbose_logical = (verbose /= 0)
   alchemy_logical = (alchemy /= 0)

   allocate (ktmp(size(parameters, dim=1)))

   maxneigh1 = maxval(nneigh1)
   maxneigh2 = maxval(nneigh2)

   ang_norm2 = get_angular_norm2(t_width)

   pmax1 = get_pmax(x1, n1)
   pmax2 = get_pmax(x2, n2)

   allocate (ksi1(size(x1, dim=1), maxval(n1), maxval(nneigh1)))
   allocate (ksi2(size(x2, dim=1), maxval(n2), maxval(nneigh2)))
   call get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose_logical, ksi1)
   call get_ksi(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose_logical, ksi2)
   !ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
   !ksi2 = get_ksi(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)

   allocate (cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
   allocate (sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

   call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start, cut_distance, &
       & cosp1, sinp1, verbose_logical)

   allocate (cosp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))
   allocate (sinp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))

   call init_cosp_sinp(x2, n2, nneigh2, three_body_power, order, cut_start, cut_distance, &
       & cosp2, sinp2, verbose_logical)

   ! Global self-scalar have their own summation and are not a general function
   allocate (self_scalar1(nm1))
   allocate (self_scalar2(nm2))

   self_scalar1 = 0.0d0
   self_scalar2 = 0.0d0

   !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:self_scalar1)
   do a = 1, nm1
      ni = n1(a)
      do i = 1, ni
         do j = 1, ni

            self_scalar1(a) = self_scalar1(a) + scalar(x1(a, i, :, :), x1(a, j, :, :), &
                & nneigh1(a, i), nneigh1(a, j), ksi1(a, i, :), ksi1(a, j, :), &
                & sinp1(a, i, :, :, :), sinp1(a, j, :, :, :), &
                & cosp1(a, i, :, :, :), cosp1(a, j, :, :, :), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)
         end do
      end do
   end do
   !$OMP END PARALLEL DO

   !$OMP PARALLEL DO PRIVATE(ni) REDUCTION(+:self_scalar2)
   do a = 1, nm2
      ni = n2(a)
      do i = 1, ni
         do j = 1, ni
            self_scalar2(a) = self_scalar2(a) + scalar(x2(a, i, :, :), x2(a, j, :, :), &
                & nneigh2(a, i), nneigh2(a, j), ksi2(a, i, :), ksi2(a, j, :), &
                & sinp2(a, i, :, :, :), sinp2(a, j, :, :, :), &
                & cosp2(a, i, :, :, :), cosp2(a, j, :, :, :), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)
         end do
      end do
   end do
   !$OMP END PARALLEL DO

   kernels(:, :, :) = 0.0d0

   !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12,ni,nj,mol_dist,ktmp)
   do b = 1, nm2
      nj = n2(b)
      do a = 1, nm1
         ni = n1(a)

         mol_dist = 0.0d0

         do i = 1, ni
            do j = 1, nj

               s12 = scalar(x1(a, i, :, :), x2(b, j, :, :), &
                   & nneigh1(a, i), nneigh2(b, j), ksi1(a, i, :), ksi2(b, j, :), &
                   & sinp1(a, i, :, :, :), sinp2(b, j, :, :, :), &
                   & cosp1(a, i, :, :, :), cosp2(b, j, :, :, :), &
                   & t_width, d_width, cut_distance, order, &
                   & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)

               mol_dist = mol_dist + s12

            end do
         end do

         ktmp = 0.0d0
         call kernel(self_scalar1(a), self_scalar2(b), mol_dist, &
             & kernel_idx, parameters, ktmp)
         kernels(:, a, b) = ktmp
         !kernels(:, a, b) = kernel(self_scalar1(a), self_scalar2(b), mol_dist, &
         !    & kernel_idx, parameters)

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (ktmp)
   deallocate (self_scalar1)
   deallocate (self_scalar2)
   deallocate (ksi1)
   deallocate (ksi2)
   deallocate (cosp1)
   deallocate (cosp2)
   deallocate (sinp1)
   deallocate (sinp2)

end subroutine fget_global_kernels_fchl

subroutine fget_atomic_kernels_fchl(na1, nf1, nn1, na2, nf2, nn2, &
       & np1, np2, npd1, npd2, npar1, npar2, &
       & x1, x2, verbose, nneigh1, nneigh2, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, &
       & kernel_idx, parameters, kernels) bind(C, name="fget_atomic_kernels_fchl")

   use iso_c_binding
   use ffchl_module, only: scalar, get_angular_norm2, &
       & get_pmax_atomic, get_ksi_atomic, init_cosp_sinp_atomic
   use ffchl_kernels, only: kernel

   implicit none

   ! Dimensions (must come first for bind(C))
   integer(c_int), intent(in), value :: na1, nf1, nn1  ! x1 dimensions: natoms, nfeatures, nneighbors
   integer(c_int), intent(in), value :: na2, nf2, nn2  ! x2 dimensions
   integer(c_int), intent(in), value :: np1, np2       ! nneigh1, nneigh2 dimensions
   integer(c_int), intent(in), value :: npd1, npd2     ! pd dimensions
   integer(c_int), intent(in), value :: npar1, npar2   ! parameters dimensions
   integer(c_int), intent(in), value :: nsigmas        ! Number of sigmas
   integer(c_int), intent(in), value :: order          ! Truncation order
   integer(c_int), intent(in), value :: kernel_idx     ! Kernel ID

   ! fchl descriptors for the training set, format (i,5,maxneighbors)
   real(c_double), dimension(na1, nf1, nn1), intent(in) :: x1
   real(c_double), dimension(na2, nf2, nn2), intent(in) :: x2

   ! Whether to be verbose with output
   integer(c_int), intent(in), value :: verbose

   ! Number of neighbors for each atom
   integer(c_int), dimension(np1), intent(in) :: nneigh1
   integer(c_int), dimension(np2), intent(in) :: nneigh2

   real(c_double), intent(in), value :: two_body_power
   real(c_double), intent(in), value :: three_body_power
   real(c_double), intent(in), value :: t_width
   real(c_double), intent(in), value :: d_width
   real(c_double), intent(in), value :: cut_start
   real(c_double), intent(in), value :: cut_distance
   real(c_double), intent(in), value :: distance_scale
   real(c_double), intent(in), value :: angular_scale

   ! Switch alchemy on or off
   integer(c_int), intent(in), value :: alchemy
   real(c_double), dimension(npd1, npd2), intent(in) :: pd

   ! Resulting kernel matrix
   real(c_double), dimension(nsigmas, na1, na2), intent(out) :: kernels

   ! Kernel parameters
   real(c_double), dimension(npar1, npar2), intent(in) :: parameters

   ! Internal counters
   integer :: i, j

   ! Temporary variables
   double precision :: s12

   ! Pre-computed terms
   double precision, allocatable, dimension(:) :: self_scalar1
   double precision, allocatable, dimension(:) :: self_scalar2
   double precision, allocatable, dimension(:, :) :: ksi1
   double precision, allocatable, dimension(:, :) :: ksi2
   double precision, allocatable, dimension(:, :, :, :) :: sinp1
   double precision, allocatable, dimension(:, :, :, :) :: sinp2
   double precision, allocatable, dimension(:, :, :, :) :: cosp1
   double precision, allocatable, dimension(:, :, :, :) :: cosp2

   ! Helper variables
   integer :: pmax1, pmax2
   double precision :: ang_norm2
   integer :: maxneigh1, maxneigh2

   ! Work kernel
   double precision, allocatable, dimension(:) :: ktmp

   ! Convert C int to Fortran logical
   logical :: verbose_logical, alchemy_logical

   verbose_logical = (verbose /= 0)
   alchemy_logical = (alchemy /= 0)

   allocate (ktmp(size(parameters, dim=1)))

   maxneigh1 = maxval(nneigh1)
   maxneigh2 = maxval(nneigh2)

   ang_norm2 = get_angular_norm2(t_width)

   pmax1 = get_pmax_atomic(x1, nneigh1)
   pmax2 = get_pmax_atomic(x2, nneigh2)

   allocate (ksi1(na1, maxval(nneigh1)))
   allocate (ksi2(na2, maxval(nneigh2)))
   call get_ksi_atomic(x1, na1, nneigh1, two_body_power, cut_start, cut_distance, verbose_logical, ksi1)
   call get_ksi_atomic(x2, na2, nneigh2, two_body_power, cut_start, cut_distance, verbose_logical, ksi2)
   !ksi1 = get_ksi_atomic(x1, na1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
   !ksi2 = get_ksi_atomic(x2, na2, nneigh2, two_body_power, cut_start, cut_distance, verbose)

   allocate (cosp1(na1, pmax1, order, maxneigh1))
   allocate (sinp1(na1, pmax1, order, maxneigh1))

   call init_cosp_sinp_atomic(x1, na1, nneigh1, three_body_power, order, cut_start, cut_distance, &
       & cosp1, sinp1, verbose_logical)

   allocate (cosp2(na2, pmax2, order, maxneigh2))
   allocate (sinp2(na2, pmax2, order, maxneigh2))

   call init_cosp_sinp_atomic(x2, na2, nneigh2, three_body_power, order, cut_start, cut_distance, &
       & cosp2, sinp2, verbose_logical)

   allocate (self_scalar1(na1))
   allocate (self_scalar2(na2))

   self_scalar1 = 0.0d0
   self_scalar2 = 0.0d0

   !$OMP PARALLEL DO
   do i = 1, na1
      self_scalar1(i) = scalar(x1(i, :, :), x1(i, :, :), &
          & nneigh1(i), nneigh1(i), ksi1(i, :), ksi1(i, :), &
          & sinp1(i, :, :, :), sinp1(i, :, :, :), &
          & cosp1(i, :, :, :), cosp1(i, :, :, :), &
          & t_width, d_width, cut_distance, order, &
          & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)
   end do
   !$OMP END PARALLEL DO

   !$OMP PARALLEL DO
   do i = 1, na2
      self_scalar2(i) = scalar(x2(i, :, :), x2(i, :, :), &
          & nneigh2(i), nneigh2(i), ksi2(i, :), ksi2(i, :), &
          & sinp2(i, :, :, :), sinp2(i, :, :, :), &
          & cosp2(i, :, :, :), cosp2(i, :, :, :), &
          & t_width, d_width, cut_distance, order, &
          & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)
   end do
   !$OMP END PARALLEL DO

   kernels(:, :, :) = 0.0d0

   !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12, ktmp)
   do i = 1, na1
      do j = 1, na2

         s12 = scalar(x1(i, :, :), x2(j, :, :), &
             & nneigh1(i), nneigh2(j), ksi1(i, :), ksi2(j, :), &
             & sinp1(i, :, :, :), sinp2(j, :, :, :), &
             & cosp1(i, :, :, :), cosp2(j, :, :, :), &
             & t_width, d_width, cut_distance, order, &
             & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)

         ktmp = 0.0d0
         call kernel(self_scalar1(i), self_scalar2(j), s12, &
                 & kernel_idx, parameters, ktmp)
         kernels(:, i, j) = ktmp
         !kernels(:, i, j) = kernel(self_scalar1(i), self_scalar2(j), s12, &
         !        & kernel_idx, parameters)

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (ktmp)
   deallocate (self_scalar1)
   deallocate (self_scalar2)
   deallocate (ksi1)
   deallocate (ksi2)
   deallocate (cosp1)
   deallocate (cosp2)
   deallocate (sinp1)
   deallocate (sinp2)

end subroutine fget_atomic_kernels_fchl

subroutine fget_atomic_symmetric_kernels_fchl(na1, nf1, nn1, np1, npd1, npd2, npar1, npar2, &
       & x1, verbose, nneigh1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, &
       & kernel_idx, parameters, kernels) bind(C, name="fget_atomic_symmetric_kernels_fchl")

   use iso_c_binding
   use ffchl_module, only: scalar, get_angular_norm2, &
       & get_pmax_atomic, get_ksi_atomic, init_cosp_sinp_atomic
   use ffchl_kernels, only: kernel

   implicit none

   ! Dimensions (must come first for bind(C))
   integer(c_int), intent(in), value :: na1, nf1, nn1  ! x1 dimensions: natoms, nfeatures, nneighbors
   integer(c_int), intent(in), value :: np1            ! nneigh1 dimension
   integer(c_int), intent(in), value :: npd1, npd2     ! pd dimensions
   integer(c_int), intent(in), value :: npar1, npar2   ! parameters dimensions
   integer(c_int), intent(in), value :: nsigmas        ! Number of sigmas
   integer(c_int), intent(in), value :: order          ! Truncation order
   integer(c_int), intent(in), value :: kernel_idx     ! Kernel ID

   ! fchl descriptors for the training set, format (i,5,maxneighbors)
   real(c_double), dimension(na1, nf1, nn1), intent(in) :: x1

   ! Whether to be verbose with output
   integer(c_int), intent(in), value :: verbose

   ! Number of neighbors for each atom
   integer(c_int), dimension(np1), intent(in) :: nneigh1

   real(c_double), intent(in), value :: two_body_power
   real(c_double), intent(in), value :: three_body_power
   real(c_double), intent(in), value :: t_width
   real(c_double), intent(in), value :: d_width
   real(c_double), intent(in), value :: cut_start
   real(c_double), intent(in), value :: cut_distance
   real(c_double), intent(in), value :: distance_scale
   real(c_double), intent(in), value :: angular_scale

   ! Switch alchemy on or off
   integer(c_int), intent(in), value :: alchemy
   real(c_double), dimension(npd1, npd2), intent(in) :: pd

   ! Kernel parameters
   real(c_double), dimension(npar1, npar2), intent(in) :: parameters

   ! Resulting kernel matrix
   real(c_double), dimension(nsigmas, na1, na1), intent(out) :: kernels

   ! Internal counters
   integer :: i, j

   ! Temporary variables necessary for parallelization
   double precision :: s12

   ! Convert C int to Fortran logical
   logical :: verbose_logical, alchemy_logical

   ! Pre-computed terms in the full distance matrix
   double precision, allocatable, dimension(:) :: self_scalar1

   ! Pre-computed terms
   double precision, allocatable, dimension(:, :) :: ksi1

   double precision, allocatable, dimension(:, :, :, :) :: sinp1
   double precision, allocatable, dimension(:, :, :, :) :: cosp1

   ! counter for periodic distance
   integer :: pmax1
   double precision :: ang_norm2

   integer :: maxneigh1

   ! Work kernel
   double precision, allocatable, dimension(:) :: ktmp
   allocate (ktmp(size(parameters, dim=1)))

   ! Convert C integers to Fortran logicals
   verbose_logical = (verbose /= 0)
   alchemy_logical = (alchemy /= 0)

   maxneigh1 = maxval(nneigh1)

   ang_norm2 = get_angular_norm2(t_width)

   pmax1 = get_pmax_atomic(x1, nneigh1)

   allocate (ksi1(na1, maxval(nneigh1)))
   call get_ksi_atomic(x1, na1, nneigh1, two_body_power, cut_start, cut_distance, verbose_logical, ksi1)
   !ksi1 = get_ksi_atomic(x1, na1, nneigh1, two_body_power, cut_start, cut_distance, verbose)

   allocate (cosp1(na1, pmax1, order, maxneigh1))
   allocate (sinp1(na1, pmax1, order, maxneigh1))

   call init_cosp_sinp_atomic(x1, na1, nneigh1, three_body_power, order, cut_start, cut_distance, &
       & cosp1, sinp1, verbose_logical)

   allocate (self_scalar1(na1))

   self_scalar1 = 0.0d0

   !$OMP PARALLEL DO
   do i = 1, na1
      self_scalar1(i) = scalar(x1(i, :, :), x1(i, :, :), &
          & nneigh1(i), nneigh1(i), ksi1(i, :), ksi1(i, :), &
          & sinp1(i, :, :, :), sinp1(i, :, :, :), &
          & cosp1(i, :, :, :), cosp1(i, :, :, :), &
          & t_width, d_width, cut_distance, order, &
          & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)
   end do
   !$OMP END PARALLEL DO

   kernels(:, :, :) = 0.0d0

   !$OMP PARALLEL DO schedule(dynamic) PRIVATE(s12, ktmp)
   do i = 1, na1
      do j = i, na1

         s12 = scalar(x1(i, :, :), x1(j, :, :), &
             & nneigh1(i), nneigh1(j), ksi1(i, :), ksi1(j, :), &
             & sinp1(i, :, :, :), sinp1(j, :, :, :), &
             & cosp1(i, :, :, :), cosp1(j, :, :, :), &
             & t_width, d_width, cut_distance, order, &
             & pd, ang_norm2, distance_scale, angular_scale, alchemy_logical)

         !kernels(:, i, j) = kernel(self_scalar1(i), self_scalar1(j), s12, &
         !        & kernel_idx, parameters)

         ktmp = 0.0d0
         call kernel(self_scalar1(i), self_scalar1(j), s12, &
                 & kernel_idx, parameters, ktmp)

         kernels(:, i, j) = ktmp
         kernels(:, j, i) = kernels(:, i, j)

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (ktmp)
   deallocate (self_scalar1)
   deallocate (ksi1)
   deallocate (cosp1)
   deallocate (sinp1)

end subroutine fget_atomic_symmetric_kernels_fchl

subroutine fget_atomic_local_kernels_fchl(x1, x2, verbose, n1, n2, nneigh1, nneigh2, &
       & nm1, nm2, na1, nsigmas, &
       & t_width, d_width, cut_start, cut_distance, order, pd, &
       & distance_scale, angular_scale, alchemy, two_body_power, three_body_power, &
       & kernel_idx, parameters, kernels)

   use ffchl_module, only: scalar, get_threebody_fourier, get_twobody_weights, &
       & get_angular_norm2, get_pmax, get_ksi, init_cosp_sinp, get_selfscalar
   use ffchl_kernels, only: kernel

   implicit none

   ! fchl descriptors for the training set, format (i,maxatoms,5,maxneighbors)
   double precision, dimension(:, :, :, :), intent(in) :: x1
   double precision, dimension(:, :, :, :), intent(in) :: x2

   ! Whether to be verbose with output
   logical, intent(in) :: verbose

   ! List of numbers of atoms in each molecule
   integer, dimension(:), intent(in) :: n1
   integer, dimension(:), intent(in) :: n2

   ! Number of neighbors for each atom in each compound
   integer, dimension(:, :), intent(in) :: nneigh1
   integer, dimension(:, :), intent(in) :: nneigh2

   ! Number of molecules
   integer, intent(in) :: nm1
   integer, intent(in) :: nm2

   integer, intent(in) :: na1

   ! Number of sigmas
   integer, intent(in) :: nsigmas

   double precision, intent(in) :: two_body_power
   double precision, intent(in) :: three_body_power

   double precision, intent(in) :: t_width
   double precision, intent(in) :: d_width
   double precision, intent(in) :: cut_start
   double precision, intent(in) :: cut_distance
   integer, intent(in) :: order
   double precision, intent(in) :: distance_scale
   double precision, intent(in) :: angular_scale

   ! -1.0 / sigma^2 for use in the kernel

   double precision, dimension(:, :), intent(in) :: pd

   integer, intent(in) :: kernel_idx
   double precision, dimension(:, :), intent(in) :: parameters

   ! Resulting alpha vector
   double precision, dimension(nsigmas, na1, nm2), intent(out) :: kernels

   integer :: idx1

   ! Internal counters
   integer :: i, j
   integer :: ni, nj
   integer :: a, b

   ! Temporary variables necessary for parallelization
   double precision :: s12

   ! Pre-computed terms in the full distance matrix
   double precision, allocatable, dimension(:, :) :: self_scalar1
   double precision, allocatable, dimension(:, :) :: self_scalar2

   ! Pre-computed terms
   double precision, allocatable, dimension(:, :, :) :: ksi1
   double precision, allocatable, dimension(:, :, :) :: ksi2

   double precision, allocatable, dimension(:, :, :, :, :) :: sinp1
   double precision, allocatable, dimension(:, :, :, :, :) :: sinp2
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp1
   double precision, allocatable, dimension(:, :, :, :, :) :: cosp2

   logical, intent(in) :: alchemy

   ! Value of PI at full FORTRAN precision.
   double precision, parameter :: pi = 4.0d0*atan(1.0d0)

   ! counter for periodic distance
   integer :: pmax1
   integer :: pmax2
   ! integer :: nneighi

   double precision :: ang_norm2

   integer :: maxneigh1
   integer :: maxneigh2

   ! Work kernel
   double precision, allocatable, dimension(:) :: ktmp
   allocate (ktmp(size(parameters, dim=1)))

   maxneigh1 = maxval(nneigh1)
   maxneigh2 = maxval(nneigh2)

   ang_norm2 = get_angular_norm2(t_width)

   pmax1 = get_pmax(x1, n1)
   pmax2 = get_pmax(x2, n2)

   allocate (ksi1(size(x1, dim=1), maxval(n1), maxval(nneigh1)))
   allocate (ksi2(size(x2, dim=1), maxval(n2), maxval(nneigh2)))
   call get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose, ksi1)
   call get_ksi(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose, ksi2)
   !ksi1 = get_ksi(x1, n1, nneigh1, two_body_power, cut_start, cut_distance, verbose)
   !ksi2 = get_ksi(x2, n2, nneigh2, two_body_power, cut_start, cut_distance, verbose)

   allocate (cosp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))
   allocate (sinp1(nm1, maxval(n1), pmax1, order, maxval(nneigh1)))

   call init_cosp_sinp(x1, n1, nneigh1, three_body_power, order, cut_start, cut_distance, &
       & cosp1, sinp1, verbose)

   allocate (cosp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))
   allocate (sinp2(nm2, maxval(n2), pmax2, order, maxval(nneigh2)))

   call init_cosp_sinp(x2, n2, nneigh2, three_body_power, order, cut_start, cut_distance, &
       & cosp2, sinp2, verbose)

   ! Pre-calculate self-scalar terms
   allocate (self_scalar1(nm1, maxval(n1)))
   allocate (self_scalar2(nm2, maxval(n2)))
   call get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
        & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose, self_scalar1)
   call get_selfscalar(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, d_width, &
        & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose, self_scalar2)
   !self_scalar1 = get_selfscalar(x1, nm1, n1, nneigh1, ksi1, sinp1, cosp1, t_width, d_width, &
   !     & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose)

   ! Pre-calculate self-scalar terms
   !self_scalar2 = get_selfscalar(x2, nm2, n2, nneigh2, ksi2, sinp2, cosp2, t_width, d_width, &
   !     & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose)

   kernels(:, :, :) = 0.0d0

   !$OMP PARALLEL DO schedule(dynamic) PRIVATE(ni,nj,idx1,s12,ktmp)
   do a = 1, nm1
      ni = n1(a)
      do i = 1, ni

         idx1 = sum(n1(:a)) - ni + i

         do b = 1, nm2
            nj = n2(b)
            do j = 1, nj

               s12 = scalar(x1(a, i, :, :), x2(b, j, :, :), &
                   & nneigh1(a, i), nneigh2(b, j), ksi1(a, i, :), ksi2(b, j, :), &
                   & sinp1(a, i, :, :, :), sinp2(b, j, :, :, :), &
                   & cosp1(a, i, :, :, :), cosp2(b, j, :, :, :), &
                   & t_width, d_width, cut_distance, order, &
                   & pd, ang_norm2, distance_scale, angular_scale, alchemy)

               ktmp = 0.0d0
               call kernel(self_scalar1(a, i), self_scalar2(b, j), s12, &
                   & kernel_idx, parameters, ktmp)
               kernels(:, idx1, b) = kernels(:, idx1, b) + ktmp

               !kernels(:, idx1, b) = kernels(:, idx1, b) &
               !    & + kernel(self_scalar1(a, i), self_scalar2(b, j), s12, &
               !    & kernel_idx, parameters)

            end do
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (ktmp)
   deallocate (self_scalar1)
   deallocate (self_scalar2)
   deallocate (ksi1)
   deallocate (ksi2)
   deallocate (cosp1)
   deallocate (cosp2)
   deallocate (sinp1)
   deallocate (sinp2)

end subroutine fget_atomic_local_kernels_fchl
