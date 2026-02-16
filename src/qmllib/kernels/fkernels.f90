subroutine fget_local_kernels_gaussian(q1, q2, n1, n2, sigmas, &
        & nm1, nm2, nsigmas, nq1, nq2, kernels) bind(C, name="fget_local_kernels_gaussian")

   use, intrinsic :: iso_c_binding
   implicit none

   ! Array dimensions
   integer(c_int), intent(in), value :: nq1  ! Size of q1 dimension 2
   integer(c_int), intent(in), value :: nq2  ! Size of q2 dimension 2
   integer(c_int), intent(in), value :: nm1
   integer(c_int), intent(in), value :: nm2
   integer(c_int), intent(in), value :: nsigmas

   double precision, dimension(3, nq1), intent(in) :: q1
   double precision, dimension(3, nq2), intent(in) :: q2

   ! List of numbers of atoms in each molecule
   integer, dimension(nm1), intent(in) :: n1
   integer, dimension(nm2), intent(in) :: n2

   ! Sigma in the Gaussian kernel
   double precision, dimension(nsigmas), intent(in) :: sigmas

   ! -1.0 / sigma^2 for use in the kernel
   double precision, dimension(nsigmas) :: inv_sigma2

   ! Resulting alpha vector
   double precision, dimension(nsigmas, nm1, nm2), intent(out) :: kernels

   ! Internal counters
   integer :: a, b, i, j, k, ni, nj

   ! Temporary variables necessary for parallelization
   double precision, allocatable, dimension(:, :) :: atomic_distance

   integer, allocatable, dimension(:) :: i_starts
   integer, allocatable, dimension(:) :: j_starts

   allocate (i_starts(nm1))
   allocate (j_starts(nm2))

   !$OMP PARALLEL DO
   do i = 1, nm1
      i_starts(i) = sum(n1(:i)) - n1(i)
   end do
   !$OMP END PARALLEL DO

   !$OMP PARALLEL DO
   do j = 1, nm2
      j_starts(j) = sum(n2(:j)) - n2(j)
   end do
   !$OMP END PARALLEL DO

   inv_sigma2(:) = -0.5d0/(sigmas(:))**2
   kernels(:, :, :) = 0.0d0

   allocate (atomic_distance(maxval(n1), maxval(n2)))
   atomic_distance(:, :) = 0.0d0

   !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj) SCHEDULE(dynamic) COLLAPSE(2)
   do a = 1, nm1
      do b = 1, nm2
         nj = n2(b)
         ni = n1(a)

         atomic_distance(:, :) = 0.0d0
         do i = 1, ni
            do j = 1, nj

               atomic_distance(i, j) = sum((q1(:, i + i_starts(a)) - q2(:, j + j_starts(b)))**2)

            end do
         end do

         do k = 1, nsigmas
            kernels(k, a, b) = sum(exp(atomic_distance(:ni, :nj)*inv_sigma2(k)))
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (atomic_distance)
   deallocate (i_starts)
   deallocate (j_starts)

end subroutine fget_local_kernels_gaussian

subroutine fget_local_kernels_laplacian(q1, q2, n1, n2, sigmas, &
        & nm1, nm2, nsigmas, nq1, nq2, kernels) bind(C, name="fget_local_kernels_laplacian")

   use, intrinsic :: iso_c_binding
   implicit none

   ! Array dimensions
   integer(c_int), intent(in), value :: nq1
   integer(c_int), intent(in), value :: nq2
   integer(c_int), intent(in), value :: nm1
   integer(c_int), intent(in), value :: nm2
   integer(c_int), intent(in), value :: nsigmas

   double precision, dimension(3, nq1), intent(in) :: q1
   double precision, dimension(3, nq2), intent(in) :: q2

   ! List of numbers of atoms in each molecule
   integer, dimension(nm1), intent(in) :: n1
   integer, dimension(nm2), intent(in) :: n2

   ! Sigma in the Gaussian kernel
   double precision, dimension(nsigmas), intent(in) :: sigmas

   ! -1.0 / sigma^2 for use in the kernel
   double precision, dimension(nsigmas) :: inv_sigma2

   ! Resulting alpha vector
   double precision, dimension(nsigmas, nm1, nm2), intent(out) :: kernels

   ! Internal counters
   integer :: a, b, i, j, k, ni, nj

   ! Temporary variables necessary for parallelization
   double precision, allocatable, dimension(:, :) :: atomic_distance

   integer, allocatable, dimension(:) :: i_starts
   integer, allocatable, dimension(:) :: j_starts

   allocate (i_starts(nm1))
   allocate (j_starts(nm2))

   !$OMP PARALLEL DO
   do i = 1, nm1
      i_starts(i) = sum(n1(:i)) - n1(i)
   end do
   !$OMP END PARALLEL DO

   !$OMP PARALLEL DO
   do j = 1, nm2
      j_starts(j) = sum(n2(:j)) - n2(j)
   end do
   !$OMP END PARALLEL DO

   inv_sigma2(:) = -1.0d0/sigmas(:)
   kernels(:, :, :) = 0.0d0

   allocate (atomic_distance(maxval(n1), maxval(n2)))
   atomic_distance(:, :) = 0.0d0

   !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj) SCHEDULE(dynamic) COLLAPSE(2)
   do a = 1, nm1
      do b = 1, nm2
         nj = n2(b)
         ni = n1(a)

         atomic_distance(:, :) = 0.0d0
         do i = 1, ni
            do j = 1, nj

               atomic_distance(i, j) = sum(abs(q1(:, i + i_starts(a)) - q2(:, j + j_starts(b))))

            end do
         end do

         do k = 1, nsigmas
            kernels(k, a, b) = sum(exp(atomic_distance(:ni, :nj)*inv_sigma2(k)))
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (atomic_distance)
   deallocate (i_starts)
   deallocate (j_starts)

end subroutine fget_local_kernels_laplacian

subroutine fget_vector_kernels_laplacian(q1, q2, n1, n2, sigmas, &
        & nm1, nm2, nsigmas, rep_size, max_atoms, kernels) bind(C, name="fget_vector_kernels_laplacian")

   use, intrinsic :: iso_c_binding
   implicit none

   ! Array dimensions
   integer(c_int), intent(in), value :: nm1
   integer(c_int), intent(in), value :: nm2
   integer(c_int), intent(in), value :: nsigmas
   integer(c_int), intent(in), value :: rep_size
   integer(c_int), intent(in), value :: max_atoms

   ! Descriptors for the training set (rep_size, max_atoms, nm)
   double precision, dimension(rep_size, max_atoms, nm1), intent(in) :: q1
   double precision, dimension(rep_size, max_atoms, nm2), intent(in) :: q2

   ! List of numbers of atoms in each molecule
   integer, dimension(nm1), intent(in) :: n1
   integer, dimension(nm2), intent(in) :: n2

   ! Sigma in the Gaussian kernel
   double precision, dimension(nsigmas), intent(in) :: sigmas

   ! -1.0 / sigma^2 for use in the kernel
   double precision, dimension(nsigmas) :: inv_sigma

   ! Resulting alpha vector
   double precision, dimension(nsigmas, nm1, nm2), intent(out) :: kernels

   ! Internal counters
   integer :: i, j, k, ni, nj, ia, ja

   ! Temporary variables necessary for parallelization
   double precision, allocatable, dimension(:, :) :: atomic_distance

   inv_sigma(:) = -1.0d0/sigmas(:)

   kernels(:, :, :) = 0.0d0

   allocate (atomic_distance(maxval(n1), maxval(n2)))
   atomic_distance(:, :) = 0.0d0

   !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj) SCHEDULE(dynamic) COLLAPSE(2)
   do j = 1, nm2
      do i = 1, nm1
         ni = n1(i)
         nj = n2(j)

         atomic_distance(:, :) = 0.0d0

         do ja = 1, nj
            do ia = 1, ni

               atomic_distance(ia, ja) = sum(abs(q1(:, ia, i) - q2(:, ja, j)))

            end do
         end do

         do k = 1, nsigmas
            kernels(k, i, j) = sum(exp(atomic_distance(:ni, :nj)*inv_sigma(k)))
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (atomic_distance)

end subroutine fget_vector_kernels_laplacian

subroutine fget_vector_kernels_gaussian(q1, q2, n1, n2, sigmas, &
        & nm1, nm2, nsigmas, rep_size, max_atoms, kernels) bind(C, name="fget_vector_kernels_gaussian")

   use, intrinsic :: iso_c_binding
   implicit none

   ! Array dimensions
   integer(c_int), intent(in), value :: nm1
   integer(c_int), intent(in), value :: nm2
   integer(c_int), intent(in), value :: nsigmas
   integer(c_int), intent(in), value :: rep_size
   integer(c_int), intent(in), value :: max_atoms

   ! Representations (rep_size, max_atoms, nm)
   double precision, dimension(rep_size, max_atoms, nm1), intent(in) :: q1
   double precision, dimension(rep_size, max_atoms, nm2), intent(in) :: q2

   ! List of numbers of atoms in each molecule
   integer, dimension(nm1), intent(in) :: n1
   integer, dimension(nm2), intent(in) :: n2

   ! Sigma in the Gaussian kernel
   double precision, dimension(nsigmas), intent(in) :: sigmas

   ! -1.0 / sigma^2 for use in the kernel
   double precision, dimension(nsigmas) :: inv_sigma2

   ! Resulting alpha vector
   double precision, dimension(nsigmas, nm1, nm2), intent(out) :: kernels

   ! Internal counters
   integer :: i, j, k, ni, nj, ia, ja

   ! Temporary variables necessary for parallelization
   double precision, allocatable, dimension(:, :) :: atomic_distance

   inv_sigma2(:) = -0.5d0/(sigmas(:))**2

   kernels(:, :, :) = 0.0d0

   allocate (atomic_distance(maxval(n1), maxval(n2)))
   atomic_distance(:, :) = 0.0d0

   !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj,ja,ia) SCHEDULE(dynamic) COLLAPSE(2)
   do j = 1, nm2
      do i = 1, nm1
         ni = n1(i)
         nj = n2(j)

         atomic_distance(:, :) = 0.0d0

         do ja = 1, nj
            do ia = 1, ni

               atomic_distance(ia, ja) = sum((q1(:, ia, i) - q2(:, ja, j))**2)

            end do
         end do

         do k = 1, nsigmas
            kernels(k, i, j) = sum(exp(atomic_distance(:ni, :nj)*inv_sigma2(k)))
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (atomic_distance)

end subroutine fget_vector_kernels_gaussian

subroutine fget_vector_kernels_gaussian_symmetric(q, n, sigmas, &
        & nm, nsigmas, rep_size, max_atoms, kernels) bind(C, name="fget_vector_kernels_gaussian_symmetric")

   use, intrinsic :: iso_c_binding
   implicit none

   ! Array dimensions
   integer(c_int), intent(in), value :: nm
   integer(c_int), intent(in), value :: nsigmas
   integer(c_int), intent(in), value :: rep_size
   integer(c_int), intent(in), value :: max_atoms

   ! Representations (rep_size, max_atoms, nm)
   double precision, dimension(rep_size, max_atoms, nm), intent(in) :: q

   ! List of numbers of atoms in each molecule
   integer, dimension(nm), intent(in) :: n

   ! Sigma in the Gaussian kernel
   double precision, dimension(nsigmas), intent(in) :: sigmas

   ! Resulting kernels
   double precision, dimension(nsigmas, nm, nm), intent(out) :: kernels

   ! Temporary variables necessary for parallelization
   double precision, allocatable, dimension(:, :) :: atomic_distance
   double precision, allocatable, dimension(:) :: inv_sigma2

   ! Internal counters
   integer :: i, j, k, ni, nj, ia, ja
   double precision :: val

   allocate (inv_sigma2(nsigmas))

   inv_sigma2 = -0.5d0/(sigmas)**2

   kernels = 1.0d0

   allocate (atomic_distance(max_atoms, max_atoms))
   atomic_distance(:, :) = 0.0d0

   !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj,ja,ia,val) SCHEDULE(dynamic) COLLAPSE(2)
   do j = 1, nm
      do i = 1, nm
         if (i .lt. j) cycle
         ni = n(i)
         nj = n(j)

         atomic_distance(:, :) = 0.0d0

         do ja = 1, nj
            do ia = 1, ni

               atomic_distance(ia, ja) = sum((q(:, ia, i) - q(:, ja, j))**2)

            end do
         end do

         do k = 1, nsigmas
            val = sum(exp(atomic_distance(:ni, :nj)*inv_sigma2(k)))
            kernels(k, i, j) = val
            kernels(k, j, i) = val
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (atomic_distance)
   deallocate (inv_sigma2)

end subroutine fget_vector_kernels_gaussian_symmetric

subroutine fget_vector_kernels_laplacian_symmetric(q, n, sigmas, &
        & nm, nsigmas, rep_size, max_atoms, kernels) bind(C, name="fget_vector_kernels_laplacian_symmetric")

   use, intrinsic :: iso_c_binding
   implicit none

   ! Array dimensions
   integer(c_int), intent(in), value :: nm
   integer(c_int), intent(in), value :: nsigmas
   integer(c_int), intent(in), value :: rep_size
   integer(c_int), intent(in), value :: max_atoms

   ! Representations (rep_size, max_atoms, nm)
   double precision, dimension(rep_size, max_atoms, nm), intent(in) :: q

   ! List of numbers of atoms in each molecule
   integer, dimension(nm), intent(in) :: n

   ! Sigma in the Laplacian kernel
   double precision, dimension(nsigmas), intent(in) :: sigmas

   ! Resulting kernels
   double precision, dimension(nsigmas, nm, nm), intent(out) :: kernels

   ! Temporary variables necessary for parallelization
   double precision, allocatable, dimension(:, :) :: atomic_distance
   double precision, allocatable, dimension(:) :: inv_sigma2

   ! Internal counters
   integer :: i, j, k, ni, nj, ia, ja
   double precision :: val

   allocate (inv_sigma2(nsigmas))

   inv_sigma2 = -1.0d0/sigmas

   kernels = 1.0d0

   allocate (atomic_distance(max_atoms, max_atoms))
   atomic_distance(:, :) = 0.0d0

   !$OMP PARALLEL DO PRIVATE(atomic_distance,ni,nj,ja,ia,val) SCHEDULE(dynamic) COLLAPSE(2)
   do j = 1, nm
      do i = 1, nm
         if (i .lt. j) cycle
         ni = n(i)
         nj = n(j)

         atomic_distance(:, :) = 0.0d0

         do ja = 1, nj
            do ia = 1, ni

               atomic_distance(ia, ja) = sum(abs(q(:, ia, i) - q(:, ja, j)))

            end do
         end do

         do k = 1, nsigmas
            val = sum(exp(atomic_distance(:ni, :nj)*inv_sigma2(k)))
            kernels(k, i, j) = val
            kernels(k, j, i) = val
         end do

      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (atomic_distance)
   deallocate (inv_sigma2)

end subroutine fget_vector_kernels_laplacian_symmetric

subroutine fgaussian_kernel(a, na, b, nb, k, sigma, rep_size) bind(C, name="fgaussian_kernel")

   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), intent(in), value :: na, nb, rep_size

   double precision, dimension(rep_size, na), intent(in) :: a
   double precision, dimension(rep_size, nb), intent(in) :: b

   double precision, dimension(na, nb), intent(inout) :: k
   double precision, intent(in), value :: sigma

   double precision, allocatable, dimension(:) :: temp

   double precision :: inv_sigma
   integer :: i, j

   inv_sigma = -0.5d0/(sigma*sigma)

   allocate (temp(rep_size))

   !$OMP PARALLEL DO PRIVATE(temp) COLLAPSE(2)
   do i = 1, nb
      do j = 1, na
         temp(:) = a(:, j) - b(:, i)
         k(j, i) = exp(inv_sigma*dot_product(temp, temp))
      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (temp)

end subroutine fgaussian_kernel

subroutine fgaussian_kernel_symmetric(x, n, k, sigma, rep_size) bind(C, name="fgaussian_kernel_symmetric")

   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), intent(in), value :: n, rep_size

   double precision, dimension(rep_size, n), intent(in) :: x

   double precision, dimension(n, n), intent(inout) :: k
   double precision, intent(in), value :: sigma

   double precision, allocatable, dimension(:) :: temp
   double precision :: val

   double precision :: inv_sigma
   integer :: i, j

   inv_sigma = -0.5d0/(sigma*sigma)

   k = 1.0d0

   allocate (temp(rep_size))

   !$OMP PARALLEL DO PRIVATE(temp, val) SCHEDULE(dynamic)
   do i = 1, n
      do j = i, n
         temp = x(:, j) - x(:, i)
         val = exp(inv_sigma*dot_product(temp, temp))
         k(j, i) = val
         k(i, j) = val
      end do
   end do
   !$OMP END PARALLEL DO

   deallocate (temp)

end subroutine fgaussian_kernel_symmetric

subroutine flaplacian_kernel(a, na, b, nb, k, sigma, rep_size) bind(C, name="flaplacian_kernel")

   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), intent(in), value :: na, nb, rep_size

   double precision, dimension(rep_size, na), intent(in) :: a
   double precision, dimension(rep_size, nb), intent(in) :: b

   double precision, dimension(na, nb), intent(inout) :: k
   double precision, intent(in), value :: sigma

   double precision :: inv_sigma

   integer :: i, j

   inv_sigma = -1.0d0/sigma

   !$OMP PARALLEL DO COLLAPSE(2)
   do i = 1, nb
      do j = 1, na
         k(j, i) = exp(inv_sigma*sum(abs(a(:, j) - b(:, i))))
      end do
   end do
   !$OMP END PARALLEL DO

end subroutine flaplacian_kernel

subroutine flaplacian_kernel_symmetric(x, n, k, sigma, rep_size) bind(C, name="flaplacian_kernel_symmetric")

   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), intent(in), value :: n, rep_size

   double precision, dimension(rep_size, n), intent(in) :: x

   double precision, dimension(n, n), intent(inout) :: k
   double precision, intent(in), value :: sigma

   double precision :: val

   double precision :: inv_sigma
   integer :: i, j

   inv_sigma = -1.0d0/sigma

   k = 1.0d0

   !$OMP PARALLEL DO PRIVATE(val) SCHEDULE(dynamic)
   do i = 1, n
      do j = i, n
         val = exp(inv_sigma*sum(abs(x(:, j) - x(:, i))))
         k(j, i) = val
         k(i, j) = val
      end do
   end do
   !$OMP END PARALLEL DO

end subroutine flaplacian_kernel_symmetric

subroutine flinear_kernel(a, na, b, nb, k, rep_size) bind(C, name="flinear_kernel")

   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), intent(in), value :: na, nb, rep_size

   double precision, dimension(rep_size, na), intent(in) :: a
   double precision, dimension(rep_size, nb), intent(in) :: b

   double precision, dimension(na, nb), intent(inout) :: k

   integer :: i, j

!$OMP PARALLEL DO COLLAPSE(2)
   do i = 1, nb
      do j = 1, na
         k(j, i) = dot_product(a(:, j), b(:, i))
      end do
   end do
!$OMP END PARALLEL DO

end subroutine flinear_kernel

subroutine fmatern_kernel_l2(a, na, b, nb, k, sigma, order, rep_size) bind(C, name="fmatern_kernel_l2")

   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), intent(in), value :: na, nb, order, rep_size

   double precision, dimension(rep_size, na), intent(in) :: a
   double precision, dimension(rep_size, nb), intent(in) :: b

   double precision, dimension(na, nb), intent(inout) :: k
   double precision, intent(in), value :: sigma

   double precision, allocatable, dimension(:) :: temp

   double precision :: inv_sigma, inv_sigma2, d, d2
   integer :: i, j

   allocate (temp(rep_size))

   if (order == 0) then
      inv_sigma = -1.0d0/sigma

      !$OMP PARALLEL DO PRIVATE(temp) COLLAPSE(2)
      do i = 1, nb
         do j = 1, na
            temp(:) = a(:, j) - b(:, i)
            k(j, i) = exp(inv_sigma*sqrt(sum(temp*temp)))
         end do
      end do
      !$OMP END PARALLEL DO
   else if (order == 1) then
      inv_sigma = -sqrt(3.0d0)/sigma

      !$OMP PARALLEL DO PRIVATE(temp, d) COLLAPSE(2)
      do i = 1, nb
         do j = 1, na
            temp(:) = a(:, j) - b(:, i)
            d = sqrt(sum(temp*temp))
            k(j, i) = exp(inv_sigma*d)*(1.0d0 - inv_sigma*d)
         end do
      end do
      !$OMP END PARALLEL DO
   else
      inv_sigma = -sqrt(5.0d0)/sigma
      inv_sigma2 = 5.0d0/(3.0d0*sigma*sigma)

      !$OMP PARALLEL DO PRIVATE(temp, d, d2) COLLAPSE(2)
      do i = 1, nb
         do j = 1, na
            temp(:) = a(:, j) - b(:, i)
            d2 = sum(temp*temp)
            d = sqrt(d2)
            k(j, i) = exp(inv_sigma*d)*(1.0d0 - inv_sigma*d + inv_sigma2*d2)
         end do
      end do
      !$OMP END PARALLEL DO
   end if

   deallocate (temp)

end subroutine fmatern_kernel_l2

subroutine fsargan_kernel(a, na, b, nb, k, sigma, gammas, ng, rep_size) bind(C, name="fsargan_kernel")

   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), intent(in), value :: na, nb, ng, rep_size

   double precision, dimension(rep_size, na), intent(in) :: a
   double precision, dimension(rep_size, nb), intent(in) :: b
   double precision, dimension(ng), intent(in) :: gammas

   double precision, dimension(na, nb), intent(inout) :: k
   double precision, intent(in), value :: sigma

   double precision, allocatable, dimension(:) :: prefactor
   double precision :: inv_sigma
   double precision :: d

   integer :: i, j, m

   inv_sigma = -1.0d0/sigma

   ! Allocate temporary
   allocate (prefactor(ng))

   !$OMP PARALLEL DO PRIVATE(d, prefactor) SCHEDULE(dynamic) COLLAPSE(2)
   do i = 1, nb
      do j = 1, na
         d = sum(abs(a(:, j) - b(:, i)))
         do m = 1, ng
            prefactor(m) = gammas(m)*(-inv_sigma*d)**m
         end do
         k(j, i) = exp(inv_sigma*d)*(1 + sum(prefactor(:)))
      end do
   end do
   !$OMP END PARALLEL DO

   ! Clean up
   deallocate (prefactor)

end subroutine fsargan_kernel
