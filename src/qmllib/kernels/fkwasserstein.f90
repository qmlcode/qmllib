
module searchtools

   implicit none

contains

   subroutine searchsorted(all_values, sorted, cdf_idx)
      !function searchsorted(all_values, sorted) result(cdf_idx)

      implicit none

      double precision, dimension(:), intent(in) :: all_values
      double precision, dimension(:), intent(in) :: sorted

      integer, dimension(:), intent(out) :: cdf_idx
      !integer, allocatable, dimension(:) :: cdf_idx
!      integer, allocatable, dimension(:) :: searchsorted

      double precision :: val

      integer :: i, j, n, m

      n = size(all_values) - 1
      m = size(sorted)

      !allocate (cdf_idx(n))

      cdf_idx(:) = 0

      do i = 1, n

         val = all_values(i)

         do j = 1, m

            !write (*,*) i, j, sorted(j), val

            ! if ((sorted(j) <= val) .and. (val < sorted(j+1)))  then
            if (sorted(j) > val) then

               cdf_idx(i) = j - 1
               !write(*,*) "found"
               exit

               !  endif
            else !iif (val > maxval(sorted)) then
               cdf_idx(i) = m
            end if

         end do

      end do

   end subroutine searchsorted
   !end function searchsorted

   recursive subroutine quicksort(a, first, last)
      implicit none
      double precision ::  a(*), x, t
      integer first, last
      integer i, j

      x = a((first + last)/2)
      i = first
      j = last
      do
         do while (a(i) < x)
            i = i + 1
         end do
         do while (x < a(j))
            j = j - 1
         end do
         if (i >= j) exit
         t = a(i); a(i) = a(j); a(j) = t
         i = i + 1
         j = j - 1
      end do
      if (first < i - 1) call quicksort(a, first, i - 1)
      if (j + 1 < last) call quicksort(a, j + 1, last)
   end subroutine quicksort

end module searchtools

subroutine fwasserstein_kernel(a, na, b, nb, k, sigma, p, q)

   use searchtools
   implicit none

   double precision, dimension(:, :), intent(in) :: a
   double precision, dimension(:, :), intent(in) :: b

   double precision, allocatable, dimension(:, :) :: asorted
   double precision, allocatable, dimension(:, :) :: bsorted

   double precision, allocatable, dimension(:) :: rep

   integer, intent(in) :: na, nb

   double precision, dimension(:, :), intent(inout) :: k
   double precision, intent(in) :: sigma

   integer, intent(in) :: p
   integer, intent(in) :: q

   double precision :: inv_sigma

   integer :: i, j, l
   integer :: rep_size

   double precision, allocatable, dimension(:) :: deltas
   double precision, allocatable, dimension(:) :: all_values

   double precision, allocatable, dimension(:) :: a_cdf
   double precision, allocatable, dimension(:) :: b_cdf
   integer, allocatable, dimension(:) :: a_cdf_idx
   integer, allocatable, dimension(:) :: b_cdf_idx

   rep_size = size(a, dim=1)
   allocate (asorted(rep_size, na))
   allocate (bsorted(rep_size, nb))
   allocate (rep(rep_size))

   allocate (all_values(rep_size*2))
   allocate (deltas(rep_size*2 - 1))

   allocate (a_cdf(rep_size*2 - 1))
   allocate (b_cdf(rep_size*2 - 1))

   allocate (a_cdf_idx(rep_size*2 - 1))
   allocate (b_cdf_idx(rep_size*2 - 1))

   asorted(:, :) = a(:, :)
   bsorted(:, :) = b(:, :)

   do i = 1, na
      rep(:) = asorted(:, i)
      call quicksort(rep, 1, rep_size)
      asorted(:, i) = rep(:)
   end do

   do i = 1, nb
      rep(:) = bsorted(:, i)
      call quicksort(rep, 1, rep_size)
      bsorted(:, i) = rep(:)
   end do

   !$OMP PARALLEL DO PRIVATE(all_values,a_cdf_idx,b_cdf_idx,a_cdf,b_cdf,deltas)
   do j = 1, nb
      do i = 1, na

         all_values(:rep_size) = asorted(:, i)
         all_values(rep_size + 1:) = bsorted(:, j)

         call quicksort(all_values, 1, 2*rep_size)

         do l = 1, 2*rep_size - 1
            deltas(l) = all_values(l + 1) - all_values(l)
         end do

         !a_cdf_idx = searchsorted(all_values, asorted(:, i))
         !b_cdf_idx = searchsorted(all_values, bsorted(:, j))
         call searchsorted(all_values, asorted(:, i), a_cdf_idx)
         call searchsorted(all_values, bsorted(:, j), b_cdf_idx)

         a_cdf(:) = a_cdf_idx(:)
         b_cdf(:) = b_cdf_idx(:)
         a_cdf(:) = a_cdf(:)/rep_size
         b_cdf(:) = b_cdf(:)/rep_size

         ! k(i,j) = exp(-sum(abs(a_cdf-b_cdf)*deltas)/sigma)
         k(i, j) = exp(-(sum((abs(a_cdf - b_cdf)**p)*deltas)**(1.0d0/p))**q/sigma)

      end do
   end do
   !$OMP END PARALLEL DO
end subroutine fwasserstein_kernel
