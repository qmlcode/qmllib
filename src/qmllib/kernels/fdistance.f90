subroutine fmanhattan_distance(A, nv, na, B, nb, D) bind(C, name="fmanhattan_distance")
   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), value :: nv, na, nb
   real(c_double), intent(in) :: A(nv, na)
   real(c_double), intent(in) :: B(nv, nb)
   real(c_double), intent(inout) :: D(na, nb)

   integer :: i, j

   ! Validate input
   if (na <= 0 .OR. nb <= 0 .OR. nv <= 0) then
      write (*, *) "ERROR: Manhattan distance"
      write (*, *) "nv=", nv, "na=", na, "nb=", nb
      write (*, *) "All dimensions must be positive"
      stop
   end if

!$OMP PARALLEL DO
   do i = 1, nb
      do j = 1, na
         D(j, i) = sum(abs(a(:, j) - b(:, i)))
      end do
   end do
!$OMP END PARALLEL DO

end subroutine fmanhattan_distance

subroutine fl2_distance(A, nv, na, B, nb, D) bind(C, name="fl2_distance")
   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), value :: nv, na, nb
   real(c_double), intent(in) :: A(nv, na)
   real(c_double), intent(in) :: B(nv, nb)
   real(c_double), intent(inout) :: D(na, nb)

   integer :: i, j

   double precision, allocatable, dimension(:) :: temp

   ! Validate input
   if (na <= 0 .OR. nb <= 0 .OR. nv <= 0) then
      write (*, *) "ERROR: L2 distance"
      write (*, *) "nv=", nv, "na=", na, "nb=", nb
      write (*, *) "All dimensions must be positive"
      stop
   end if

   allocate (temp(nv))

!$OMP PARALLEL DO PRIVATE(temp)
   do i = 1, nb
      do j = 1, na
         temp(:) = A(:, j) - B(:, i)
         D(j, i) = sqrt(sum(temp*temp))
      end do
   end do
!$OMP END PARALLEL DO

   deallocate (temp)

end subroutine fl2_distance

subroutine fp_distance_double(A, nv, na, B, nb, D, p) bind(C, name="fp_distance_double")
   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), value :: nv, na, nb
   real(c_double), intent(in) :: A(nv, na)
   real(c_double), intent(in) :: B(nv, nb)
   real(c_double), intent(inout) :: D(na, nb)
   real(c_double), value :: p

   integer :: i, j

   double precision, allocatable, dimension(:) :: temp
   double precision :: inv_p

   ! Validate input
   if (na <= 0 .OR. nb <= 0 .OR. nv <= 0) then
      write (*, *) "ERROR: Lp distance (double)"
      write (*, *) "nv=", nv, "na=", na, "nb=", nb
      write (*, *) "All dimensions must be positive"
      stop
   end if

   inv_p = 1.0d0/p

   allocate (temp(nv))

!$OMP PARALLEL DO PRIVATE(temp)
   do i = 1, nb
      do j = 1, na
         temp(:) = abs(A(:, j) - B(:, i))
         D(j, i) = (sum(temp**p))**inv_p
      end do
   end do
!$OMP END PARALLEL DO

   deallocate (temp)

end subroutine fp_distance_double

subroutine fp_distance_integer(A, nv, na, B, nb, D, p) bind(C, name="fp_distance_integer")
   use, intrinsic :: iso_c_binding
   implicit none

   integer(c_int), value :: nv, na, nb, p
   real(c_double), intent(in) :: A(nv, na)
   real(c_double), intent(in) :: B(nv, nb)
   real(c_double), intent(inout) :: D(na, nb)

   integer :: i, j

   double precision, allocatable, dimension(:) :: temp
   double precision :: inv_p

   ! Validate input
   if (na <= 0 .OR. nb <= 0 .OR. nv <= 0) then
      write (*, *) "ERROR: Lp distance (integer)"
      write (*, *) "nv=", nv, "na=", na, "nb=", nb
      write (*, *) "All dimensions must be positive"
      stop
   end if

   inv_p = 1.0d0/dble(p)

   allocate (temp(nv))

!$OMP PARALLEL DO PRIVATE(temp)
   do i = 1, nb
      do j = 1, na
         temp(:) = abs(A(:, j) - B(:, i))
         D(j, i) = (sum(temp**p))**inv_p
      end do
   end do
!$OMP END PARALLEL DO

   deallocate (temp)

end subroutine fp_distance_integer
