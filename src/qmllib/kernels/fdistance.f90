
subroutine fmanhattan_distance(A, B, D)

   implicit none

   double precision, dimension(:, :), intent(in) :: A
   double precision, dimension(:, :), intent(in) :: B
   double precision, dimension(:, :), intent(inout) :: D

   integer :: na, nb
   integer :: i, j

   na = size(A, dim=2)
   nb = size(B, dim=2)

!$OMP PARALLEL DO
   do i = 1, nb
      do j = 1, na
         D(j, i) = sum(abs(a(:, j) - b(:, i)))
      end do
   end do
!$OMP END PARALLEL DO

end subroutine fmanhattan_distance

subroutine fl2_distance(A, B, D)

   implicit none

   double precision, dimension(:, :), intent(in) :: A
   double precision, dimension(:, :), intent(in) :: B
   double precision, dimension(:, :), intent(inout) :: D

   integer :: na, nb, nv
   integer :: i, j

   double precision, allocatable, dimension(:) :: temp

   nv = size(A, dim=1)

   na = size(A, dim=2)
   nb = size(B, dim=2)

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

subroutine fp_distance_double(A, B, D, p)

   implicit none

   double precision, dimension(:, :), intent(in) :: A
   double precision, dimension(:, :), intent(in) :: B
   double precision, dimension(:, :), intent(inout) :: D
   double precision, intent(in) :: p

   integer :: na, nb, nv
   integer :: i, j

   double precision, allocatable, dimension(:) :: temp
   double precision :: inv_p

   nv = size(A, dim=1)

   na = size(A, dim=2)
   nb = size(B, dim=2)

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

subroutine fp_distance_integer(A, B, D, p)

   implicit none

   double precision, dimension(:, :), intent(in) :: A
   double precision, dimension(:, :), intent(in) :: B
   double precision, dimension(:, :), intent(inout) :: D
   integer, intent(in) :: p

   integer :: na, nb, nv
   integer :: i, j

   double precision, allocatable, dimension(:) :: temp
   double precision :: inv_p

   nv = size(A, dim=1)

   na = size(A, dim=2)
   nb = size(B, dim=2)

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
