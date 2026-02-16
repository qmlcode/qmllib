   subroutine check_openmp(compiled_with_openmp) bind(C, name="check_openmp")
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int), intent(out) :: compiled_with_openmp

      compiled_with_openmp = 0

!$    compiled_with_openmp = 1

   end subroutine check_openmp

   function get_threads() result(nt) bind(C, name="get_threads")
!$    use omp_lib
      use, intrinsic :: iso_c_binding
      implicit none
      integer(c_int) :: nt

      nt = 0
!$    nt = omp_get_max_threads()

   end function get_threads
