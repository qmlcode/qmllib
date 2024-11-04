
   subroutine check_openmp(compiled_with_openmp)

      implicit none
      logical, intent(out):: compiled_with_openmp

      compiled_with_openmp = .false.

!$    compiled_with_openmp = .true.

   end subroutine check_openmp

   function get_threads() result(nt)
!$    use omp_lib
      implicit none
      integer :: nt

      nt = 0
!$    nt = omp_get_max_threads()

   end function get_threads
