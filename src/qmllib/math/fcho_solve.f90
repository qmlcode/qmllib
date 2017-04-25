
!

!






!


!








subroutine fcho_solve(A,y,x)

    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, dimension(:), intent(in) :: y
    double precision, dimension(:), intent(inout) :: x
    
    integer :: info, na

    na = size(A, dim=1)

    call dpotrf("U", na, A, na, info)
    if (info > 0) then
        write (*,*) "WARNING: Cholesky decomposition DPOTRF() exited with error code:", info
    endif

    x(:na) = y(:na)

    call dpotrs("U", na, 1, A, na, x, na, info)
    if (info > 0) then
        write (*,*) "WARNING: Cholesky solve DPOTRS() exited with error code:", info
    endif

end subroutine fcho_solve

subroutine fcho_invert(A)

    implicit none

    double precision, dimension(:,:), intent(inout) :: A
    integer :: info, na

    na = size(A, dim=1)

    call dpotrf("U", na, A , na, info)
    if (info > 0) then
        write (*,*) "WARNING: Cholesky decomposition DPOTRF() exited with error code:", info
    endif

    call dpotri("U", na, A , na, info )
    if (info > 0) then
        write (*,*) "WARNING: Cholesky inversion DPOTRI() exited with error code:", info
    endif

end subroutine fcho_invert
