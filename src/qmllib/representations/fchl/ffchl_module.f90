module ffchl_module

   implicit none

contains

   function get_pmax(x, na) result(pmax)

      implicit none

      ! FCHL descriptors for the set, format (nm1,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :, :), intent(in) :: x

      ! List of numbers of atoms in each molecule
      integer, dimension(:), intent(in) :: na

      ! Internal counter
      integer :: a

      ! Max nuclear charge
      integer :: pmax

      pmax = 0

      do a = 1, size(na, dim=1)
         pmax = max(pmax, int(maxval(x(a, 1, 2, :na(a)))))
      end do

   end function get_pmax

   function get_pmax_atomic(x, nneigh) result(pmax)

      implicit none

      ! FCHL descriptors for the set, format (nm1,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :), intent(in) :: x

      ! List of numbers of atoms in each molecule
      integer, dimension(:), intent(in) :: nneigh

      ! Internal counter
      integer :: a

      ! Max nuclear charge
      integer :: pmax

      pmax = 0

      do a = 1, size(nneigh, dim=1)
         pmax = max(pmax, int(maxval(x(a, 2, :nneigh(a)))))
      end do

   end function get_pmax_atomic

   function get_pmax_displaced(x, na) result(pmax)

      implicit none

      ! FCHL descriptors for the training set, format (nm1,3,2,maxatoms,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :, :, :, :, :), intent(in) :: x

      ! List of numbers of atoms in each molecule
      integer, dimension(:), intent(in) :: na

      ! Internal counter
      integer :: a, b, i, xyz, pm

      ! Max nuclear charge
      integer :: pmax

      pmax = 0

      do a = 1, size(na, dim=1)
         b = na(a)
         do xyz = 1, 3
            do pm = 1, size(x, dim=3)
               do i = 1, b
                  pmax = max(pmax, int(maxval(x(a, xyz, pm, i, 1, 2, :na(a)))))
               end do
            end do
         end do
      end do

   end function get_pmax_displaced

   pure function cut_function(r, cut_start, cut_distance) result(f)

      implicit none

      ! Distance
      double precision, intent(in) :: r

      ! Lower limit of damping
      double precision, intent(in) :: cut_start

      ! Upper limit of damping
      double precision, intent(in) :: cut_distance

      ! Intermediate variables
      double precision :: x
      double precision :: rl
      double precision :: ru

      ! Damping function at distance r
      double precision :: f

      ru = cut_distance
      rl = cut_start*cut_distance

      if (r > ru) then

         f = 0.0d0

      else if (r < rl) then

         f = 1.0d0

      else

         x = (ru - r)/(ru - rl)
         f = (10.0d0*x**3) - (15.0d0*x**4) + (6.0d0*x**5)

      end if

   end function cut_function

   pure function get_angular_norm2(t_width) result(ang_norm2)

      implicit none

      ! Theta-width
      double precision, intent(in) :: t_width

      ! The resulting angular norm (squared)
      double precision ang_norm2

      ! Integration limit - bigger than 100 should suffice.
      integer, parameter :: limit = 10000

      ! Pi at double precision.
      double precision, parameter :: pi = 4.0d0*atan(1.0d0)

      integer :: n

      ang_norm2 = 0.0d0

      do n = -limit, limit
         ang_norm2 = ang_norm2 + exp(-((t_width*n)**2)) &
                    & *(2.0d0 - 2.0d0*cos(n*pi))
      end do

      ang_norm2 = sqrt(ang_norm2*pi)*2.0d0

   end function

   function get_twobody_weights(x, neighbors, power, cut_start, cut_distance, dim1) result(ksi)

      implicit none

      double precision, dimension(:, :), intent(in) :: x
      integer, intent(in) :: neighbors
      double precision, intent(in) :: power
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance
      integer, intent(in) :: dim1

      double precision, dimension(dim1) :: ksi
      integer :: i

      ksi = 0.0d0

      do i = 2, neighbors

         ksi(i) = cut_function(x(1, i), cut_start, cut_distance)/x(1, i)**power

      end do

   end function get_twobody_weights

! Calculate the Fourier terms for the FCHL three-body expansion
   function get_threebody_fourier(x, neighbors, order, power, cut_start, cut_distance, &
       & dim1, dim2, dim3) result(fourier)

      implicit none

      ! Input representation, dimension=(5,n).
      double precision, dimension(:, :), intent(in) :: x

      ! Number of neighboring atoms to iterate over.
      integer, intent(in) :: neighbors

      ! Fourier-expansion order.
      integer, intent(in) :: order

      ! Power law
      double precision, intent(in) :: power

      ! Lower limit of damping function
      double precision, intent(in) :: cut_start

      ! Upper limit of damping function
      double precision, intent(in) :: cut_distance

      ! Dimensions or the output array.
      integer, intent(in) :: dim1, dim2, dim3

      ! dim(1,:,:,:) are cos terms, dim(2,:,:,:) are sine terms.
      double precision, dimension(2, dim1, dim2, dim3) :: fourier

      ! Pi at double precision.
      double precision, parameter :: pi = 4.0d0*atan(1.0d0)

      ! Internal counters.
      integer :: j, k, m

      ! Indexes for the periodic-table distance matrix.
      integer :: pj, pk

      ! Angle between atoms for the three-body term.
      double precision :: theta

      ! Three-body weight
      double precision :: ksi3

      ! Temporary variables for cos and sine Fourier terms.
      double precision :: cos_m, sin_m

      fourier = 0.0d0

      do j = 2, neighbors
         do k = j + 1, neighbors

            ksi3 = calc_ksi3(X(:, :), j, k, power, cut_start, cut_distance)
            theta = calc_angle(x(3:5, j), x(3:5, 1), x(3:5, k))

            pj = int(x(2, k))
            pk = int(x(2, j))

            do m = 1, order

               cos_m = (cos(m*theta) - cos((theta + pi)*m))*ksi3
               sin_m = (sin(m*theta) - sin((theta + pi)*m))*ksi3

               fourier(1, pj, m, j) = fourier(1, pj, m, j) + cos_m
               fourier(2, pj, m, j) = fourier(2, pj, m, j) + sin_m

               fourier(1, pk, m, k) = fourier(1, pk, m, k) + cos_m
               fourier(2, pk, m, k) = fourier(2, pk, m, k) + sin_m

            end do

         end do
      end do

      return

   end function get_threebody_fourier

   pure function calc_angle(a, b, c) result(angle)

      implicit none

      double precision, intent(in), dimension(3) :: a
      double precision, intent(in), dimension(3) :: b
      double precision, intent(in), dimension(3) :: c

      double precision, dimension(3) :: v1
      double precision, dimension(3) :: v2

      double precision :: cos_angle
      double precision :: angle

      v1 = a - b
      v2 = c - b

      v1 = v1/norm2(v1)
      v2 = v2/norm2(v2)

      cos_angle = dot_product(v1, v2)

      ! Clipping
      if (cos_angle > 1.0d0) cos_angle = 1.0d0
      if (cos_angle < -1.0d0) cos_angle = -1.0d0

      angle = acos(cos_angle)

   end function calc_angle

   pure function calc_cos_angle(a, b, c) result(cos_angle)

      implicit none

      double precision, intent(in), dimension(3) :: a
      double precision, intent(in), dimension(3) :: b
      double precision, intent(in), dimension(3) :: c

      double precision, dimension(3) :: v1
      double precision, dimension(3) :: v2

      double precision :: cos_angle

      v1 = a - b
      v2 = c - b

      v1 = v1/norm2(v1)
      v2 = v2/norm2(v2)

      cos_angle = dot_product(v1, v2)

   end function calc_cos_angle

   function calc_ksi3(X, j, k, power, cut_start, cut_distance) result(ksi3)

      implicit none

      double precision, dimension(:, :), intent(in) :: X

      integer, intent(in) :: j
      integer, intent(in) :: k
      double precision, intent(in) :: power
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance

      double precision :: cos_i, cos_j, cos_k
      double precision :: di, dj, dk

      double precision :: ksi3
      double precision :: cut

      cos_i = calc_cos_angle(x(3:5, k), x(3:5, 1), x(3:5, j))
      cos_j = calc_cos_angle(x(3:5, j), x(3:5, k), x(3:5, 1))
      cos_k = calc_cos_angle(x(3:5, 1), x(3:5, j), x(3:5, k))

      dk = x(1, j)
      dj = x(1, k)
      di = norm2(x(3:5, j) - x(3:5, k))

      cut = cut_function(dk, cut_start, cut_distance)* &
          & cut_function(dj, cut_start, cut_distance)* &
          & cut_function(di, cut_start, cut_distance)

      ksi3 = cut*(1.0d0 + 3.0d0*cos_i*cos_j*cos_k)/(di*dj*dk)**power

   end function calc_ksi3

   pure function scalar(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
       & t_width, d_width, cut_distance, order, pd, ang_norm2, &
       & distance_scale, angular_scale, alchemy) result(aadist)

      implicit none

      double precision, dimension(:, :), intent(in) :: X1
      double precision, dimension(:, :), intent(in) :: X2

      integer, intent(in) :: N1
      integer, intent(in) :: N2

      double precision, dimension(:), intent(in) :: ksi1
      double precision, dimension(:), intent(in) :: ksi2

      double precision, dimension(:, :, :), intent(in) :: sin1
      double precision, dimension(:, :, :), intent(in) :: sin2
      double precision, dimension(:, :, :), intent(in) :: cos1
      double precision, dimension(:, :, :), intent(in) :: cos2

      double precision, intent(in) :: t_width
      double precision, intent(in) :: d_width
      double precision, intent(in) :: cut_distance
      integer, intent(in) :: order
      double precision, dimension(:, :), intent(in) :: pd
      double precision, intent(in) :: angular_scale
      double precision, intent(in) :: distance_scale

      double precision :: true_angular_scale
      double precision :: true_distance_scale

      double precision, intent(in):: ang_norm2

      double precision :: aadist

      logical, intent(in) :: alchemy

      ! We changed the convention for the scaling factors when the paper was under review
      ! so this is a quick fix
      true_angular_scale = angular_scale/sqrt(8.0d0)
      true_distance_scale = distance_scale/16.0d0

      if (alchemy) then
         aadist = scalar_alchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
             & t_width, d_width, order, pd, ang_norm2, &
             & true_distance_scale, true_angular_scale)
      else
         aadist = scalar_noalchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
             & t_width, d_width, ang_norm2, &
             & true_distance_scale, true_angular_scale)
      end if

   end function scalar

   pure function scalar_noalchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
       & t_width, d_width, ang_norm2, &
       & distance_scale, angular_scale) result(aadist)

      implicit none

      double precision, dimension(:, :), intent(in) :: X1
      double precision, dimension(:, :), intent(in) :: X2

      integer, intent(in) :: N1
      integer, intent(in) :: N2

      double precision, dimension(:), intent(in) :: ksi1
      double precision, dimension(:), intent(in) :: ksi2

      double precision, dimension(:, :, :), intent(in) :: sin1
      double precision, dimension(:, :, :), intent(in) :: sin2
      double precision, dimension(:, :, :), intent(in) :: cos1
      double precision, dimension(:, :, :), intent(in) :: cos2

      double precision, intent(in) :: t_width
      double precision, intent(in) :: d_width

      double precision, intent(in) :: angular_scale
      double precision, intent(in) :: distance_scale

      double precision :: aadist

      double precision :: d

      integer :: i, j, p
      double precision :: angular
      double precision :: maxgausdist2

      integer :: pmax1
      integer :: pmax2
      integer :: pmax

      double precision :: inv_width
      double precision :: r2

      double precision :: s

      double precision, parameter :: pi = 4.0d0*atan(1.0d0)

      double precision :: g1

      logical, allocatable, dimension(:) :: mask1
      logical, allocatable, dimension(:) :: mask2

      double precision, intent(in):: ang_norm2

      ! cached sine and cos terms
      double precision, allocatable, dimension(:, :) :: cos1c, cos2c, sin1c, sin2c

      if (int(x1(2, 1)) /= int(x2(2, 1))) then
         aadist = 0.0d0
         return
      end if

      pmax1 = int(maxval(x1(2, :n1)))
      pmax2 = int(maxval(x2(2, :n2)))

      pmax = min(pmax1, pmax2)

      allocate (mask1(pmax1))
      allocate (mask2(pmax2))
      mask1 = .true.
      mask2 = .true.

      do i = 1, n1
         mask1(int(x1(2, i))) = .false.
      end do

      do i = 1, n2
         mask2(int(x2(2, i))) = .false.
      end do

      allocate (cos1c(pmax, n1))
      allocate (cos2c(pmax, n2))
      allocate (sin1c(pmax, n1))
      allocate (sin2c(pmax, n2))

      cos1c = 0.0d0
      cos2c = 0.0d0
      sin1c = 0.0d0
      sin2c = 0.0d0

      p = 0

      do i = 1, pmax
         if (mask1(i)) cycle
         if (mask2(i)) cycle

         p = p + 1

         cos1c(p, :n1) = cos1(i, 1, :n1)
         cos2c(p, :n2) = cos2(i, 1, :n2)
         sin1c(p, :n1) = sin1(i, 1, :n1)
         sin2c(p, :n2) = sin2(i, 1, :n2)

      end do

      pmax = p

      ! Pre-computed constants
      g1 = sqrt(2.0d0*pi)/ang_norm2
      s = g1*exp(-(t_width)**2/2.0d0)
      inv_width = -1.0d0/(4.0d0*d_width**2)
      maxgausdist2 = (8.0d0*d_width)**2

      ! Initialize scalar product
      aadist = 1.0d0

      do i = 2, n1
         do j = 2, n2

            if (int(x1(2, i)) /= int(x2(2, j))) cycle

            r2 = (x2(1, j) - x1(1, i))**2
            if (r2 >= maxgausdist2) cycle

            d = exp(r2*inv_width)

            angular = (sum(cos1c(:pmax, i)*cos2c(:pmax, j)) &
                   & + sum(sin1c(:pmax, i)*sin2c(:pmax, j)))*s

            aadist = aadist + d*(ksi1(i)*ksi2(j)*distance_scale &
                & + angular*angular_scale)

         end do
      end do

      deallocate (mask1)
      deallocate (mask2)
      deallocate (cos1c)
      deallocate (cos2c)
      deallocate (sin1c)
      deallocate (sin2c)

   end function scalar_noalchemy

   pure function scalar_alchemy(X1, X2, N1, N2, ksi1, ksi2, sin1, sin2, cos1, cos2, &
       & t_width, d_width, order, pd, ang_norm2, &
       & distance_scale, angular_scale) result(aadist)

      implicit none

      double precision, dimension(:, :), intent(in) :: X1
      double precision, dimension(:, :), intent(in) :: X2

      integer, intent(in) :: N1
      integer, intent(in) :: N2

      double precision, dimension(:), intent(in) :: ksi1
      double precision, dimension(:), intent(in) :: ksi2

      double precision, dimension(:, :, :), intent(in) :: sin1
      double precision, dimension(:, :, :), intent(in) :: sin2
      double precision, dimension(:, :, :), intent(in) :: cos1
      double precision, dimension(:, :, :), intent(in) :: cos2

      double precision, intent(in) :: t_width
      double precision, intent(in) :: d_width
      ! double precision, intent(in) :: cut_distance
      integer, intent(in) :: order
      double precision, dimension(:, :), intent(in) :: pd
      double precision, intent(in) :: angular_scale
      double precision, intent(in) :: distance_scale

      double precision :: aadist

      double precision :: d

      integer :: m_1, m_2

      integer :: i, m, p1, p2

      double precision :: angular
      double precision :: maxgausdist2

      integer :: pmax1
      integer :: pmax2

      double precision :: inv_width
      double precision :: r2

      double precision, dimension(order) :: s

      double precision, parameter :: pi = 4.0d0*atan(1.0d0)

      double precision :: g1
      double precision :: a0

      logical, allocatable, dimension(:) :: mask1
      logical, allocatable, dimension(:) :: mask2

      double precision :: temp, sin1_temp, cos1_temp
      double precision, intent(in):: ang_norm2

      pmax1 = int(maxval(x1(2, :n1)))
      pmax2 = int(maxval(x2(2, :n2)))

      allocate (mask1(pmax1))
      allocate (mask2(pmax2))
      mask1 = .true.
      mask2 = .true.

      do i = 1, n1
         mask1(int(x1(2, i))) = .false.
      end do

      do i = 1, n2
         mask2(int(x2(2, i))) = .false.
      end do

      a0 = 0.0d0
      g1 = sqrt(2.0d0*pi)/ang_norm2

      do m = 1, order
         s(m) = g1*exp(-(t_width*m)**2/2.0d0)
      end do

      inv_width = -1.0d0/(4.0d0*d_width**2)

      maxgausdist2 = (8.0d0*d_width)**2

      aadist = 1.0d0

      do m_1 = 2, N1

         ! if (X1(1, m_1) > cut_distance) exit

         do m_2 = 2, N2

            ! if (X2(1, m_2) > cut_distance) exit

            r2 = (X2(1, m_2) - X1(1, m_1))**2

            if (r2 < maxgausdist2) then

               d = exp(r2*inv_width)*pd(int(x1(2, m_1)), int(x2(2, m_2)))

               angular = a0*a0

               do m = 1, order

                  temp = 0.0d0

                  do p1 = 1, pmax1
                     if (mask1(p1)) cycle
                     cos1_temp = cos1(p1, m, m_1)
                     sin1_temp = sin1(p1, m, m_1)

                     do p2 = 1, pmax2
                        if (mask2(p2)) cycle

                        temp = temp + (cos1_temp*cos2(p2, m, m_2) &
                            & + sin1_temp*sin2(p2, m, m_2))*pd(p2, p1)

                     end do
                  end do

                  angular = angular + temp*s(m)

               end do

               aadist = aadist + d*(ksi1(m_1)*ksi2(m_2)*distance_scale &
                   & + angular*angular_scale)

            end if
         end do
      end do

      aadist = aadist*pd(int(x1(2, 1)), int(x2(2, 1)))

      deallocate (mask1)
      deallocate (mask2)
   end function scalar_alchemy

   subroutine get_ksi(x, na, nneigh, two_body_power, cut_start, cut_distance, verbose, ksi)! result(ksi)

      implicit none

      ! FCHL descriptors for the training set, format (i,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :, :), intent(in) :: x

      ! List of numbers of atoms in each molecule
      integer, dimension(:), intent(in) :: na

      ! Number of neighbors for each atom in each compound
      integer, dimension(:, :), intent(in) :: nneigh

      ! Decaying powerlaws for two-body term
      double precision, intent(in) :: two_body_power

      ! Fraction of cut_distance at which cut-off starts
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance

      ! Whether to be verbose with output
      logical, intent(in) :: verbose

      ! Pre-computed two-body weights
      !double precision, allocatable, dimension(:, :, :) :: ksi
      double precision, dimension(:, :, :) :: ksi

      ! Internal counters
      integer :: maxneigh, maxatoms, nm, a, ni, i

      maxneigh = maxval(nneigh)
      maxatoms = maxval(na)
      nm = size(x, dim=1)

      !allocate (ksi(nm, maxatoms, maxneigh))

      ksi = 0.0d0

      !$OMP PARALLEL DO PRIVATE(ni)
      do a = 1, nm
         ni = na(a)
         do i = 1, ni
            ksi(a, i, :) = get_twobody_weights(x(a, i, :, :), nneigh(a, i), &
                & two_body_power, cut_start, cut_distance, maxneigh)
         end do
      end do
      !$OMP END PARALLEL do

      !end function get_ksi
   end subroutine get_ksi

   !function get_ksi_displaced(x, na, nneigh, two_body_power, cut_start, cut_distance, verbose) result(ksi)
   subroutine get_ksi_displaced(x, na, nneigh, two_body_power, cut_start, cut_distance, verbose, ksi)

      implicit none

      ! FCHL descriptors for the training set, format (i,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :, :, :, :, :), intent(in) :: x

      ! List of numbers of atoms in each molecule
      integer, dimension(:), intent(in) :: na

      ! Number of neighbors for each atom in each compound
      integer, dimension(:, :, :, :, :), intent(in) :: nneigh

      ! Decaying powerlaws for two-body term
      double precision, intent(in) :: two_body_power

      ! Fraction of cut_distance at which cut-off starts
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance

      ! Whether to be verbose with output
      logical, intent(in) :: verbose

      ! Pre-computed two-body weights
      !double precision, allocatable, dimension(:, :, :, :, :, :) :: ksi
      double precision, dimension(:, :, :, :, :, :), intent(out) :: ksi

      ! Internal counters
      integer :: maxneigh, maxatoms, nm, a, ni, i, j, pm, xyz, ndisp

      maxneigh = maxval(nneigh)
      maxatoms = maxval(na)
      nm = size(x, dim=1)
      ndisp = size(x, dim=3)

      !allocate (ksi(nm, 3, ndisp, maxatoms, maxatoms, maxneigh))

      ksi = 0.0d0

      !$OMP PARALLEL DO PRIVATE(ni)
      do a = 1, nm
         ni = na(a)
         do xyz = 1, 3
            do pm = 1, ndisp
               do i = 1, ni
                  do j = 1, ni
                     ksi(a, xyz, pm, i, j, :) = get_twobody_weights( &
             & x(a, xyz, pm, i, j, :, :), nneigh(a, xyz, pm, i, j), &
             & two_body_power, cut_start, cut_distance, maxneigh)
                  end do
               end do
            end do
         end do
      end do
      !$OMP END PARALLEL do

      !end function get_ksi_displaced
   end subroutine get_ksi_displaced

   !function get_ksi_atomic(x, na, nneigh, two_body_power, cut_start, cut_distance, verbose) result(ksi)
   subroutine get_ksi_atomic(x, na, nneigh, two_body_power, cut_start, cut_distance, verbose, ksi)

      implicit none

      ! FCHL descriptors for the training set, format (i,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :), intent(in) :: x

      ! List of numbers of atoms
      integer, intent(in) :: na

      ! Number of neighbors for each atom in each compound
      integer, dimension(:), intent(in) :: nneigh

      ! Decaying powerlaws for two-body term
      double precision, intent(in) :: two_body_power

      ! Fraction of cut_distance at which cut-off starts
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance

      ! Whether to be verbose with output
      logical, intent(in) :: verbose

      ! Pre-computed two-body weights
      ! double precision, allocatable, dimension(:, :) :: ksi
      double precision, dimension(:, :), intent(out) :: ksi

      ! Internal counters
      integer :: maxneigh, nm, i

      maxneigh = maxval(nneigh)
      nm = size(x, dim=1)

      ! allocate (ksi(na, maxneigh))

      ksi = 0.0d0

      !$OMP PARALLEL DO
      do i = 1, na
         ksi(i, :) = get_twobody_weights(x(i, :, :), nneigh(i), &
             & two_body_power, cut_start, cut_distance, maxneigh)
      end do
      !$OMP END PARALLEL do

      !end function get_ksi_atomic
   end subroutine get_ksi_atomic

   subroutine init_cosp_sinp(x, na, nneigh, three_body_power, order, cut_start, cut_distance, cosp, sinp, verbose)

      implicit none

      ! FCHL descriptors for the training set, format (i,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :, :), intent(in) :: x

      ! List of numbers of atoms in each molecule
      integer, dimension(:), intent(in) :: na

      ! Number of neighbors for each atom in each compound
      integer, dimension(:, :), intent(in) :: nneigh

      ! Decaying powerlaws for two-body term
      double precision, intent(in) :: three_body_power

      integer, intent(in) :: order

      ! Fraction of cut_distance at which cut-off starts
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance

      ! Whether to be verbose with output
      logical, intent(in) :: verbose

      ! Cosine and sine terms for each atomtype
      double precision, dimension(:, :, :, :, :), intent(out) :: cosp
      double precision, dimension(:, :, :, :, :), intent(out) :: sinp

      ! Internal counters
      integer :: maxneigh, maxatoms, pmax, nm, a, ni, i

      double precision, allocatable, dimension(:, :, :, :) :: fourier

      maxneigh = maxval(nneigh)
      maxatoms = maxval(na)
      nm = size(x, dim=1)

      pmax = get_pmax(x, na)

      cosp = 0.0d0
      sinp = 0.0d0

      !$OMP PARALLEL DO PRIVATE(ni, fourier) schedule(dynamic)
      do a = 1, nm
         ni = na(a)
         do i = 1, ni

            fourier = get_threebody_fourier(x(a, i, :, :), &
                & nneigh(a, i), order, three_body_power, cut_start, cut_distance, pmax, order, maxneigh)

            cosp(a, i, :, :, :) = fourier(1, :, :, :)
            sinp(a, i, :, :, :) = fourier(2, :, :, :)

         end do
      end do
      !$OMP END PARALLEL DO

   end subroutine init_cosp_sinp

   subroutine init_cosp_sinp_displaced(x, na, nneigh, three_body_power, order, cut_start, cut_distance, cosp, sinp, verbose)

      implicit none

      ! FCHL descriptors for the training set, format (i,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :, :, :, :, :), intent(in) :: x

      ! List of numbers of atoms in each molecule
      integer, dimension(:), intent(in) :: na

      ! Number of neighbors for each atom in each compound
      integer, dimension(:, :, :, :, :), intent(in) :: nneigh

      ! Decaying powerlaws for two-body term
      double precision, intent(in) :: three_body_power

      integer, intent(in) :: order

      ! Fraction of cut_distance at which cut-off starts
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance

      ! Whether to be verbose with output
      logical, intent(in) :: verbose

      ! Cosine and sine terms for each atomtype
      double precision, dimension(:, :, :, :, :, :, :) :: sinp
      double precision, dimension(:, :, :, :, :, :, :) :: cosp

      ! Internal counters
      integer :: maxneigh, maxatoms, pmax, nm, a, ni, i, j, xyz, pm, xyz_pm

      double precision, allocatable, dimension(:, :, :, :) :: fourier

      integer :: ndisp

      maxneigh = maxval(nneigh)
      maxatoms = maxval(na)
      nm = size(x, dim=1)

      ndisp = size(x, dim=3)

      pmax = get_pmax_displaced(x, na)

      cosp = 0.0d0
      sinp = 0.0d0

      !$OMP PARALLEL DO PRIVATE(ni, fourier, xyz_pm) schedule(dynamic)
      do a = 1, nm
         ni = na(a)
         do xyz = 1, 3
         do pm = 1, ndisp
         do i = 1, ni
            do j = 1, ni

               xyz_pm = ndisp*(xyz - 1) + pm

               fourier = get_threebody_fourier(x(a, xyz, pm, i, j, :, :), &
                   & nneigh(a, xyz, pm, i, j), &
                   & order, three_body_power, cut_start, cut_distance, &
                   & pmax, order, maxneigh)

               cosp(a, xyz_pm, i, j, :, :, :) = fourier(1, :, :, :)
               sinp(a, xyz_pm, i, j, :, :, :) = fourier(2, :, :, :)

            end do
         end do
         end do
         end do
      end do
      !$OMP END PARALLEL do

   end subroutine init_cosp_sinp_displaced

   subroutine init_cosp_sinp_atomic(x, na, nneigh, three_body_power, order, cut_start, cut_distance, cosp, sinp, verbose)

      implicit none

      ! FCHL descriptors for the training set, format (i,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :), intent(in) :: x

      ! List of numbers of atom
      integer, intent(in) :: na

      ! Number of neighbors for each atom
      integer, dimension(:), intent(in) :: nneigh

      ! Decaying powerlaws for two-body term
      double precision, intent(in) :: three_body_power

      ! Fourier truncation order
      integer, intent(in) :: order

      ! Fraction of cut_distance at which cut-off starts
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance

      ! Whether to be verbose with output
      logical, intent(in) :: verbose

      ! Cosine and sine terms for each atomtype
      double precision, dimension(:, :, :, :), intent(out) :: cosp
      double precision, dimension(:, :, :, :), intent(out) :: sinp

      ! Internal counters
      integer :: maxneigh, pmax, nm, i

      ! Internal temporary variable
      double precision, allocatable, dimension(:, :, :, :) :: fourier

      maxneigh = maxval(nneigh)
      nm = size(x, dim=1)

      pmax = get_pmax_atomic(x, nneigh)

      cosp = 0.0d0
      sinp = 0.0d0

      !$OMP PARALLEL DO PRIVATE(fourier)
      do i = 1, na

         fourier = get_threebody_fourier(x(i, :, :), &
             & nneigh(i), order, three_body_power, cut_start, cut_distance, pmax, order, maxneigh)

         cosp(i, :, :, :) = fourier(1, :, :, :)
         sinp(i, :, :, :) = fourier(2, :, :, :)

      end do
      !$OMP END PARALLEL DO

   end subroutine init_cosp_sinp_atomic

   !function get_selfscalar(x, nm, na, nneigh, ksi, sinp, cosp, t_width, d_width, &
   !        & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose) result(self_scalar)
   subroutine get_selfscalar(x, nm, na, nneigh, ksi, sinp, cosp, t_width, d_width, &
           & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose, self_scalar)

      implicit none

      ! FCHL descriptors for the training set, format (i,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :, :), intent(in) :: x

      ! Number of molecules molecule
      integer, intent(in) :: nm

      ! List of numbers of atoms in each molecule
      integer, dimension(:), intent(in) :: na

      ! Number of neighbors for each atom in each compound
      integer, dimension(:, :), intent(in) :: nneigh

      ! Pre-computed two-body weights
      double precision, dimension(:, :, :), intent(in) :: ksi

      ! Cosine and sine terms for each atomtype
      double precision, dimension(:, :, :, :, :), intent(in) :: sinp
      double precision, dimension(:, :, :, :, :), intent(in) :: cosp

      ! Angular Gaussian width
      double precision, intent(in) :: t_width

      ! Distance Gaussian width
      double precision, intent(in) :: d_width

      double precision, intent(in) :: cut_distance

      ! Truncation order for Fourier terms
      integer, intent(in) :: order

      ! Periodic table distance matrix
      double precision, dimension(:, :), intent(in) :: pd

      ! Angular normalization constant
      double precision, intent(in) :: ang_norm2

      ! Scaling for angular and distance terms
      double precision, intent(in) :: distance_scale
      double precision, intent(in) :: angular_scale

      ! Switch alchemy on or off
      logical, intent(in) :: alchemy

      ! Whether to be verbose with output
      logical, intent(in) :: verbose

      !double precision, allocatable, dimension(:, :) :: self_scalar
      double precision, dimension(:, :) :: self_scalar

      ! Internal counters
      integer :: a, ni, i

      !allocate (self_scalar(nm, maxval(na)))

      !$OMP PARALLEL DO PRIVATE(ni)
      do a = 1, nm
         ni = na(a)
         do i = 1, ni
            self_scalar(a, i) = scalar(x(a, i, :, :), x(a, i, :, :), &
                & nneigh(a, i), nneigh(a, i), ksi(a, i, :), ksi(a, i, :), &
                & sinp(a, i, :, :, :), sinp(a, i, :, :, :), &
                & cosp(a, i, :, :, :), cosp(a, i, :, :, :), &
                & t_width, d_width, cut_distance, order, &
                & pd, ang_norm2, distance_scale, angular_scale, alchemy)
         end do
      end do
      !$OMP END PARALLEL DO

      !end function get_selfscalar
   end subroutine get_selfscalar

   !function get_selfscalar_displaced(x, nm, na, nneigh, ksi, sinp, cosp, t_width, d_width, &
   !        & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose) result(self_scalar)
   subroutine get_selfscalar_displaced(x, nm, na, nneigh, ksi, sinp, cosp, t_width, d_width, &
           & cut_distance, order, pd, ang_norm2, distance_scale, angular_scale, alchemy, verbose, self_scalar)

      implicit none

      ! FCHL descriptors for the training set, format (i,maxatoms,5,maxneighbors)
      double precision, dimension(:, :, :, :, :, :, :), intent(in) :: x

      ! Number of molecules molecule
      integer, intent(in) :: nm

      ! List of numbers of atoms in each molecule
      integer, dimension(:), intent(in) :: na

      ! Number of neighbors for each atom in each compound
      integer, dimension(:, :, :, :, :), intent(in) :: nneigh

      ! Pre-computed two-body weights
      double precision, dimension(:, :, :, :, :, :), intent(in) :: ksi

      ! Cosine and sine terms for each atomtype
      double precision, dimension(:, :, :, :, :, :, :), intent(in) :: sinp
      double precision, dimension(:, :, :, :, :, :, :), intent(in) :: cosp

      ! Angular Gaussian width
      double precision, intent(in) :: t_width

      ! Distance Gaussian width
      double precision, intent(in) :: d_width

      double precision, intent(in) :: cut_distance

      ! Truncation order for Fourier terms
      integer, intent(in) :: order

      ! Periodic table distance matrix
      double precision, dimension(:, :), intent(in) :: pd

      ! Angular normalization constant
      double precision, intent(in) :: ang_norm2

      ! Scaling for angular and distance terms
      double precision, intent(in) :: distance_scale
      double precision, intent(in) :: angular_scale

      ! Switch alchemy on or off
      logical, intent(in) :: alchemy

      ! Whether to be verbose with output
      logical, intent(in) :: verbose

      !double precision, allocatable, dimension(:, :, :, :, :) :: self_scalar
      double precision, dimension(:, :, :, :, :) :: self_scalar

      ! Internal counters
      integer :: a, ni, i, j, pm, xyz, xyz_pm, ndisp

      ndisp = size(x, dim=3)
      ! allocate (self_scalar(nm, 3, ndisp, maxval(na), maxval(na)))
      self_scalar = 0.0d0

      !$OMP PARALLEL DO PRIVATE(ni, xyz_pm) schedule(dynamic)
      do a = 1, nm
         ni = na(a)
         do xyz = 1, 3
         do pm = 1, ndisp
         do i = 1, ni
            do j = 1, ni

               ! z_pm = 2*xyz + pm - 2
               xyz_pm = ndisp*(xyz - 1) + pm

               self_scalar(a, xyz, pm, i, j) = scalar(x(a, xyz, pm, i, j, :, :), x(a, xyz, pm, i, j, :, :), &
                   & nneigh(a, xyz, pm, i, j), nneigh(a, xyz, pm, i, j), &
                   & ksi(a, xyz, pm, i, j, :), ksi(a, xyz, pm, i, j, :), &
                   & sinp(a, xyz_pm, i, j, :, :, :), sinp(a, xyz_pm, i, j, :, :, :), &
                   & cosp(a, xyz_pm, i, j, :, :, :), cosp(a, xyz_pm, i, j, :, :, :), &
                   & t_width, d_width, cut_distance, order, &
                   & pd, ang_norm2, distance_scale, angular_scale, alchemy)
            end do
         end do
         end do
         end do
      end do
      !$OMP END PARALLEL do

      !end function get_selfscalar_displaced
   end subroutine get_selfscalar_displaced

end module ffchl_module
