module ffchl_module_ef

   implicit none

contains

   !function get_ksi_ef(x, na, nneigh, two_body_power, cut_start, cut_distance, ef_scale, df, verbose) result(ksi)
   subroutine get_ksi_ef(x, na, nneigh, two_body_power, cut_start, cut_distance, ef_scale, df, verbose, ksi)

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

      ! Electric field displacement
      double precision, intent(in) :: ef_scale
      double precision, intent(in) :: df

      ! Display output
      logical, intent(in) :: verbose

      ! Pre-computed two-body weights
      ! double precision, allocatable, dimension(:, :, :, :, :) :: ksi
      double precision, dimension(:, :, :, :, :) :: ksi

      ! Internal counters
      integer :: maxneigh, maxatoms, nm, a, ni, i, xyz, pm

      ! Electric field
      double precision, dimension(3) :: field

      maxneigh = maxval(nneigh)
      maxatoms = maxval(na)
      nm = size(x, dim=1)

      !allocate (ksi(nm, 3, 2, maxatoms, maxneigh))

      ksi = 0.0d0

      !$OMP PARALLEL DO PRIVATE(ni, field)
      do a = 1, nm
         ni = na(a)
         do xyz = 1, 3
            do pm = 1, 2

               field = 0.0d0
               field(xyz) = (pm - 1.5d0)*2.0d0*df

               ! write(*,*) xyz, pm, field

               do i = 1, ni

                  ksi(a, xyz, pm, i, :) = get_twobody_weights_ef(x(a, i, :, :), field, nneigh(a, i), &
                      & two_body_power, cut_start, cut_distance, maxneigh, ef_scale)

               end do
            end do
         end do
      end do
      !$OMP END PARALLEL do

      ! end function get_ksi_ef
   end subroutine get_ksi_ef

   !function get_ksi_ef_field(x, na, nneigh, two_body_power, cut_start, cut_distance, fields, ef_scale, verbose) result(ksi)
   subroutine get_ksi_ef_field(x, na, nneigh, two_body_power, cut_start, cut_distance, fields, ef_scale, verbose, ksi)

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

      ! Electric fields for each representation
      double precision, dimension(:, :), intent(in) :: fields

      ! Display output
      logical, intent(in) :: verbose

      ! Pre-computed two-body weights
      !double precision, allocatable, dimension(:, :, :) :: ksi
      double precision, dimension(:, :, :), intent(out) :: ksi

      ! Internal counters
      integer :: maxneigh, maxatoms, nm, a, ni, i

      double precision, intent(in) :: ef_scale

      ! Electric field displacement
      ! double precision, intent(in) :: df

      maxneigh = maxval(nneigh)
      maxatoms = maxval(na)
      nm = size(x, dim=1)

      !allocate (ksi(nm, maxatoms, maxneigh))

      ksi = 0.0d0

      !$OMP PARALLEL DO PRIVATE(ni)
      do a = 1, nm
         ni = na(a)

         do i = 1, ni

            ksi(a, i, :) = get_twobody_weights_ef(x(a, i, :, :), fields(a, :), nneigh(a, i), &
                & two_body_power, cut_start, cut_distance, maxneigh, ef_scale)
         end do

      end do
      !$OMP END PARALLEL do

      !end function get_ksi_ef_field
   end subroutine get_ksi_ef_field

   subroutine init_cosp_sinp_ef(x, na, nneigh, three_body_power, order, cut_start, cut_distance, &
          & cosp, sinp, ef_scale, df, verbose)

      use ffchl_module, only: get_pmax

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
      double precision, intent(in) :: ef_scale

      ! Electric field displacement
      double precision, intent(in) :: df

      ! Cosine and sine terms for each atomtype
      double precision, dimension(:, :, :, :, :, :, :), intent(out) :: cosp
      double precision, dimension(:, :, :, :, :, :, :), intent(out) :: sinp

      ! Display output
      logical, intent(in) :: verbose

      ! Internal counters
      integer :: maxneigh, maxatoms, pmax, nm, a, ni, i

      double precision, allocatable, dimension(:, :, :, :) :: fourier

      ! Internal counters
      integer :: xyz, pm

      ! Electric field
      double precision, dimension(3) :: field

      maxneigh = maxval(nneigh)
      maxatoms = maxval(na)
      nm = size(x, dim=1)

      pmax = get_pmax(x, na)

      cosp = 0.0d0
      sinp = 0.0d0

      !$OMP PARALLEL DO PRIVATE(ni, fourier, field) schedule(dynamic)
      do a = 1, nm
         ni = na(a)
         do xyz = 1, 3
            do pm = 1, 2

               field = 0.0d0
               field(xyz) = (pm - 1.5d0)*2.0d0*df

               do i = 1, ni

                  fourier = get_threebody_fourier_ef(x(a, i, :, :), field, &
                  & nneigh(a, i), order, three_body_power, cut_start, cut_distance, pmax, order, maxneigh, ef_scale)

                  cosp(a, xyz, pm, i, :, :, :) = fourier(1, :, :, :)
                  sinp(a, xyz, pm, i, :, :, :) = fourier(2, :, :, :)

               end do
            end do
         end do
      end do
      !$OMP END PARALLEL DO

   end subroutine init_cosp_sinp_ef

   subroutine init_cosp_sinp_ef_field(x, na, nneigh, three_body_power, &
           & order, cut_start, cut_distance, cosp, sinp, fields, ef_scale, verbose)

      use ffchl_module, only: get_pmax

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
      double precision, intent(in) :: ef_scale

      ! Electric field displacement
      ! double precision, intent(in) :: df

      ! Display output
      logical, intent(in) :: verbose

      ! Cosine and sine terms for each atomtype
      double precision, dimension(:, :, :, :, :), intent(out) :: cosp
      double precision, dimension(:, :, :, :, :), intent(out) :: sinp

      ! Electric fields for each representation
      double precision, dimension(:, :), intent(in) :: fields

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

            fourier = get_threebody_fourier_ef(x(a, i, :, :), fields(a, :), &
            & nneigh(a, i), order, three_body_power, cut_start, cut_distance, pmax, order, maxneigh, ef_scale)

            cosp(a, i, :, :, :) = fourier(1, :, :, :)
            sinp(a, i, :, :, :) = fourier(2, :, :, :)
         end do

      end do
      !$OMP END PARALLEL DO

   end subroutine init_cosp_sinp_ef_field

   ! Calculate the Fourier terms for the FCHL three-body expansion
   function get_threebody_fourier_ef(x, field, neighbors, order, power, cut_start, cut_distance, &
       & dim1, dim2, dim3, ef_scale) result(fourier)

      use ffchl_module, only: calc_angle

      implicit none

      ! Input representation, dimension=(5,n).
      double precision, dimension(:, :), intent(in) :: x

      double precision, dimension(3), intent(in) :: field

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

      double precision, intent(in) :: ef_scale

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

            ksi3 = calc_ksi3_ef(X(:, :), field, j, k, neighbors, power, cut_start, cut_distance, ef_scale)
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

   end function get_threebody_fourier_ef

   function calc_ksi3_ef(X, field, j, k, neighbors, power, cut_start, cut_distance, ef_scale) result(ksi3)

      use ffchl_module, only: cut_function, calc_cos_angle

      implicit none

      double precision, dimension(:, :), intent(in) :: X

      double precision, dimension(3), intent(in) :: field

      integer, intent(in) :: j
      integer, intent(in) :: k

      ! Number of neighboring atoms to iterate over.
      integer, intent(in) :: neighbors

      double precision, intent(in) :: power
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance
      double precision, intent(in) :: ef_scale

      double precision :: cos_i, cos_j, cos_k
      double precision :: di, dj, dk

      double precision :: ksi3
      double precision :: cut

      ! Center of nuclear charge
      double precision, dimension(3) :: coz
      double precision, dimension(3) :: dipole

      double precision :: total_charge

      integer :: i
      ! coz
      ! dipole

      cos_i = calc_cos_angle(x(3:5, k), x(3:5, 1), x(3:5, j))
      cos_j = calc_cos_angle(x(3:5, j), x(3:5, k), x(3:5, 1))
      cos_k = calc_cos_angle(x(3:5, 1), x(3:5, j), x(3:5, k))

      dk = x(1, j)
      dj = x(1, k)
      di = norm2(x(3:5, j) - x(3:5, k))

      cut = cut_function(dk, cut_start, cut_distance)* &
          & cut_function(dj, cut_start, cut_distance)* &
          & cut_function(di, cut_start, cut_distance)

      total_charge = 0.0d0

      coz = 0.0d0

      do i = 1, neighbors
         coz = coz + x(3:5, i)*x(2, i)
         total_charge = total_charge + x(2, i)
      end do

      coz = coz/total_charge

      dipole = (x(3:5, 1) - coz)*x(6, 1) + (x(3:5, j) - coz)*x(6, j) &
           & + (x(3:5, k) - coz)*x(6, k)

      ! ksi3 = cut * (1.0d0 + 3.0d0 * cos_i*cos_j*cos_k) / (di * dj * dk)**power

      ksi3 = cut*((1.0d0 + 3.0d0*cos_i*cos_j*cos_k)/(di*dj*dk)**power &
         & + ef_scale*dot_product(dipole, field))

      ! ksi3 = cut * dot_product(dipole, field)

   end function calc_ksi3_ef

   function get_twobody_weights_ef(x, field, neighbors, power, cut_start, cut_distance, dim1, ef_scale) result(ksi)

      use ffchl_module, only: cut_function

      implicit none

      double precision, dimension(:, :), intent(in) :: x
      double precision, dimension(3), intent(in) :: field
      integer, intent(in) :: neighbors
      double precision, intent(in) :: power
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance
      double precision, intent(in) :: ef_scale
      integer, intent(in) :: dim1

      double precision, dimension(dim1) :: ksi

      ! Electric field displacement
      ! double precision, intent(in) :: df

      integer :: i

      double precision, dimension(3) :: dipole

      ! Center of nuclear charge
      double precision, dimension(3) :: coz
      double precision :: total_charge

      ksi = 0.0d0

      coz = 0.0d0
      total_charge = 0.0d0
      do i = 1, neighbors
         coz = coz + x(3:5, i)*x(2, i)
         total_charge = total_charge + x(2, i)
      end do

      coz = coz/total_charge

      do i = 2, neighbors

         dipole = (x(3:5, 1) - coz)*x(6, 1) + (x(3:5, i) - coz)*x(6, i)

         ksi(i) = cut_function(x(1, i), cut_start, cut_distance) &
                 & *(1.0d0/x(1, i)**power + ef_scale*dot_product(dipole, field))

         ! ksi(i) = cut_function(x(1, i), cut_start, cut_distance) &
         !     & * (dot_product(dipole, field))

      end do

   end function get_twobody_weights_ef

   function get_twobody_weights_ef_field(x, field, neighbors, power, cut_start, cut_distance, dim1, ef_scale) result(ksi)

      use ffchl_module, only: cut_function

      implicit none

      double precision, dimension(:, :), intent(in) :: x
      double precision, dimension(3), intent(in) :: field
      integer, intent(in) :: neighbors
      double precision, intent(in) :: power
      double precision, intent(in) :: cut_start
      double precision, intent(in) :: cut_distance
      double precision, intent(in) :: ef_scale
      integer, intent(in) :: dim1

      double precision, dimension(dim1) :: ksi

      ! Electric field displacement
      ! double precision, intent(in) :: df

      integer :: i

      double precision, dimension(3) :: dipole

      ! Center of nuclear charge
      double precision, dimension(3) :: coz
      double precision :: total_charge

      ksi = 0.0d0

      coz = 0.0d0
      total_charge = 0.0d0
      do i = 1, neighbors
         coz = coz + x(3:5, i)*x(2, i)
         total_charge = total_charge + x(2, i)
      end do

      coz = coz/total_charge

      do i = 2, neighbors

         dipole = (x(3:5, 1) - coz)*x(6, 1) + (x(3:5, i) - coz)*x(6, i)

         ksi(i) = cut_function(x(1, i), cut_start, cut_distance) &
                 & *(1.0d0/x(1, i)**power + ef_scale*dot_product(dipole, field))

      end do

   end function get_twobody_weights_ef_field

end module ffchl_module_ef
