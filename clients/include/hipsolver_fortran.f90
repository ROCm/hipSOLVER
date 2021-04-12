! ************************************************************************
!  Copyright 2020 Advanced Micro Devices, Inc.
! ************************************************************************

module hipsolver_interface
    use iso_c_binding
    use hipsolver

    contains

    !------------!
    !   LAPACK   !
    !------------!

    ! ******************** GETRF ********************
    function hipsolverSgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork) &
            result(res) &
            bind(c, name = 'hipsolverSgetrf_bufferSizeFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: lwork
        integer(c_int) :: res
        res = hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork)
    end function hipsolverSgetrf_bufferSizeFortran
    
    function hipsolverDgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork) &
            result(res) &
            bind(c, name = 'hipsolverDgetrf_bufferSizeFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: lwork
        integer(c_int) :: res
        res = hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lwork)
    end function hipsolverDgetrf_bufferSizeFortran
    
    function hipsolverCgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork) &
            result(res) &
            bind(c, name = 'hipsolverCgetrf_bufferSizeFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: lwork
        integer(c_int) :: res
        res = hipsolverCgetrf_bufferSize(handle, m, n, A, lda, lwork)
    end function hipsolverCgetrf_bufferSizeFortran
    
    function hipsolverZgetrf_bufferSizeFortran(handle, m, n, A, lda, lwork) &
            result(res) &
            bind(c, name = 'hipsolverZgetrf_bufferSizeFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: lwork
        integer(c_int) :: res
        res = hipsolverZgetrf_bufferSize(handle, m, n, A, lda, lwork)
    end function hipsolverZgetrf_bufferSizeFortran

    function hipsolverSgetrfFortran(handle, m, n, A, lda, work, ipiv, info) &
            result(res) &
            bind(c, name = 'hipsolverSgetrfFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: work
        type(c_ptr), value :: ipiv
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipsolverSgetrf(handle, m, n, A, lda, work, ipiv, info)
    end function hipsolverSgetrfFortran
    
    function hipsolverDgetrfFortran(handle, m, n, A, lda, work, ipiv, info) &
            result(res) &
            bind(c, name = 'hipsolverDgetrfFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: work
        type(c_ptr), value :: ipiv
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipsolverDgetrf(handle, m, n, A, lda, work, ipiv, info)
    end function hipsolverDgetrfFortran
    
    function hipsolverCgetrfFortran(handle, m, n, A, lda, work, ipiv, info) &
            result(res) &
            bind(c, name = 'hipsolverCgetrfFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: work
        type(c_ptr), value :: ipiv
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipsolverCgetrf(handle, m, n, A, lda, work, ipiv, info)
    end function hipsolverCgetrfFortran

    function hipsolverZgetrfFortran(handle, m, n, A, lda, work, ipiv, info) &
            result(res) &
            bind(c, name = 'hipsolverZgetrfFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(c_int), value :: m
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: work
        type(c_ptr), value :: ipiv
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipsolverZgetrf(handle, m, n, A, lda, work, ipiv, info)
    end function hipsolverZgetrfFortran

    ! ******************** POTRF ********************
    function hipsolverSpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork) &
            result(res) &
            bind(c, name = 'hipsolverSpotrf_bufferSizeFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: lwork
        integer(c_int) :: res
        res = hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork)
    end function hipsolverSpotrf_bufferSizeFortran
    
    function hipsolverDpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork) &
            result(res) &
            bind(c, name = 'hipsolverDpotrf_bufferSizeFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: lwork
        integer(c_int) :: res
        res = hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, lwork)
    end function hipsolverDpotrf_bufferSizeFortran
    
    function hipsolverCpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork) &
            result(res) &
            bind(c, name = 'hipsolverCpotrf_bufferSizeFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: lwork
        integer(c_int) :: res
        res = hipsolverCpotrf_bufferSize(handle, uplo, n, A, lda, lwork)
    end function hipsolverCpotrf_bufferSizeFortran
    
    function hipsolverZpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork) &
            result(res) &
            bind(c, name = 'hipsolverZpotrf_bufferSizeFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: lwork
        integer(c_int) :: res
        res = hipsolverZpotrf_bufferSize(handle, uplo, n, A, lda, lwork)
    end function hipsolverZpotrf_bufferSizeFortran

    function hipsolverSpotrfFortran(handle, uplo, n, A, lda, work, lwork, info) &
            result(res) &
            bind(c, name = 'hipsolverSpotrfFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: work
        integer(c_int), value :: lwork
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, info)
    end function hipsolverSpotrfFortran

    function hipsolverDpotrfFortran(handle, uplo, n, A, lda, work, lwork, info) &
            result(res) &
            bind(c, name = 'hipsolverDpotrfFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: work
        integer(c_int), value :: lwork
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, info)
    end function hipsolverDpotrfFortran

    function hipsolverCpotrfFortran(handle, uplo, n, A, lda, work, lwork, info) &
            result(res) &
            bind(c, name = 'hipsolverCpotrfFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: work
        integer(c_int), value :: lwork
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipsolverCpotrf(handle, uplo, n, A, lda, work, lwork, info)
    end function hipsolverCpotrfFortran

    function hipsolverZpotrfFortran(handle, uplo, n, A, lda, work, lwork, info) &
            result(res) &
            bind(c, name = 'hipsolverZpotrfFortran')
        use iso_c_binding
        use hipsolver_enums
        implicit none
        type(c_ptr), value :: handle
        integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
        integer(c_int), value :: n
        type(c_ptr), value :: A
        integer(c_int), value :: lda
        type(c_ptr), value :: work
        integer(c_int), value :: lwork
        type(c_ptr), value :: info
        integer(c_int) :: res
        res = hipsolverZpotrf(handle, uplo, n, A, lda, work, lwork, info)
    end function hipsolverZpotrfFortran
    
end module hipsolver_interface
