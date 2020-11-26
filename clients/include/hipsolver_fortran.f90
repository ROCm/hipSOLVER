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

    ! getrf
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
    
end module hipsolver_interface
