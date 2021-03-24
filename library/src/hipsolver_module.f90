! ************************************************************************
!  Copyright 2020 Advanced Micro Devices, Inc.
! ************************************************************************

module hipsolver_enums
    use iso_c_binding

    !---------------------!
    !   hipSOLVER types   !
    !---------------------!

    enum, bind(c)
        enumerator :: HIPSOLVER_OP_N = 111
        enumerator :: HIPSOLVER_OP_T = 112
        enumerator :: HIPSOLVER_OP_C = 113
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_FILL_MODE_UPPER = 121
        enumerator :: HIPSOLVER_FILL_MODE_LOWER = 122
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_STATUS_SUCCESS           = 0
        enumerator :: HIPSOLVER_STATUS_NOT_INITIALIZED   = 1
        enumerator :: HIPSOLVER_STATUS_ALLOC_FAILED      = 2
        enumerator :: HIPSOLVER_STATUS_INVALID_VALUE     = 3
        enumerator :: HIPSOLVER_STATUS_MAPPING_ERROR     = 4
        enumerator :: HIPSOLVER_STATUS_EXECUTION_FAILED  = 5
        enumerator :: HIPSOLVER_STATUS_INTERNAL_ERROR    = 6
        enumerator :: HIPSOLVER_STATUS_NOT_SUPPORTED     = 7
        enumerator :: HIPSOLVER_STATUS_ARCH_MISMATCH     = 8
        enumerator :: HIPSOLVER_STATUS_HANDLE_IS_NULLPTR = 9
    end enum

end module hipsolver_enums

module hipsolver
    use iso_c_binding

    !---------!
    !   Aux   !
    !---------!
    
    interface
        function hipsolverCreate(handle) &
                result(c_int) &
                bind(c, name = 'hipsolverCreate')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function hipsolverCreate
    end interface

    interface
        function hipsolverDestroy(handle) &
                result(c_int) &
                bind(c, name = 'hipsolverDestroy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function hipsolverDestroy
    end interface

    interface
        function hipsolverSetStream(handle, streamId) &
                result(c_int) &
                bind(c, name = 'hipsolverSetStream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: streamId
        end function hipsolverSetStream
    end interface

    interface
        function hipsolverGetStream(handle, streamId) &
                result(c_int) &
                bind(c, name = 'hipsolverGetStream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: streamId
        end function hipsolverGetStream
    end interface

    !------------!
    !   LAPACK   !
    !------------!

    ! getrf
    interface
        function hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork) &
                result(c_int) &
                bind(c, name = 'hipsolverSgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSgetrf_bufferSize
    end interface
    
    interface
        function hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lwork) &
                result(c_int) &
                bind(c, name = 'hipsolverDgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDgetrf_bufferSize
    end interface
    
    interface
        function hipsolverCgetrf_bufferSize(handle, m, n, A, lda, lwork) &
                result(c_int) &
                bind(c, name = 'hipsolverCgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCgetrf_bufferSize
    end interface
    
    interface
        function hipsolverZgetrf_bufferSize(handle, m, n, A, lda, lwork) &
                result(c_int) &
                bind(c, name = 'hipsolverZgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZgetrf_bufferSize
    end interface

    interface
        function hipsolverSgetrf(handle, m, n, A, lda, work, ipiv, info) &
                result(c_int) &
                bind(c, name = 'hipsolverSgetrf')
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
        end function hipsolverSgetrf
    end interface
    
    interface
        function hipsolverDgetrf(handle, m, n, A, lda, work, ipiv, info) &
                result(c_int) &
                bind(c, name = 'hipsolverDgetrf')
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
        end function hipsolverDgetrf
    end interface
    
    interface
        function hipsolverCgetrf(handle, m, n, A, lda, work, ipiv, info) &
                result(c_int) &
                bind(c, name = 'hipsolverCgetrf')
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
        end function hipsolverCgetrf
    end interface
    
    interface
        function hipsolverZgetrf(handle, m, n, A, lda, work, ipiv, info) &
                result(c_int) &
                bind(c, name = 'hipsolverZgetrf')
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
        end function hipsolverZgetrf
    end interface

    ! potrf
    interface
        function hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
                result(c_int) &
                bind(c, name = 'hipsolverSpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSpotrf_bufferSize
    end interface
    
    interface
        function hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
                result(c_int) &
                bind(c, name = 'hipsolverDpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDpotrf_bufferSize
    end interface
    
    interface
        function hipsolverCpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
                result(c_int) &
                bind(c, name = 'hipsolverCpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCpotrf_bufferSize
    end interface
    
    interface
        function hipsolverZpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
                result(c_int) &
                bind(c, name = 'hipsolverZpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZpotrf_bufferSize
    end interface

    interface
        function hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, info) &
                result(c_int) &
                bind(c, name = 'hipsolverSpotrf')
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
        end function hipsolverSpotrf
    end interface

    interface
        function hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, info) &
                result(c_int) &
                bind(c, name = 'hipsolverDpotrf')
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
        end function hipsolverDpotrf
    end interface

    interface
        function hipsolverCpotrf(handle, uplo, n, A, lda, work, lwork, info) &
                result(c_int) &
                bind(c, name = 'hipsolverCpotrf')
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
        end function hipsolverCpotrf
    end interface

    interface
        function hipsolverZpotrf(handle, uplo, n, A, lda, work, lwork, info) &
                result(c_int) &
                bind(c, name = 'hipsolverZpotrf')
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
        end function hipsolverZpotrf
    end interface
    
end module hipsolver
