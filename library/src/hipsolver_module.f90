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
        enumerator :: HIPSOLVER_SIDE_LEFT  = 141
        enumerator :: HIPSOLVER_SIDE_RIGHT = 142
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_EIG_MODE_NOVECTOR = 201
        enumerator :: HIPSOLVER_EIG_MODE_VECTOR   = 202
    end enum

    enum, bind(c)
        enumerator :: HIPSOLVER_EIG_TYPE_1 = 211
        enumerator :: HIPSOLVER_EIG_TYPE_2 = 212
        enumerator :: HIPSOLVER_EIG_TYPE_3 = 213
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
        enumerator :: HIPSOLVER_STATUS_INVALID_ENUM      = 10
        enumerator :: HIPSOLVER_STATUS_UNKNOWN           = 11
    end enum

end module hipsolver_enums

module hipsolver
    use iso_c_binding

    !---------!
    !   Aux   !
    !---------!
    
    interface
        function hipsolverCreate(handle) &
                bind(c, name = 'hipsolverCreate')
            use iso_c_binding
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCreate_
            type(c_ptr), value :: handle
        end function hipsolverCreate
    end interface

    interface
        function hipsolverDestroy(handle) &
                bind(c, name = 'hipsolverDestroy')
            use iso_c_binding
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDestroy_
            type(c_ptr), value :: handle
        end function hipsolverDestroy
    end interface

    interface
        function hipsolverSetStream(handle, streamId) &
                bind(c, name = 'hipsolverSetStream')
            use iso_c_binding
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSetStream_
            type(c_ptr), value :: handle
            type(c_ptr), value :: streamId
        end function hipsolverSetStream
    end interface

    interface
        function hipsolverGetStream(handle, streamId) &
                bind(c, name = 'hipsolverGetStream')
            use iso_c_binding
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverGetStream_
            type(c_ptr), value :: handle
            type(c_ptr), value :: streamId
        end function hipsolverGetStream
    end interface

    !------------!
    !   LAPACK   !
    !------------!
    
    ! ******************** ORGBR/UNGBR ********************
    interface
        function hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverSorgbr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverSorgbr_bufferSize
    end interface
    
    interface
        function hipsolverDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverDorgbr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverDorgbr_bufferSize
    end interface
    
    interface
        function hipsolverCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverCungbr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverCungbr_bufferSize
    end interface
    
    interface
        function hipsolverZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverZungbr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverZungbr_bufferSize
    end interface
    
    interface
        function hipsolverSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSorgbr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgbr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSorgbr
    end interface
    
    interface
        function hipsolverDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDorgbr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgbr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDorgbr
    end interface
    
    interface
        function hipsolverCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverCungbr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungbr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCungbr
    end interface
    
    interface
        function hipsolverZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZungbr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungbr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZungbr
    end interface
    
    ! ******************** ORGQR/UNGQR ********************
    interface
        function hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverSorgqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverSorgqr_bufferSize
    end interface
    
    interface
        function hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverDorgqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverDorgqr_bufferSize
    end interface
    
    interface
        function hipsolverCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverCungqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverCungqr_bufferSize
    end interface
    
    interface
        function hipsolverZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverZungqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverZungqr_bufferSize
    end interface
    
    interface
        function hipsolverSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSorgqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgqr_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSorgqr
    end interface
    
    interface
        function hipsolverDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDorgqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgqr_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDorgqr
    end interface
    
    interface
        function hipsolverCungqr(handle, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverCungqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungqr_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCungqr
    end interface
    
    interface
        function hipsolverZungqr(handle, m, n, k, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZungqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungqr_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZungqr
    end interface
    
    ! ******************** ORGTR/UNGTR ********************
    interface
        function hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverSorgtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverSorgtr_bufferSize
    end interface
    
    interface
        function hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverDorgtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverDorgtr_bufferSize
    end interface
    
    interface
        function hipsolverCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverCungtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverCungtr_bufferSize
    end interface
    
    interface
        function hipsolverZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
                bind(c, name = 'hipsolverZungtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverZungtr_bufferSize
    end interface
    
    interface
        function hipsolverSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSorgtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSorgtr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSorgtr
    end interface
    
    interface
        function hipsolverDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDorgtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDorgtr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDorgtr
    end interface
    
    interface
        function hipsolverCungtr(handle, uplo, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverCungtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCungtr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCungtr
    end interface
    
    interface
        function hipsolverZungtr(handle, uplo, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZungtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZungtr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZungtr
    end interface
    
    ! ******************** ORMQR/UNMQR ********************
    interface
        function hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverSormqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverSormqr_bufferSize
    end interface
    
    interface
        function hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverDormqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverDormqr_bufferSize
    end interface
    
    interface
        function hipsolverCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverCunmqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverCunmqr_bufferSize
    end interface
    
    interface
        function hipsolverZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverZunmqr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverZunmqr_bufferSize
    end interface
    
    interface
        function hipsolverSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverSormqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormqr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSormqr
    end interface
    
    interface
        function hipsolverDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverDormqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormqr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDormqr
    end interface
    
    interface
        function hipsolverCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverCunmqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmqr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCunmqr
    end interface
    
    interface
        function hipsolverZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverZunmqr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmqr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZunmqr
    end interface
    
    ! ******************** ORMTR/UNMTR ********************
    interface
        function hipsolverSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverSormtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverSormtr_bufferSize
    end interface
    
    interface
        function hipsolverDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverDormtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverDormtr_bufferSize
    end interface
    
    interface
        function hipsolverCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverCunmtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverCunmtr_bufferSize
    end interface
    
    interface
        function hipsolverZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork) &
                bind(c, name = 'hipsolverZunmtr_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: lwork
        end function hipsolverZunmtr_bufferSize
    end interface
    
    interface
        function hipsolverSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverSormtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSormtr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSormtr
    end interface
    
    interface
        function hipsolverDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverDormtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDormtr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDormtr
    end interface
    
    interface
        function hipsolverCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverCunmtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCunmtr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCunmtr
    end interface
    
    interface
        function hipsolverZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info) &
                bind(c, name = 'hipsolverZunmtr')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZunmtr_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZunmtr
    end interface
    
    ! ******************** GEBRD ********************
    interface
        function hipsolverSgebrd_bufferSize(handle, m, n, lwork) &
                bind(c, name = 'hipsolverSgebrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgebrd_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverSgebrd_bufferSize
    end interface
    
    interface
        function hipsolverDgebrd_bufferSize(handle, m, n, lwork) &
                bind(c, name = 'hipsolverDgebrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgebrd_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverDgebrd_bufferSize
    end interface
    
    interface
        function hipsolverCgebrd_bufferSize(handle, m, n, lwork) &
                bind(c, name = 'hipsolverCgebrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgebrd_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverCgebrd_bufferSize
    end interface
    
    interface
        function hipsolverZgebrd_bufferSize(handle, m, n, lwork) &
                bind(c, name = 'hipsolverZgebrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgebrd_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverZgebrd_bufferSize
    end interface

    interface
        function hipsolverSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info) &
                bind(c, name = 'hipsolverSgebrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgebrd_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tauq
            type(c_ptr), value :: taup
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSgebrd
    end interface

    interface
        function hipsolverDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info) &
                bind(c, name = 'hipsolverDgebrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgebrd_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tauq
            type(c_ptr), value :: taup
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDgebrd
    end interface

    interface
        function hipsolverCgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info) &
                bind(c, name = 'hipsolverCgebrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgebrd_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tauq
            type(c_ptr), value :: taup
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCgebrd
    end interface

    interface
        function hipsolverZgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info) &
                bind(c, name = 'hipsolverZgebrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgebrd_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tauq
            type(c_ptr), value :: taup
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZgebrd
    end interface
    
    ! ******************** GELS ********************
    interface
        function hipsolverSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverSSgels_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgels_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverSSgels_bufferSize
    end interface
    
    interface
        function hipsolverDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverDDgels_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgels_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverDDgels_bufferSize
    end interface
    
    interface
        function hipsolverCCgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverCCgels_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgels_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverCCgels_bufferSize
    end interface
    
    interface
        function hipsolverZZgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverZZgels_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgels_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverZZgels_bufferSize
    end interface

    interface
        function hipsolverSSgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverSSgels')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgels_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverSSgels
    end interface

    interface
        function hipsolverDDgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverDDgels')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgels_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverDDgels
    end interface

    interface
        function hipsolverCCgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverCCgels')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgels_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverCCgels
    end interface

    interface
        function hipsolverZZgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverZZgels')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgels_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverZZgels
    end interface
    
    ! ******************** GEQRF ********************
    interface
        function hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSgeqrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSgeqrf_bufferSize
    end interface

    interface
        function hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverDgeqrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDgeqrf_bufferSize
    end interface

    interface
        function hipsolverCgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverCgeqrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCgeqrf_bufferSize
    end interface

    interface
        function hipsolverZgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverZgeqrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZgeqrf_bufferSize
    end interface

    interface
        function hipsolverSgeqrf(handle, m, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSgeqrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgeqrf_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSgeqrf
    end interface

    interface
        function hipsolverDgeqrf(handle, m, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDgeqrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgeqrf_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDgeqrf
    end interface

    interface
        function hipsolverCgeqrf(handle, m, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverCgeqrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgeqrf_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCgeqrf
    end interface

    interface
        function hipsolverZgeqrf(handle, m, n, A, lda, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZgeqrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgeqrf_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZgeqrf
    end interface

    ! ******************** GESV ********************
    interface
        function hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverSSgesv_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverSSgesv_bufferSize
    end interface
    
    interface
        function hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverDDgesv_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverDDgesv_bufferSize
    end interface
    
    interface
        function hipsolverCCgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverCCgesv_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverCCgesv_bufferSize
    end interface
    
    interface
        function hipsolverZZgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork) &
                bind(c, name = 'hipsolverZZgesv_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: lwork
        end function hipsolverZZgesv_bufferSize
    end interface

    interface
        function hipsolverSSgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverSSgesv')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSSgesv_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverSSgesv
    end interface

    interface
        function hipsolverDDgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverDDgesv')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDDgesv_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverDDgesv
    end interface

    interface
        function hipsolverCCgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverCCgesv')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCCgesv_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverCCgesv
    end interface

    interface
        function hipsolverZZgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info) &
                bind(c, name = 'hipsolverZZgesv')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZZgesv_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: X
            integer(c_int), value :: ldx
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: niters
            type(c_ptr), value :: info
        end function hipsolverZZgesv
    end interface
    
    ! ******************** GESVD ********************
    interface
        function hipsolverSgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
                bind(c, name = 'hipsolverSgesvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgesvd_bufferSize_
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverSgesvd_bufferSize
    end interface
    
    interface
        function hipsolverDgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
                bind(c, name = 'hipsolverDgesvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgesvd_bufferSize_
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverDgesvd_bufferSize
    end interface
    
    interface
        function hipsolverCgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
                bind(c, name = 'hipsolverCgesvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgesvd_bufferSize_
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverCgesvd_bufferSize
    end interface
    
    interface
        function hipsolverZgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
                bind(c, name = 'hipsolverZgesvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgesvd_bufferSize_
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: lwork
        end function hipsolverZgesvd_bufferSize
    end interface

    interface
        function hipsolverSgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) &
                bind(c, name = 'hipsolverSgesvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgesvd_
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: rwork
            type(c_ptr), value :: info
        end function hipsolverSgesvd
    end interface

    interface
        function hipsolverDgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) &
                bind(c, name = 'hipsolverDgesvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgesvd_
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: rwork
            type(c_ptr), value :: info
        end function hipsolverDgesvd
    end interface

    interface
        function hipsolverCgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) &
                bind(c, name = 'hipsolverCgesvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgesvd_
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: rwork
            type(c_ptr), value :: info
        end function hipsolverCgesvd
    end interface

    interface
        function hipsolverZgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info) &
                bind(c, name = 'hipsolverZgesvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgesvd_
            type(c_ptr), value :: handle
            integer(c_signed_char), value :: jobu
            integer(c_signed_char), value :: jobv
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: S
            type(c_ptr), value :: U
            integer(c_int), value :: ldu
            type(c_ptr), value :: V
            integer(c_int), value :: ldv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: rwork
            type(c_ptr), value :: info
        end function hipsolverZgesvd
    end interface

    ! ******************** GETRF ********************
    interface
        function hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_bufferSize_
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
                bind(c, name = 'hipsolverDgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_bufferSize_
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
                bind(c, name = 'hipsolverCgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_bufferSize_
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
                bind(c, name = 'hipsolverZgetrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZgetrf_bufferSize
    end interface

    interface
        function hipsolverSgetrf(handle, m, n, A, lda, work, lwork, ipiv, info) &
                bind(c, name = 'hipsolverSgetrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrf_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipsolverSgetrf
    end interface
    
    interface
        function hipsolverDgetrf(handle, m, n, A, lda, work, lwork, ipiv, info) &
                bind(c, name = 'hipsolverDgetrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrf_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipsolverDgetrf
    end interface
    
    interface
        function hipsolverCgetrf(handle, m, n, A, lda, work, lwork, ipiv, info) &
                bind(c, name = 'hipsolverCgetrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrf_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipsolverCgetrf
    end interface
    
    interface
        function hipsolverZgetrf(handle, m, n, A, lda, work, lwork, ipiv, info) &
                bind(c, name = 'hipsolverZgetrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrf_
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: info
        end function hipsolverZgetrf
    end interface

    ! ******************** GETRS ********************
    interface
        function hipsolverSgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork) &
                bind(c, name = 'hipsolverSgetrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverSgetrs_bufferSize
    end interface
    
    interface
        function hipsolverDgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork) &
                bind(c, name = 'hipsolverDgetrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverDgetrs_bufferSize
    end interface
    
    interface
        function hipsolverCgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork) &
                bind(c, name = 'hipsolverCgetrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverCgetrs_bufferSize
    end interface
    
    interface
        function hipsolverZgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork) &
                bind(c, name = 'hipsolverZgetrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverZgetrs_bufferSize
    end interface

    interface
        function hipsolverSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverSgetrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSgetrs_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSgetrs
    end interface
    
    interface
        function hipsolverDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverDgetrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDgetrs_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDgetrs
    end interface
    
    interface
        function hipsolverCgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverCgetrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCgetrs_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCgetrs
    end interface
    
    interface
        function hipsolverZgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverZgetrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZgetrs_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_OP_N)), value :: trans
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZgetrs
    end interface

    ! ******************** POTRF ********************
    interface
        function hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_bufferSize_
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
                bind(c, name = 'hipsolverDpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_bufferSize_
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
                bind(c, name = 'hipsolverCpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_bufferSize_
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
                bind(c, name = 'hipsolverZpotrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_bufferSize_
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
                bind(c, name = 'hipsolverSpotrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrf_
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
                bind(c, name = 'hipsolverDpotrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrf_
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
                bind(c, name = 'hipsolverCpotrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrf_
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
                bind(c, name = 'hipsolverZpotrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrf_
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
    
    ! ******************** POTRF_BATCHED ********************
    interface
        function hipsolverSpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
                bind(c, name = 'hipsolverSpotrfBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverSpotrfBatched_bufferSize
    end interface
    
    interface
        function hipsolverDpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
                bind(c, name = 'hipsolverDpotrfBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverDpotrfBatched_bufferSize
    end interface
    
    interface
        function hipsolverCpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
                bind(c, name = 'hipsolverCpotrfBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverCpotrfBatched_bufferSize
    end interface
    
    interface
        function hipsolverZpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
                bind(c, name = 'hipsolverZpotrfBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverZpotrfBatched_bufferSize
    end interface

    interface
        function hipsolverSpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverSpotrfBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrfBatched_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverSpotrfBatched
    end interface
    
    interface
        function hipsolverDpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverDpotrfBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrfBatched_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverDpotrfBatched
    end interface
    
    interface
        function hipsolverCpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverCpotrfBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrfBatched_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverCpotrfBatched
    end interface
    
    interface
        function hipsolverZpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverZpotrfBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrfBatched_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverZpotrfBatched
    end interface

    ! ******************** POTRI ********************
    interface
        function hipsolverSpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSpotri_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSpotri_bufferSize
    end interface
    
    interface
        function hipsolverDpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverDpotri_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDpotri_bufferSize
    end interface
    
    interface
        function hipsolverCpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverCpotri_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCpotri_bufferSize
    end interface
    
    interface
        function hipsolverZpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
                bind(c, name = 'hipsolverZpotri_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZpotri_bufferSize
    end interface

    interface
        function hipsolverSpotri(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverSpotri')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotri_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSpotri
    end interface

    interface
        function hipsolverDpotri(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverDpotri')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotri_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDpotri
    end interface

    interface
        function hipsolverCpotri(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverCpotri')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotri_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCpotri
    end interface

    interface
        function hipsolverZpotri(handle, uplo, n, A, lda, work, lwork, info) &
                bind(c, name = 'hipsolverZpotri')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotri_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZpotri
    end interface

    ! ******************** POTRS ********************
    interface
        function hipsolverSpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
                bind(c, name = 'hipsolverSpotrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverSpotrs_bufferSize
    end interface
    
    interface
        function hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
                bind(c, name = 'hipsolverDpotrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverDpotrs_bufferSize
    end interface
    
    interface
        function hipsolverCpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
                bind(c, name = 'hipsolverCpotrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverCpotrs_bufferSize
    end interface
    
    interface
        function hipsolverZpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
                bind(c, name = 'hipsolverZpotrs_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
        end function hipsolverZpotrs_bufferSize
    end interface

    interface
        function hipsolverSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverSpotrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrs_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSpotrs
    end interface

    interface
        function hipsolverDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverDpotrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrs_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDpotrs
    end interface

    interface
        function hipsolverCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverCpotrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrs_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCpotrs
    end interface

    interface
        function hipsolverZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info) &
                bind(c, name = 'hipsolverZpotrs')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrs_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZpotrs
    end interface

    ! ******************** POTRS_BATCHED ********************
    interface
        function hipsolverSpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, batch_count) &
                bind(c, name = 'hipsolverSpotrsBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverSpotrsBatched_bufferSize
    end interface
    
    interface
        function hipsolverDpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, batch_count) &
                bind(c, name = 'hipsolverDpotrsBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverDpotrsBatched_bufferSize
    end interface
    
    interface
        function hipsolverCpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, batch_count) &
                bind(c, name = 'hipsolverCpotrsBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverCpotrsBatched_bufferSize
    end interface
    
    interface
        function hipsolverZpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, batch_count) &
                bind(c, name = 'hipsolverZpotrsBatched_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: lwork
            integer(c_int), value :: batch_count
        end function hipsolverZpotrsBatched_bufferSize
    end interface

    interface
        function hipsolverSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverSpotrsBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSpotrsBatched_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverSpotrsBatched
    end interface

    interface
        function hipsolverDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverDpotrsBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDpotrsBatched_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverDpotrsBatched
    end interface

    interface
        function hipsolverCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverCpotrsBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCpotrsBatched_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverCpotrsBatched
    end interface

    interface
        function hipsolverZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, batch_count) &
                bind(c, name = 'hipsolverZpotrsBatched')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZpotrsBatched_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            integer(c_int), value :: nrhs
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
            integer(c_int), value :: batch_count
        end function hipsolverZpotrsBatched
    end interface

    ! ******************** SYEVD/HEEVD ********************
    interface
        function hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork) &
                bind(c, name = 'hipsolverSsyevd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: lwork
        end function hipsolverSsyevd_bufferSize
    end interface
    
    interface
        function hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork) &
                bind(c, name = 'hipsolverDsyevd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: lwork
        end function hipsolverDsyevd_bufferSize
    end interface
    
    interface
        function hipsolverCheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork) &
                bind(c, name = 'hipsolverCheevd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: lwork
        end function hipsolverCheevd_bufferSize
    end interface
    
    interface
        function hipsolverZheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork) &
                bind(c, name = 'hipsolverZheevd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: lwork
        end function hipsolverZheevd_bufferSize
    end interface

    interface
        function hipsolverSsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, info) &
                bind(c, name = 'hipsolverSsyevd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsyevd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSsyevd
    end interface

    interface
        function hipsolverDsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, info) &
                bind(c, name = 'hipsolverDsyevd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsyevd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDsyevd
    end interface

    interface
        function hipsolverCheevd(handle, jobz, uplo, n, A, lda, D, work, lwork, info) &
                bind(c, name = 'hipsolverCheevd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCheevd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCheevd
    end interface

    interface
        function hipsolverZheevd(handle, jobz, uplo, n, A, lda, D, work, lwork, info) &
                bind(c, name = 'hipsolverZheevd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZheevd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZheevd
    end interface

    ! ******************** SYGVD/HEGVD ********************
    interface
        function hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork) &
                bind(c, name = 'hipsolverSsygvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: D
            type(c_ptr), value :: lwork
        end function hipsolverSsygvd_bufferSize
    end interface
    
    interface
        function hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork) &
                bind(c, name = 'hipsolverDsygvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: D
            type(c_ptr), value :: lwork
        end function hipsolverDsygvd_bufferSize
    end interface
    
    interface
        function hipsolverChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork) &
                bind(c, name = 'hipsolverChegvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: D
            type(c_ptr), value :: lwork
        end function hipsolverChegvd_bufferSize
    end interface
    
    interface
        function hipsolverZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, lwork) &
                bind(c, name = 'hipsolverZhegvd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: D
            type(c_ptr), value :: lwork
        end function hipsolverZhegvd_bufferSize
    end interface

    interface
        function hipsolverSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, info) &
                bind(c, name = 'hipsolverSsygvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsygvd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: D
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSsygvd
    end interface

    interface
        function hipsolverDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, info) &
                bind(c, name = 'hipsolverDsygvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsygvd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: D
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDsygvd
    end interface

    interface
        function hipsolverChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, info) &
                bind(c, name = 'hipsolverChegvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChegvd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: D
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverChegvd
    end interface

    interface
        function hipsolverZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, D, work, lwork, info) &
                bind(c, name = 'hipsolverZhegvd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhegvd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
            integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: D
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZhegvd
    end interface

    ! ******************** SYTRD/HETRD ********************
    interface
        function hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
                bind(c, name = 'hipsolverSsytrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverSsytrd_bufferSize
    end interface
    
    interface
        function hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
                bind(c, name = 'hipsolverDsytrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverDsytrd_bufferSize
    end interface
    
    interface
        function hipsolverChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
                bind(c, name = 'hipsolverChetrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverChetrd_bufferSize
    end interface
    
    interface
        function hipsolverZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
                bind(c, name = 'hipsolverZhetrd_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_bufferSize_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: lwork
        end function hipsolverZhetrd_bufferSize
    end interface

    interface
        function hipsolverSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info) &
                bind(c, name = 'hipsolverSsytrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSsytrd
    end interface

    interface
        function hipsolverDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info) &
                bind(c, name = 'hipsolverDsytrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDsytrd
    end interface

    interface
        function hipsolverChetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info) &
                bind(c, name = 'hipsolverChetrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverChetrd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverChetrd
    end interface

    interface
        function hipsolverZhetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info) &
                bind(c, name = 'hipsolverZhetrd')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZhetrd_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: D
            type(c_ptr), value :: E
            type(c_ptr), value :: tau
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZhetrd
    end interface

    ! ******************** SYTRF ********************
    interface
        function hipsolverSsytrf_bufferSize(handle, n, A, lda, lwork) &
                bind(c, name = 'hipsolverSsytrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverSsytrf_bufferSize
    end interface
    
    interface
        function hipsolverDsytrf_bufferSize(handle, n, A, lda, lwork) &
                bind(c, name = 'hipsolverDsytrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverDsytrf_bufferSize
    end interface
    
    interface
        function hipsolverCsytrf_bufferSize(handle, n, A, lda, lwork) &
                bind(c, name = 'hipsolverCsytrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverCsytrf_bufferSize
    end interface
    
    interface
        function hipsolverZsytrf_bufferSize(handle, n, A, lda, lwork) &
                bind(c, name = 'hipsolverZsytrf_bufferSize')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_bufferSize_
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: lwork
        end function hipsolverZsytrf_bufferSize
    end interface

    interface
        function hipsolverSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info) &
                bind(c, name = 'hipsolverSsytrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverSsytrf_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverSsytrf
    end interface

    interface
        function hipsolverDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info) &
                bind(c, name = 'hipsolverDsytrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverDsytrf_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverDsytrf
    end interface

    interface
        function hipsolverCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info) &
                bind(c, name = 'hipsolverCsytrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverCsytrf_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverCsytrf
    end interface

    interface
        function hipsolverZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info) &
                bind(c, name = 'hipsolverZsytrf')
            use iso_c_binding
            use hipsolver_enums
            implicit none
            integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: hipsolverZsytrf_
            type(c_ptr), value :: handle
            integer(kind(HIPSOLVER_FILL_MODE_LOWER)), value :: uplo
            integer(c_int), value :: n
            type(c_ptr), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: ipiv
            type(c_ptr), value :: work
            integer(c_int), value :: lwork
            type(c_ptr), value :: info
        end function hipsolverZsytrf
    end interface
    
end module hipsolver
