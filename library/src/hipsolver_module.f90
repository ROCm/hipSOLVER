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
        enumerator :: HIPSOLVER_FILL_MODE_FULL  = 123
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
    
end module hipsolver
