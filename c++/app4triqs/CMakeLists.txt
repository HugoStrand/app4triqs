set(LIBRARY_SOURCES
    arithmetic.cpp
    )

add_library(arithmetic_c ${LIBRARY_SOURCES})
target_link_libraries(arithmetic_c PUBLIC triqs)

# Pubic options : any option affecting the headers should go here
target_compile_options(arithmetic_c PUBLIC)

# Private options : any option here shoud affect ONLY the cpp
target_compile_options(arithmetic_c PRIVATE 
                       -DCTHYB_GIT_HASH=${CTHYB_GIT_HASH} -DTRIQS_GIT_HASH=${TRIQS_GIT_HASH}
                       ) 

# Install
install(TARGETS arithmetic_c DESTINATION lib)

# FIXME : WHY DO WE INSTALL HEADERS HERE ??
set(HEADERS_TO_INSTALL
    arithmetic.hpp
)
install(FILES ${HEADERS_TO_INSTALL} DESTINATION include/triqs/app4triqs)
