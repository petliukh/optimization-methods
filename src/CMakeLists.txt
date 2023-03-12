set(THIS optimization-methods-lib)
set(SOURCES graphical_method.cpp math_utils.cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC OFF)

add_library(${THIS} STATIC ${SOURCES})

target_include_directories(${THIS} PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(${THIS} PUBLIC Eigen3::Eigen muparser)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/eigen
                 ${CMAKE_BINARY_DIR}/external/eigen)

add_subdirectory(${CMAKE_SOURCE_DIR}/external/muparser
                 ${CMAKE_BINARY_DIR}/external/muparser)
