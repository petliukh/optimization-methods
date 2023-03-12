set(THIS optimization-methods-tests)
set(SOURCES graphical_method_test.cpp)
set(CMAKE_AUTOMOC OFF)
set(CMAKE_AUTOUIC OFF)
set(CMAKE_AUTORCC OFF)

add_executable(${THIS} ${SOURCES})

add_subdirectory(${CMAKE_SOURCE_DIR}/external/googletest
                 ${CMAKE_BINARY_DIR}/external/googletest)

target_link_libraries(${THIS} optimization-methods-lib gtest gtest_main)

add_test(NAME ${THIS} COMMAND ${THIS})
