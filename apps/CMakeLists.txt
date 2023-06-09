find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(Qwt REQUIRED)

set(APP_SOURCES main.cpp main_window.cpp main_window.h main_window.ui)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
  qt_add_executable(optimization-methods MANUAL_FINALIZATION ${APP_SOURCES})
  # Define target properties for Android with Qt 6 as: set_property(TARGET
  # optimization-methods APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
  # ${CMAKE_CURRENT_SOURCE_DIR}/android) For more information, see
  # https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
  if(ANDROID)
    add_library(optimization-methods SHARED ${APP_SOURCES})
    # Define properties for Android with Qt 5 after find_package() calls as:
    # set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
  else()
    add_executable(optimization-methods ${APP_SOURCES})
  endif()
endif()

target_link_libraries(
  optimization-methods PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${QWT_LIBRARIES}
                               optimization-methods-lib)
include_directories(${QWT_INCLUDE_DIRS})

set_target_properties(
  optimization-methods
  PROPERTIES MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
             MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
             MACOSX_BUNDLE_SHORT_VERSION_STRING
             ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
             MACOSX_BUNDLE TRUE
             WIN32_EXECUTABLE TRUE)

install(
  TARGETS optimization-methods
  BUNDLE DESTINATION .
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
  qt_finalize_executable(optimization-methods)
endif()
