cmake_minimum_required(VERSION 2.8)
project( Helipad )

file(GLOB_RECURSE heli_SOURCES "src/src/*.cpp")
file(GLOB_RECURSE heli_HEADERS "src/include/*.h")

set (heli_INCLUDE_DIRS "")
foreach (_headerFile ${heli_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND heli_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES heli_INCLUDE_DIRS)

include_directories(${heli_INCLUDE_DIRS})

find_package( OpenCV REQUIRED )
add_executable (test ${heli_SOURCES})
target_link_libraries( test ${OpenCV_LIBS} )
