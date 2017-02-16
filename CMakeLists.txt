cmake_minimum_required(VERSION 2.8) 
PROJECT (CognexProject)
FIND_PACKAGE(OpenCV  REQUIRED )
set (COGNEXPROJECT_REQUIRED_LIBRARIES ${OpenCV_LIBS})
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

set( NAME_SRC
    ./src/main.cpp
	./src/TestEnv.cpp
	./src/Fir.cpp
)

set( NAME_HEADERS       
	./include/TestEnv.h
	./include/Fir.h
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
link_directories(${OpenCV_LIB_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( CognexProject ${NAME_SRC} ${NAME_HEADERS} )
target_link_libraries( CognexProject ${OpenCV_LIBS} )