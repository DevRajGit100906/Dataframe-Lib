cmake_minimum_required(VERSION 3.20)
project(DataFrameLib LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_path(GRAPHVIZ_INCLUDE_DIR graphviz/cgraph.h)
find_library(GRAPHVIZ_CGRAPH_LIB cgraph)
find_library(GRAPHVIZ_GVC_LIB gvc)
add_library(dataframelib
    src/expr.cpp
    src/eager_dataframe.cpp
    src/kernel.cpp
    src/lazy_dataframe.cpp
    src/logical_plan.cpp
    src/query_optimizer.cpp
)
target_include_directories(dataframelib PUBLIC include)
target_include_directories(dataframelib PRIVATE ${GRAPHVIZ_INCLUDE_DIR})
target_link_libraries(dataframelib
    PUBLIC Arrow::arrow_shared Parquet::parquet_shared
    PRIVATE ${GRAPHVIZ_CGRAPH_LIB} ${GRAPHVIZ_GVC_LIB}
)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
target_compile_options(dataframelib PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -mtune=native -funroll-loops -ftree-vectorize>
)
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_OK OUTPUT IPO_ERR)
if(IPO_OK)
    set_target_properties(dataframelib PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
