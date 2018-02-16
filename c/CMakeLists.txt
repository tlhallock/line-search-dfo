cmake_minimum_required(VERSION 3.8)
project(c)


set(CMAKE_CXX_STANDARD 11)
set(COMMON_FILES
    base/PolynomialBasis.cpp
    utils/math_utils.cpp
        dfo/Lagrange.cpp dfo/Lagrange.h opt/Optimization.cpp opt/Optimization.h opt/NewMat2Armadillo.cpp opt/NewMat2Armadillo.h fcn/Utils.cpp fcn/Utils.h opt/NLOpt.cpp opt/NLOpt.h sys/Plotter.cpp sys/Plotter.h base/Bounds.cpp base/Bounds.h sys/Config.cpp sys/Config.h base/TrustRegion.cpp base/TrustRegion.h base/Shifter.cpp base/Shifter.h alg/algorithm.cpp alg/algorithm.h alg/Ellipse.cpp alg/Ellipse.h)


include_directories(SYSTEM /work/research/line-search-dfo/c/libs/prefix/include)
link_directories(/work/research/line-search-dfo/c/libs/prefix/lib)

include_directories(SYSTEM /work/research/line-search-dfo/c/libs/prefix/usr/include)
link_directories(/work/research/line-search-dfo/c/libs/prefix/usr/lib/x86_64-linux-gnu)

add_executable(c main.cpp ${COMMON_FILES})


target_link_libraries(c newmat.a opt.a nlopt armadillo newmat blas)