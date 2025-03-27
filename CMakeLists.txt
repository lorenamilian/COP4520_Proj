# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("RocketLaunchPrediction")
cmake_policy(SET CMP0079 NEW)
# Include sub-projects.


add_subdirectory ("RocketLaunchPrediction")
 find_package(Armadillo REQUIRED)
 find_package(OpenBLAS CONFIG REQUIRED)
 find_package(nlohmann_json 3.2.0 REQUIRED)
 find_package(cpr REQUIRED)
target_link_libraries(RocketLaunchPrediction PRIVATE 
    armadillo
    nlohmann_json::nlohmann_json
    cpr::cpr
)