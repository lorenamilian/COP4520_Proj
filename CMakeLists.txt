cmake_minimum_required(VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

# Set project name
project("RocketLaunchPrediction")
cmake_policy(SET CMP0079 NEW)

# Disable deprecation warnings as errors for macOS
if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
endif()

# Configure FTXUI to ignore deprecation warnings
# Find required packages
find_package(Armadillo REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(cpr REQUIRED)
find_package(ftxui 5 REQUIRED)

# Add the executable target (main.cpp should be your source file)
add_executable(RocketLaunchPrediction src/main.cpp)

add_executable(launchpredictor src/terminalInterface.cpp src/RocketLaunchPrediction.cpp)

# Set C++17 standard explicitly since FTXUI requires it
target_compile_features(RocketLaunchPrediction PRIVATE cxx_std_20)
target_compile_features(launchpredictor PRIVATE cxx_std_20)
target_link_libraries(launchpredictor PRIVATE
    armadillo
    nlohmann_json::nlohmann_json
    cpr::cpr
    ftxui::screen
    ftxui::dom
    ftxui::component
)

# Link libraries
target_link_libraries(RocketLaunchPrediction PRIVATE
    armadillo
    nlohmann_json::nlohmann_json
    cpr::cpr
    ftxui::screen
    ftxui::dom
    ftxui::component
)

# Enable LSP support (optional, for clangd, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
