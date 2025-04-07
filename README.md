## Cloning and Installation Instructions

General Instuctions/Dependencies

You will need mlpack, armadillo, openblas, ftxui, and ensmallen to run this C++ project.

Build and run src/terminalInterface.cpp




## Detailed instructions to run the project in Visual Studio

Watch this youtube video for a guide on installing vcpkg, cmake, and configuring in Visual Studio if needed

https://www.youtube.com/watch?v=FeBzSYiWkEU

Install cmake

Install vcpkg

open a terminal window and run the following commands

	vcpkg install armadillo
	vcpkg install openblas
	vcpkg install ensmallen
	vcpkg install ftxui

Install mlpack to C:/mlpack/mlpack (or another folder just be aware that you will have to account for that later)

Download Visual Studio - should be free for students

Install C++ package (Create new project -> Install more tools and features -> Desktop development with C++ -> Modify

Cmake should automatically install and configure the project for you if it is configured correctly on your machine

Open src/terminalInterface.cpp, select "Current Document" from the dropdown at the top of the window, and hit the green play button to build.

If cmake fails go to Tools->Options->Cmake

Make sure that "Never use CMake Presets" is selected and hit OK

After doing this open the CMakeSettings.json file, which should take you to a settings page

Add the path to vcpkg.cmake in the "CMake Toolchain file: field" -- Mine is "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"

It should try installing/configuring the project again

