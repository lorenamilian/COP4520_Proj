## Cloning and Installation Instructions

Install cmake

Install vcpkg

open a terminal window and run the following commands

	vcpkg install armadillo
	vcpkg install openblas
	vcpkg install ensmallen


Install mlpack to C:/mlpack/mlpack (or another folder just be aware that you will have to account for that later)

Download Visual Studio (not VS Code) - should be free for students

Install C++ package (Create new project -> Install more tools and features -> Desktop development with C++ -> Modify

Open visual studio and select clone a repository and enter the URL

Hit clone

## To run the code you will need to switch the the Joey-mlpack branch

On the top toolbar click Git-> Manage Branches

On the left hand side of the window select remotes/origin->Joey-mlpack

This should switch you to the branch that contains the mlpack project

Cmake should automatically install and configure the project for you if it is configured correctly on your machine

You should now be able to hit the green play button to build and run the code



##Troubleshooting

If cmake fails after switching to the Joey-mlpack branch go to Tools->Options->Cmake

Make sure that "Never use CMake Presets" is selected and hit OK

It should try installing/configuring the project again

