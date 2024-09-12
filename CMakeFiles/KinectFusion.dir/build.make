﻿# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib

# Include any dependencies generated for this target.
include CMakeFiles\KinectFusion.dir\depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles\KinectFusion.dir\compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles\KinectFusion.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\KinectFusion.dir\flags.make

CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.obj: CMakeFiles\KinectFusion.dir\flags.make
CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.obj: src\kinectfusion.cpp
CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.obj: CMakeFiles\KinectFusion.dir\compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/KinectFusion.dir/src/kinectfusion.cpp.obj"
	$(CMAKE_COMMAND) -E cmake_cl_compile_depends --dep-file=CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.obj.d --working-dir=C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib --filter-prefix="Note: including file: " -- C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /showIncludes /FoCMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.obj /FdCMakeFiles\KinectFusion.dir\KinectFusion.pdb /FS -c C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\src\kinectfusion.cpp
<<

CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/KinectFusion.dir/src/kinectfusion.cpp.i"
	C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\cl.exe > CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\src\kinectfusion.cpp
<<

CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/KinectFusion.dir/src/kinectfusion.cpp.s"
	C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.s /c C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\src\kinectfusion.cpp
<<

CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.obj: CMakeFiles\KinectFusion.dir\flags.make
CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.obj: src\pose_estimation.cpp
CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.obj: CMakeFiles\KinectFusion.dir\compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/KinectFusion.dir/src/pose_estimation.cpp.obj"
	$(CMAKE_COMMAND) -E cmake_cl_compile_depends --dep-file=CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.obj.d --working-dir=C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib --filter-prefix="Note: including file: " -- C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /showIncludes /FoCMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.obj /FdCMakeFiles\KinectFusion.dir\KinectFusion.pdb /FS -c C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\src\pose_estimation.cpp
<<

CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/KinectFusion.dir/src/pose_estimation.cpp.i"
	C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\cl.exe > CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\src\pose_estimation.cpp
<<

CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/KinectFusion.dir/src/pose_estimation.cpp.s"
	C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.s /c C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\src\pose_estimation.cpp
<<

CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.obj: CMakeFiles\KinectFusion.dir\flags.make
CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.obj: src\surface_measurement.cpp
CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.obj: CMakeFiles\KinectFusion.dir\compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/KinectFusion.dir/src/surface_measurement.cpp.obj"
	$(CMAKE_COMMAND) -E cmake_cl_compile_depends --dep-file=CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.obj.d --working-dir=C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib --filter-prefix="Note: including file: " -- C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /showIncludes /FoCMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.obj /FdCMakeFiles\KinectFusion.dir\KinectFusion.pdb /FS -c C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\src\surface_measurement.cpp
<<

CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/KinectFusion.dir/src/surface_measurement.cpp.i"
	C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\cl.exe > CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\src\surface_measurement.cpp
<<

CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/KinectFusion.dir/src/surface_measurement.cpp.s"
	C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.s /c C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\src\surface_measurement.cpp
<<

# Object files for target KinectFusion
KinectFusion_OBJECTS = \
"CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.obj" \
"CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.obj" \
"CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.obj"

# External object files for target KinectFusion
KinectFusion_EXTERNAL_OBJECTS =

KinectFusion.lib: CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.obj
KinectFusion.lib: CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.obj
KinectFusion.lib: CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.obj
KinectFusion.lib: CMakeFiles\KinectFusion.dir\build.make
KinectFusion.lib: CMakeFiles\KinectFusion.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library KinectFusion.lib"
	$(CMAKE_COMMAND) -P CMakeFiles\KinectFusion.dir\cmake_clean_target.cmake
	C:\PROGRA~2\MIB055~1\2022\BUILDT~1\VC\Tools\MSVC\1441~1.341\bin\Hostx86\x86\lib.exe /nologo /machine:X86 /out:KinectFusion.lib @CMakeFiles\KinectFusion.dir\objects1.rsp

# Rule to build all files generated by this target.
CMakeFiles\KinectFusion.dir\build: KinectFusion.lib
.PHONY : CMakeFiles\KinectFusion.dir\build

CMakeFiles\KinectFusion.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\KinectFusion.dir\cmake_clean.cmake
.PHONY : CMakeFiles\KinectFusion.dir\clean

CMakeFiles\KinectFusion.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\CMakeFiles\KinectFusion.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles\KinectFusion.dir\depend

