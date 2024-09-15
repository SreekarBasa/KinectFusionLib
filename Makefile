﻿# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.30

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake cache editor..."
	echo >nul && "C:\Program Files\CMake\bin\cmake-gui.exe" -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache\fast: edit_cache
.PHONY : edit_cache\fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake to regenerate build system..."
	echo >nul && "C:\Program Files\CMake\bin\cmake.exe" --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache\fast: rebuild_cache
.PHONY : rebuild_cache\fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\CMakeFiles C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\\CMakeFiles\progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 /nologo -$(MAKEFLAGS) all
	$(CMAKE_COMMAND) -E cmake_progress_start C:\Users\91901\Desktop\BTP\kinect\KinectFusionLib\CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 /nologo -$(MAKEFLAGS) clean
.PHONY : clean

# The main clean target
clean\fast: clean
.PHONY : clean\fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 /nologo -$(MAKEFLAGS) preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall\fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 /nologo -$(MAKEFLAGS) preinstall
.PHONY : preinstall\fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles\Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named KinectFusion

# Build rule for target.
KinectFusion: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles\Makefile2 /nologo -$(MAKEFLAGS) KinectFusion
.PHONY : KinectFusion

# fast build rule for target.
KinectFusion\fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\build
.PHONY : KinectFusion\fast

src\kinectfusion.obj: src\kinectfusion.cpp.obj
.PHONY : src\kinectfusion.obj

# target to build an object file
src\kinectfusion.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.obj
.PHONY : src\kinectfusion.cpp.obj

src\kinectfusion.i: src\kinectfusion.cpp.i
.PHONY : src\kinectfusion.i

# target to preprocess a source file
src\kinectfusion.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.i
.PHONY : src\kinectfusion.cpp.i

src\kinectfusion.s: src\kinectfusion.cpp.s
.PHONY : src\kinectfusion.s

# target to generate assembly for a file
src\kinectfusion.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\kinectfusion.cpp.s
.PHONY : src\kinectfusion.cpp.s

src\pose_estimation.obj: src\pose_estimation.cpp.obj
.PHONY : src\pose_estimation.obj

# target to build an object file
src\pose_estimation.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.obj
.PHONY : src\pose_estimation.cpp.obj

src\pose_estimation.i: src\pose_estimation.cpp.i
.PHONY : src\pose_estimation.i

# target to preprocess a source file
src\pose_estimation.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.i
.PHONY : src\pose_estimation.cpp.i

src\pose_estimation.s: src\pose_estimation.cpp.s
.PHONY : src\pose_estimation.s

# target to generate assembly for a file
src\pose_estimation.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\pose_estimation.cpp.s
.PHONY : src\pose_estimation.cpp.s

src\surface_measurement.obj: src\surface_measurement.cpp.obj
.PHONY : src\surface_measurement.obj

# target to build an object file
src\surface_measurement.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.obj
.PHONY : src\surface_measurement.cpp.obj

src\surface_measurement.i: src\surface_measurement.cpp.i
.PHONY : src\surface_measurement.i

# target to preprocess a source file
src\surface_measurement.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.i
.PHONY : src\surface_measurement.cpp.i

src\surface_measurement.s: src\surface_measurement.cpp.s
.PHONY : src\surface_measurement.s

# target to generate assembly for a file
src\surface_measurement.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\surface_measurement.cpp.s
.PHONY : src\surface_measurement.cpp.s

src\test.obj: src\test.cpp.obj
.PHONY : src\test.obj

# target to build an object file
src\test.cpp.obj:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\test.cpp.obj
.PHONY : src\test.cpp.obj

src\test.i: src\test.cpp.i
.PHONY : src\test.i

# target to preprocess a source file
src\test.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\test.cpp.i
.PHONY : src\test.cpp.i

src\test.s: src\test.cpp.s
.PHONY : src\test.s

# target to generate assembly for a file
src\test.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles\KinectFusion.dir\build.make /nologo -$(MAKEFLAGS) CMakeFiles\KinectFusion.dir\src\test.cpp.s
.PHONY : src\test.cpp.s

# Help Target
help:
	@echo The following are some of the valid targets for this Makefile:
	@echo ... all (the default if no target is provided)
	@echo ... clean
	@echo ... depend
	@echo ... edit_cache
	@echo ... rebuild_cache
	@echo ... KinectFusion
	@echo ... src/kinectfusion.obj
	@echo ... src/kinectfusion.i
	@echo ... src/kinectfusion.s
	@echo ... src/pose_estimation.obj
	@echo ... src/pose_estimation.i
	@echo ... src/pose_estimation.s
	@echo ... src/surface_measurement.obj
	@echo ... src/surface_measurement.i
	@echo ... src/surface_measurement.s
	@echo ... src/test.obj
	@echo ... src/test.i
	@echo ... src/test.s
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles\Makefile.cmake 0
.PHONY : cmake_check_build_system

