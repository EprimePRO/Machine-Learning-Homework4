# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2017.3.4\bin\cmake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2017.3.4\bin\cmake\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic\cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/Logistic.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Logistic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Logistic.dir/flags.make

CMakeFiles/Logistic.dir/main.cpp.obj: CMakeFiles/Logistic.dir/flags.make
CMakeFiles/Logistic.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Logistic.dir/main.cpp.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\Logistic.dir\main.cpp.obj -c "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic\main.cpp"

CMakeFiles/Logistic.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Logistic.dir/main.cpp.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic\main.cpp" > CMakeFiles\Logistic.dir\main.cpp.i

CMakeFiles/Logistic.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Logistic.dir/main.cpp.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic\main.cpp" -o CMakeFiles\Logistic.dir\main.cpp.s

CMakeFiles/Logistic.dir/main.cpp.obj.requires:

.PHONY : CMakeFiles/Logistic.dir/main.cpp.obj.requires

CMakeFiles/Logistic.dir/main.cpp.obj.provides: CMakeFiles/Logistic.dir/main.cpp.obj.requires
	$(MAKE) -f CMakeFiles\Logistic.dir\build.make CMakeFiles/Logistic.dir/main.cpp.obj.provides.build
.PHONY : CMakeFiles/Logistic.dir/main.cpp.obj.provides

CMakeFiles/Logistic.dir/main.cpp.obj.provides.build: CMakeFiles/Logistic.dir/main.cpp.obj


# Object files for target Logistic
Logistic_OBJECTS = \
"CMakeFiles/Logistic.dir/main.cpp.obj"

# External object files for target Logistic
Logistic_EXTERNAL_OBJECTS =

Logistic.exe: CMakeFiles/Logistic.dir/main.cpp.obj
Logistic.exe: CMakeFiles/Logistic.dir/build.make
Logistic.exe: CMakeFiles/Logistic.dir/linklibs.rsp
Logistic.exe: CMakeFiles/Logistic.dir/objects1.rsp
Logistic.exe: CMakeFiles/Logistic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic\cmake-build-debug\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Logistic.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\Logistic.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Logistic.dir/build: Logistic.exe

.PHONY : CMakeFiles/Logistic.dir/build

CMakeFiles/Logistic.dir/requires: CMakeFiles/Logistic.dir/main.cpp.obj.requires

.PHONY : CMakeFiles/Logistic.dir/requires

CMakeFiles/Logistic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\Logistic.dir\cmake_clean.cmake
.PHONY : CMakeFiles/Logistic.dir/clean

CMakeFiles/Logistic.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic" "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic" "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic\cmake-build-debug" "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic\cmake-build-debug" "C:\Users\emanu\Documents\School\UTD S20\Machine Learning\HW4\Machine-Learning-Homework4\Logistic\cmake-build-debug\CMakeFiles\Logistic.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Logistic.dir/depend

