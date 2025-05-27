import os
import shutil
import fnmatch
import importlib
from distutils.core import setup
from distutils.dir_util import copy_tree
from distutils.extension import Extension
from distutils.command.build import build as _build

# Import CMakeExtension and CMakeBuild from setup_cmake_utils.py
from setup_cmake_utils import CMakeExtension, CMakeBuild


# Custom BuildCommand to copy the .so file and clean up the build directory
class BuildCommand(_build):
    def run(self):
        # Run the original build command
        self.run_command("build_ext")

        # Copy the generated .so file to the current directory
        build_lib_path = None
        for root, dirnames, filenames in os.walk(os.getcwd()):
            if fnmatch.fnmatch(root, os.path.join(os.getcwd(), "*build", "lib*")):
                build_lib_path = root
                break
        assert build_lib_path, "Could not find the build directory!"

        copy_tree(build_lib_path, os.getcwd())

        # Clean up the build directory
        build_dir = os.path.join(os.getcwd(), "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)

        # Verify that the module can be imported
        module_name = "ikfast_ur5e_robotiq"
        try:
            importlib.import_module(module_name)
            print(f"\nikfast module {module_name} imported successfully!")
        except ImportError as e:
            print(f"\nikfast module {module_name} import failed!")
            raise e


def main():
    setup(
        name="ikfast_ur5e_robotiq",
        version="1.0",
        description="ikfast wrapper for ur5e manipulator with robotiq gripper",
        ext_modules=[CMakeExtension("ikfast_ur5e_robotiq")],  # Use CMakeExtension
        setup_requires=["wheel"],
        cmdclass={
            "build": BuildCommand,  # Use the custom BuildCommand
            "build_ext": CMakeBuild,  # Use CMakeBuild from setup_cmake_utils.py
        },
    )


if __name__ == "__main__":
    main()
