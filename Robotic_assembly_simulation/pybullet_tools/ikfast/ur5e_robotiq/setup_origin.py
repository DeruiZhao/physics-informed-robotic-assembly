import setuptools
from distutils.core import setup, Extension
from distutils.command.build import build as _build
import os
import shutil


class BuildCommand(_build):
    def run(self):
        _build.run(self)

        for root, dirs, files in os.walk(self.build_lib):
            for file in files:
                if file.endswith(".so"):
                    so_file = os.path.join(root, file)

                    shutil.copy(so_file, os.path.dirname(os.path.abspath(__file__)))

        build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)


def main():
    setup(
        name="ikfast_ur5e_robotiq",
        version="1.0",
        description="ikfast wrapper for ur5e manipulator with robotiq gripper",
        ext_modules=[
            Extension(
                "ikfast_ur5e_robotiq",
                ["ikfast_ur5e_robotiq.cpp", "ikfast_pybind_ur5e_robotiq.cpp"],
            )
        ],
        setup_requires=["wheel"],
        cmdclass={
            "build": BuildCommand,
        },
    )


if __name__ == "__main__":
    main()
