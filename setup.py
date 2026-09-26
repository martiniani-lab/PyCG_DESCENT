"""Build PyCG_DESCENT. Works with `pip install .` and `python setup.py build_ext -i`.

pele must be importable (installed, or a source checkout on PYTHONPATH). Its C++
headers are found with pele.get_include(). The Cython extensions are compiled by
CMake (from CMakeLists.txt.in) inside build_ext; setuptools then just copies them.

Options (command-line flags for direct `setup.py` use, env vars for pip):
  -j N / PYCGD_JOBS               parallel build jobs (default: all cores)
  -c COMPILER                      unix (default) or intel
  --build-type / PYCGD_BUILD_TYPE Release (default), Debug, RelWithDebInfo, MemCheck
  --native / PYCGD_NATIVE         1 (default) adds -march=native; set 0 for portable binaries
"""
import argparse
import importlib.machinery
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as old_build_ext
from setuptools.command.build_py import build_py

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-j", type=int, default=int(os.environ.get("PYCGD_JOBS", os.cpu_count() or 4)))
parser.add_argument("-c", "--compiler", type=str, default=None)
parser.add_argument("--opt-report", action="store_true", default=False,
                    help="Print optimization report (for Intel compiler)")
parser.add_argument("--build-type", type=str, default=os.environ.get("PYCGD_BUILD_TYPE", "Release"),
                    help="Release, Debug, RelWithDebInfo, MemCheck")
parser.add_argument("--native", type=int, default=int(os.environ.get("PYCGD_NATIVE", 1)),
                    help="Compile with -march=native (non-portable binaries)")
jargs, remaining_args = parser.parse_known_args(sys.argv)

if not jargs.compiler or jargs.compiler in ("unix", "gnu", "gcc"):
    idcompiler = "unix"
elif jargs.compiler in ("intelem", "intel", "icc", "icpc"):
    idcompiler = "intel"
else:
    raise ValueError("unknown compiler " + jargs.compiler)
if jargs.compiler:
    remaining_args += ["-c", idcompiler]
sys.argv = remaining_args

build_type = jargs.build_type
build_type_args = {
    "Release": ["-O3", "-DNDEBUG"],
    "Debug": ["-ggdb3", "-O0"],
    "RelWithDebInfo": ["-g", "-O3"],
    "MemCheck": ["-g", "-O0", "-fsanitize=address", "-fsanitize=leak"],
}
if build_type not in build_type_args:
    raise ValueError(f"Unknown build type: {build_type}")
# env CXXFLAGS first (conda-forge hardening/arch flags), ours after so they win
cmake_compiler_extra_args = (
    shlex.split(os.environ.get("CXXFLAGS", ""))
    + ["-std=c++2a", "-Wall", "-Wextra", "-pedantic", "-fPIC"]
    + build_type_args[build_type]
)
if jargs.native and build_type in ("Release", "RelWithDebInfo"):
    cmake_compiler_extra_args += ["-march=native"]
if idcompiler == "unix":
    cmake_compiler_extra_args += ["-fopenmp"]
else:
    cmake_compiler_extra_args += ["-axCORE-AVX2", "-qopenmp", "-ip", "-unroll"]
    if jargs.opt_report:
        cmake_compiler_extra_args.append("-qopt-report=5")

cmake_build_dir = os.path.join("build", "cmake")

cxx_files = [
    "PyCG_DESCENT/_pycgd.cxx",
]


def git_version():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C"})
        return out.stdout.strip().decode() or "Unknown"
    except OSError:
        return "Unknown"


def package_dir(name):
    """Directory of an installed dependency, found without importing it. pele is not on
    PyPI so it can't be a build requirement; under pip's build isolation the
    environment's site-packages is off sys.path, so it is searched too."""
    site = list({sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]})
    spec = importlib.util.find_spec(name) or importlib.machinery.PathFinder.find_spec(name, site)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"{name} must be installed first: "
                           f"pip install git+https://github.com/martiniani-lab/{name}")
    return os.path.abspath(spec.submodule_search_locations[0])


def pele_paths():
    """(pele C++ source dir, pele python package dir, extra cmake prefix paths)"""
    pele_pkg = package_dir("pele")
    # same rule as pele.get_include(): installed next to the package, or a checkout's source/
    pele_include = os.path.join(pele_pkg, "source")
    if not os.path.isdir(pele_include):
        pele_include = os.path.join(os.path.dirname(pele_pkg), "source")
    # a pele source checkout may carry its own sundials/eigen in extern/install
    extern = os.path.join(os.path.dirname(pele_include), "extern", "install")
    return pele_include, pele_pkg, [extern] if os.path.isdir(extern) else []


def generate_cython(pele_pkg):
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    cmd = [sys.executable, os.path.join(cwd, "cythonize.py"), "PyCG_DESCENT",
           "-I", os.path.join(pele_pkg, "potentials"), "-I", os.path.dirname(pele_pkg)]
    if build_type in ["Debug", "RelWithDebInfo", "MemCheck"]:
        cmd += ["--gdb", "--annotate", "-X", "linetrace=True", "-X", "boundscheck=True",
                "-X", "wraparound=False", "-X", "cdivision=False"]
    cmd += ["-X", "language_level=3", "-X", "c_string_type=unicode", "-X", "c_string_encoding=utf-8"]
    if subprocess.call(cmd, cwd=cwd) != 0:
        raise RuntimeError("Running cythonize failed!")


def get_ldflags():
    """linker flags for libpython (only used on macOS, see CMakeLists.txt.in)"""
    gv = sysconfig.get_config_var
    libs = (gv("LIBS") or "").split() + (gv("SYSLIBS") or "").split()
    if not gv("Py_ENABLE_SHARED"):
        libs.insert(0, "-L" + gv("LIBDIR"))
    if not gv("PYTHONFRAMEWORK"):
        # -stack_size is only valid for executables, see
        # https://github.com/kovidgoyal/kitty/issues/289#issuecomment-416040645
        libs += [f for f in (gv("LINKFORSHARED") or "").split() if not f.startswith("-Wl,-stack_size")]
    return " ".join(libs)


def write_cmakelists(pele_include, prefix_paths):
    import numpy as np

    with open("CMakeLists.txt.in") as fin:
        template = fin.read()
    python_includes = {sysconfig.get_path("include"), sysconfig.get_path("platinclude")}
    cmake_txt = (
        template.replace("__PELE_INCLUDE__", pele_include)
        .replace("__EXTRA_PREFIX_PATH__", " ".join(prefix_paths))
        .replace("__PYTHON_INCLUDE__", " ".join(sorted(python_includes)))
        .replace("__NUMPY_INCLUDE__", np.get_include())
        .replace("__PYTHON_LDFLAGS__", get_ldflags())
        .replace("__COMPILER_EXTRA_ARGS__", '"%s"' % " ".join(cmake_compiler_extra_args))
    )
    with open("CMakeLists.txt", "w") as fout:
        fout.write(cmake_txt + "\n")
        for src in cxx_files:
            fout.write(f"make_cython_lib(${{CMAKE_CURRENT_SOURCE_DIR}}/{src})\n")


def get_compiler_env(cid):
    """CC/CXX from the environment (e.g. conda compilers) are respected"""
    env = os.environ.copy()
    if cid == "unix":
        # match pele: on macOS use the newest homebrew gcc-N (Apple clang has no -fopenmp,
        # and libc++ vs libstdc++ would break C++ objects shared with pele's extensions)
        if sys.platform.startswith("darwin") and "CC" not in env:
            version = next((v for v in range(20, 9, -1) if shutil.which(f"gcc-{v}")), None)
            if version is None:
                raise RuntimeError("Could not find a homebrew GNU compiler gcc-N (N=10..20) on PATH. "
                                   "Install one or set CC and CXX.")
            env["CC"], env["CXX"] = shutil.which(f"gcc-{version}"), shutil.which(f"g++-{version}")
        env.setdefault("CC", "gcc")
        env.setdefault("CXX", "g++")
        args = []
    else:
        env["CC"], env["CXX"], env["AR"] = shutil.which("icc"), shutil.which("icpc"), shutil.which("xiar")
        args = [f"-DCMAKE_AR={env['AR']}"]
    return env, args + [f"-DCMAKE_C_COMPILER={env['CC']}", f"-DCMAKE_CXX_COMPILER={env['CXX']}"]


def run_cmake():
    os.makedirs(cmake_build_dir, exist_ok=True)
    cwd = os.path.abspath(os.path.dirname(__file__))
    env, cmake_args = get_compiler_env(idcompiler)
    if shutil.which("ninja"):
        cache = os.path.join(cmake_build_dir, "CMakeCache.txt")
        # cmake refuses to switch generators in an existing build dir
        if os.path.isfile(cache) and "CMAKE_GENERATOR:INTERNAL=Ninja\n" not in open(cache).read():
            os.remove(cache)
            shutil.rmtree(os.path.join(cmake_build_dir, "CMakeFiles"), ignore_errors=True)
        cmake_args += ["-G", "Ninja"]
    if build_type == "Release":
        # CMake picks the LTO-aware archiver (gcc-ar) for the static PyCG_DESCENT_lib
        cmake_args += ["-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"]
    subprocess.check_call(["cmake", *cmake_args, cwd], cwd=cmake_build_dir, env=env)
    subprocess.check_call(["cmake", "--build", ".", "-j", str(jargs.j)], cwd=cmake_build_dir, env=env)
    print("CMake build completed")


class build_ext_precompiled(old_build_ext):
    """Build everything with CMake, then copy each library (stored in
    extension.sources[0]) to where setuptools expects the extension."""

    def run(self):
        pele_include, pele_pkg, prefix_paths = pele_paths()
        generate_cython(pele_pkg)
        write_cmakelists(pele_include, prefix_paths)
        run_cmake()
        super().run()

    def build_extension(self, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        lib = ext.sources[0]
        if not os.path.isfile(lib):
            raise RuntimeError(f"file does not exist: {lib} Did CMake not run correctly")
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        shutil.copy2(lib, ext_path)


class build_py_with_source(build_py):
    """also install the C sources/headers as PyCG_DESCENT/source, see PyCG_DESCENT.get_include()"""

    def run(self):
        super().run()
        dest = os.path.join(self.build_lib, "PyCG_DESCENT", "source")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree("source", dest, ignore=shutil.ignore_patterns("*.rst"))


extensions = [
    Extension(
        src.replace("/", ".").rsplit(".", 1)[0],
        [os.path.join(cmake_build_dir, os.path.basename(src).replace(".cxx", ".so"))],
    )
    for src in cxx_files
]

# written before setup() so build_py installs the current one
with open("PyCG_DESCENT/version.py", "w") as f:
    f.write("\n# THIS FILE IS GENERATED FROM SCIPY SETUP.PY\ngit_revision = '%s'\n" % git_version())

# metadata lives in pyproject.toml
setup(
    packages=find_packages(include=["PyCG_DESCENT", "PyCG_DESCENT.*"]),
    package_data={"": ["*.pxd"]},
    cmdclass={"build_ext": build_ext_precompiled, "build_py": build_py_with_source},
    ext_modules=extensions,
)
