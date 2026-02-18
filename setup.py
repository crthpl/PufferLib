# Debug command:
#    DEBUG=1 python setup.py build_ext --inplace --force
#    CUDA_VISIBLE_DEVICES=None LD_PRELOAD=$(gcc -print-file-name=libasan.so) python3.12 -m pufferlib.clean_pufferl eval --train.device cpu

from setuptools import find_packages, find_namespace_packages, setup, Extension
import numpy
import os
import glob
import urllib.request
import zipfile
import tarfile
import platform
import shutil
import pybind11

from setuptools.command.build_ext import build_ext

# Build flags
DEBUG = os.getenv("DEBUG", "0") == "1"
NO_OCEAN = os.getenv("NO_OCEAN", "0") == "1"
NO_TRAIN = os.getenv("NO_TRAIN", "0") == "1"
NO_TORCH = os.getenv("NO_TORCH", "0") == "1"

# Import torch build utilities (unless NO_TORCH or NO_TRAIN)
HAS_TORCH = False
if not NO_TRAIN and not NO_TORCH:
    from torch.utils import cpp_extension
    from torch.utils.cpp_extension import (
        CppExtension,
        CUDAExtension,
        BuildExtension,
        CUDA_HOME,
        ROCM_HOME
    )
    HAS_TORCH = True
    # build cuda extension if torch can find CUDA or HIP/ROCM in the system
    BUID_CUDA_EXT = bool(CUDA_HOME or ROCM_HOME)
else:
    BUID_CUDA_EXT = False

# Use ccache if available for faster rebuilds
if shutil.which('ccache'):
    os.environ.setdefault('CC', 'ccache cc')
    os.environ.setdefault('CXX', 'ccache c++')

# Build raylib for your platform
RAYLIB_URL = 'https://github.com/raysan5/raylib/releases/download/5.5/'
RAYLIB_NAME = 'raylib-5.5_macos' if platform.system() == "Darwin" else 'raylib-5.5_linux_amd64'
RLIGHTS_URL = 'https://raw.githubusercontent.com/raysan5/raylib/refs/heads/master/examples/shaders/rlights.h'

def download_raylib(platform, ext):
    if not os.path.exists(platform):
        print(f'Downloading Raylib {platform}')
        urllib.request.urlretrieve(RAYLIB_URL + platform + ext, platform + ext)
        if ext == '.zip':
            with zipfile.ZipFile(platform + ext, 'r') as zip_ref:
                zip_ref.extractall()
        else:
            with tarfile.open(platform + ext, 'r') as tar_ref:
                tar_ref.extractall()

        os.remove(platform + ext)
        urllib.request.urlretrieve(RLIGHTS_URL, platform + '/include/rlights.h')

if not NO_OCEAN:
    download_raylib('raylib-5.5_webassembly', '.zip')
    download_raylib(RAYLIB_NAME, '.tar.gz')

BOX2D_URL = 'https://github.com/capnspacehook/box2d/releases/latest/download/'
BOX2D_NAME = 'box2d-macos-arm64' if platform.system() == "Darwin" else 'box2d-linux-amd64'

def download_box2d(platform):
    if not os.path.exists(platform):
        ext = ".tar.gz"

        print(f'Downloading Box2D {platform}')
        urllib.request.urlretrieve(BOX2D_URL + platform + ext, platform + ext)
        with tarfile.open(platform + ext, 'r') as tar_ref:
            tar_ref.extractall()

        os.remove(platform + ext)

if not NO_OCEAN:
    download_box2d('box2d-web')
    download_box2d(BOX2D_NAME)

# Shared compile args for all platforms
extra_compile_args = [
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-DPLATFORM_DESKTOP',
]
extra_link_args = [
    '-fwrapv',
    '-fopenmp',
]
cxx_args = [
    '-fdiagnostics-color=always',
    '-std=c++17',
    '-fopenmp',
]
nvcc_args = [
    '-Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1',
    '-std=c++17',
]

if DEBUG:
    extra_compile_args += [
        '-O0',
        '-g',
        #'-fsanitize=address,undefined,bounds,pointer-overflow,leak',
        #'-fno-omit-frame-pointer',
    ]
    extra_link_args += [
        '-g',
        #'-fsanitize=address,undefined,bounds,pointer-overflow,leak',
    ]
    cxx_args += [
        '-O0',
        '-g',
    ]
    nvcc_args += [
        '-O0',
        '-g',
    ]
else:
    extra_compile_args += [
        '-O2',
        '-flto=auto',
        '-fno-semantic-interposition',
        '-fvisibility=hidden',
    ]
    extra_link_args += [
        '-O2',
    ]
    cxx_args += [
        '-O2',
        '-fno-semantic-interposition',
        '-Wno-c++11-narrowing',
    ]
    nvcc_args += [
        '-O3',
    ]

system = platform.system()
if system == 'Linux':
    extra_compile_args += [
        '-Wno-alloc-size-larger-than',
        '-fmax-errors=3',
    ]
    extra_link_args += [
        '-Bsymbolic-functions',
    ]
elif system == 'Darwin':
    extra_compile_args += [
        '-Wno-error=int-conversion',
        '-Wno-error=incompatible-function-pointer-types',
        '-Wno-error=implicit-function-declaration',
    ]
    extra_link_args += [
        '-framework', 'Cocoa',
        '-framework', 'OpenGL',
        '-framework', 'IOKit',
    ]
else:
    raise ValueError(f'Unsupported system: {system}')

# Default Gym/Gymnasium/PettingZoo versions
# Gym:
# - 0.26 still has deprecation warnings and is the last version of the package
# - 0.25 adds a breaking API change to reset, step, and render_modes
# - 0.24 is broken
# - 0.22-0.23 triggers deprecation warnings by calling its own functions
# - 0.21 is the most stable version
# - <= 0.20 is missing dict methods for gym.spaces.Dict
# - 0.18-0.21 require setuptools<=65.5.0

# Extensions
extnames = ["pufferlib._C", "squared_torch._C"]

class BuildExt(build_ext):
    def run(self):
        # Propagate any build_ext options (e.g., --inplace, --force) to subcommands
        build_ext_opts = self.distribution.command_options.get('build_ext', {})
        if build_ext_opts:
            # Copy flags so build_torch and build_c respect inplace/force
            self.distribution.command_options['build_torch'] = build_ext_opts.copy()
            self.distribution.command_options['build_c'] = build_ext_opts.copy()

        # Run the torch and C builds (which will handle copying when inplace is set)
        if HAS_TORCH:
            self.run_command('build_torch')
        elif not NO_TRAIN:
            # NO_TORCH but not NO_TRAIN: build _C.so via subprocess (no env linked)
            _build_notorch_C()
        self.run_command('build_c')

class CBuildExt(build_ext):
    def run(self, *args, **kwargs):
        self.extensions = [e for e in self.extensions if e.name not in extnames]
        super().run(*args, **kwargs)

# Define cmdclass outside of setup to add dynamic commands
cmdclass = {
    "build_ext": BuildExt,
    "build_c": CBuildExt,
}

if HAS_TORCH:
    class TorchBuildExt(cpp_extension.BuildExtension):
        def run(self):
            self.extensions = [e for e in self.extensions if e.name in extnames]
            super().run()

    cmdclass["build_torch"] = TorchBuildExt

INCLUDE = [f'{BOX2D_NAME}/include', f'{BOX2D_NAME}/src']
RAYLIB_A = f'{RAYLIB_NAME}/lib/libraylib.a'
extension_kwargs = dict(
    include_dirs=INCLUDE,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    extra_objects=[RAYLIB_A],
)

# Find C extensions
c_extensions = []
c_extension_paths = []
if not NO_OCEAN:
    c_extension_paths = glob.glob('pufferlib/ocean/**/binding.c', recursive=True)
    c_extensions = [
        Extension(
            path.rstrip('.c').replace('/', '.'),
            sources=[path],
            **extension_kwargs,
        )
        for path in c_extension_paths if 'matsci' not in path
    ]
    c_extension_paths = [os.path.join(*path.split('/')[:-1]) for path in c_extension_paths]

    for c_ext in c_extensions:
        if "impulse_wars" in c_ext.name:
            print(f"Adding {c_ext.name} to extra objects")
            c_ext.extra_objects.append(f'{BOX2D_NAME}/libbox2d.a')
            # TODO: Figure out why this is necessary for some users
            impulse_include = 'pufferlib/ocean/impulse_wars/include'
            if impulse_include not in c_ext.include_dirs:
                c_ext.include_dirs.append(impulse_include)

        if 'matsci' in c_ext.name:
            c_ext.include_dirs.append('/usr/local/include')
            c_ext.extra_link_args.extend(['-L/usr/local/lib', '-llammps'])


class ProfilerBuildExt(build_ext):
    user_options = build_ext.user_options + [
        ('no-torch', None, 'Build profiler without torch support'),
        ('env=', None, 'Static env to link (e.g., breakout, drive)'),
        ('precision=', None, 'Precision: float or bf16 (default: float)'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.no_torch = False
        self.env = None
        self.precision = 'float'

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        import subprocess
        import sysconfig
        import torch.utils.cpp_extension as cpp_ext

        src = 'profile_kernels.cu'
        out = 'profile'

        nvcc = cpp_ext._join_cuda_home('bin', 'nvcc')
        arch = '-arch=sm_89'

        cmd = [nvcc, '-O3', arch, '-I.', src, '-o', out]

        if not self.no_torch:
            out = 'profile_torch'
            lib_paths = cpp_ext.library_paths()
            nvtx_lib_dir = os.path.join(cpp_ext.CUDA_HOME, 'lib64')

            precision_flag = '-DPRECISION_FLOAT' if self.precision == 'float' else ''
            cmd = [nvcc, '-O3', arch, '-DUSE_TORCH',
                   '-I.', '-Ipufferlib/extensions']
            if precision_flag:
                cmd.append(precision_flag)

            # Optional static env for envspeed profiling
            if self.env:
                static_lib = f'pufferlib/extensions/libstatic_{self.env}.a'
                if not os.path.exists(static_lib):
                    raise RuntimeError(f'Static library not found: {static_lib}\n'
                                       f'Build it first with: python setup.py build_{self.env}')
                cmd += [f'-DENV_NAME={self.env}', '-DUSE_STATIC_ENV',
                        f'-I./{RAYLIB_NAME}/include']

            cmd += ['-I' + sysconfig.get_path('include')]
            cmd += ['-I' + p for p in cpp_ext.include_paths()]
            cmd += ['-L' + p for p in lib_paths]
            cmd += ['-L' + nvtx_lib_dir]
            cmd += ['-Xlinker', '-rpath,' + ':'.join(lib_paths)]
            cmd += ['-Xlinker', '--no-as-needed']
            cmd += ['-lc10', '-lc10_cuda', '-ltorch', '-ltorch_cpu', '-ltorch_cuda', '-lnvToolsExt', '-ldl', '-lnccl']
            cmd += ['-Xlinker', '--unresolved-symbols=ignore-in-shared-libs']
            if self.env:
                static_lib = f'pufferlib/extensions/libstatic_{self.env}.a'
                cmd += [static_lib, f'./{RAYLIB_NAME}/lib/libraylib.a', '-lGL']
            cmd += ['-lomp5']
            cmd += [src, '-o', out]

        print(f'Building profiler: {" ".join(cmd)}')
        subprocess.check_call(cmd)
        print(f'Built: {out}')

cmdclass["build_profiler"] = ProfilerBuildExt

# Static env builds: clang-compiled env + gcc/nvcc torch extension
# Discover envs by listing folders in pufferlib/ocean
OCEAN_DIR = 'pufferlib/ocean'
STATIC_ENVS = [
    name for name in os.listdir(OCEAN_DIR)
    if os.path.isdir(os.path.join(OCEAN_DIR, name))
    and not name.startswith('__')
    and os.path.exists(f'pufferlib/ocean/{name}/binding.h')
]

def _build_static_lib(env_name):
    """Build a static .a library for a given env using clang."""
    import subprocess
    env_binding_src = 'pufferlib/extensions/env_binding.c'
    static_lib = f'pufferlib/extensions/libstatic_{env_name}.a'
    static_obj = f'pufferlib/extensions/libstatic_{env_name}.o'

    clang_cmd = [
        'clang', '-c', '-O2', '-DNDEBUG',
        '-I.', '-Ipufferlib/extensions', f'-Ipufferlib/ocean/{env_name}',
        f'-I./{RAYLIB_NAME}/include', '-I/usr/local/cuda/include',
        '-DPLATFORM_DESKTOP',
        '-fno-semantic-interposition', '-fvisibility=hidden',
        '-fPIC', '-fopenmp',
        env_binding_src, '-o', static_obj
    ]
    print(f'Building static env: {" ".join(clang_cmd)}')
    subprocess.check_call(clang_cmd)

    ar_cmd = ['ar', 'rcs', static_lib, static_obj]
    print(f'Creating static library: {" ".join(ar_cmd)}')
    subprocess.check_call(ar_cmd)
    return static_lib

def _build_notorch_C(static_lib=None):
    """Build _C.so via subprocess using nvcc + g++ (no torch dependency)."""
    import subprocess
    import sysconfig

    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
    nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    output = f'pufferlib/_C{ext_suffix}'

    # Step 1: nvcc compile modules.cu -> modules.o
    modules_cu = 'pufferlib/extensions/cuda/modules.cu'
    modules_o = 'pufferlib/extensions/cuda/modules.o'
    nvcc_cmd = [
        nvcc, '-c', '-Xcompiler', '-fPIC',
        '-Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1',
        '-std=c++17',
        '-I.', '-Ipufferlib/extensions',
        f'-I{cuda_home}/include',
    ]
    if DEBUG:
        nvcc_cmd += ['-O0', '-g']
    else:
        nvcc_cmd += ['-O3']
    nvcc_cmd += [modules_cu, '-o', modules_o]
    print(f'[NO_TORCH] nvcc: {" ".join(nvcc_cmd)}')
    subprocess.check_call(nvcc_cmd)

    # Step 2: g++ compile bindings.cpp -> bindings.o
    bindings_cpp = 'pufferlib/extensions/bindings.cpp'
    bindings_o = 'pufferlib/extensions/bindings.o'
    python_include = sysconfig.get_path('include')
    pybind_include = pybind11.get_include()
    gxx_cmd = [
        'g++', '-c', '-fPIC', '-std=c++17',
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-DPLATFORM_DESKTOP',
        '-I.', '-Ipufferlib/extensions',
        f'-I{python_include}',
        f'-I{pybind_include}',
        f'-I{numpy.get_include()}',
        f'-I{cuda_home}/include',
        f'-I{RAYLIB_NAME}/include',
        '-fopenmp',
    ]
    if DEBUG:
        gxx_cmd += ['-O0', '-g']
    else:
        gxx_cmd += ['-O2', '-fno-semantic-interposition', '-fvisibility=hidden']
    gxx_cmd += [bindings_cpp, '-o', bindings_o]
    print(f'[NO_TORCH] g++: {" ".join(gxx_cmd)}')
    subprocess.check_call(gxx_cmd)

    # Step 3: Link into shared library
    link_cmd = [
        'g++', '-shared', '-fPIC',
        '-fopenmp',
        modules_o, bindings_o,
    ]
    if static_lib:
        link_cmd.append(static_lib)
    link_cmd += [RAYLIB_A]
    link_cmd += [
        f'-L{cuda_home}/lib64',
        '-lcudart', '-lnccl', '-lnvidia-ml', '-lcublas', '-lcusolver', '-lcurand',
        '-lnvToolsExt', '-lomp5',
    ]
    if DEBUG:
        link_cmd += ['-g']
    else:
        link_cmd += ['-O2']
    link_cmd += extra_link_args
    link_cmd += ['-o', output]
    print(f'[NO_TORCH] link: {" ".join(link_cmd)}')
    subprocess.check_call(link_cmd)
    print(f'[NO_TORCH] Built: {output}')

def create_static_env_build_class(env_name):
    """Create a build class that compiles env with clang, and optionally links with torch extension."""
    if HAS_TORCH:
        # With torch: build static lib + torch extension linked against it
        class StaticEnvBuildExt(cpp_extension.BuildExtension):
            def run(self):
                static_lib = _build_static_lib(env_name)
                self.extensions = [e for e in self.extensions if e.name == 'pufferlib._C']
                for ext in self.extensions:
                    ext.extra_objects = [RAYLIB_A, static_lib]
                super().run()
        return StaticEnvBuildExt
    else:
        # Without torch: build static .a library + _C.so via subprocess
        class StaticEnvBuildExtNoTorch(build_ext):
            def run(self):
                static_lib = _build_static_lib(env_name)
                _build_notorch_C(static_lib)
        return StaticEnvBuildExtNoTorch

# Add build_<env> for static-linked envs (always available, torch optional)
for env_name in STATIC_ENVS:
    cmdclass[f"build_{env_name}"] = create_static_env_build_class(env_name)

if not NO_OCEAN:
    def create_env_build_class(full_name):
        class EnvBuildExt(build_ext):
            def run(self):
                self.extensions = [e for e in self.extensions if e.name == full_name]
                super().run()
        return EnvBuildExt

    # Add a build_<env>_so command for each env (dynamic .so build)
    for c_ext in c_extensions:
        env_name = c_ext.name.split('.')[-2]
        cmdclass[f"build_{env_name}_so"] = create_env_build_class(c_ext.name)


# Torch extensions (only when torch is available and not disabled)
torch_extensions = []
if HAS_TORCH:
    import torch
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or torch.utils.cpp_extension.CUDA_HOME or '/usr/local/cuda'
    nvtx_lib_dir = os.path.join(cuda_home, 'lib64')  # Common on Linux; fall back to 'lib' if needed
    nvtx_lib = 'nvToolsExt'

    torch_sources = [
        "pufferlib/extensions/bindings.cpp",
    ]
    if BUID_CUDA_EXT:
        extension = CUDAExtension
        torch_sources.append("pufferlib/extensions/cuda/squared_torch.cu")
        torch_sources.append("pufferlib/extensions/cuda/modules.cu")
    else:
        extension = CppExtension

    # Note: Use build_<envname> (e.g. build_breakout, build_drive) to build with static env linking
    # build_torch alone won't link any env - it's for the training code only
    torch_extensions = [
       extension(
            "pufferlib._C",
            torch_sources,
            include_dirs=[pybind11.get_include(), torch.utils.cpp_extension.include_paths()[0]],
            extra_compile_args = {
                "cxx": extra_compile_args + cxx_args + ['-DPUFFERLIB_TORCH'],
                "nvcc": nvcc_args + ['-DPUFFERLIB_TORCH'],
            },
            extra_link_args=extra_link_args,
            extra_objects=[RAYLIB_A],
            libraries=[nvtx_lib, 'omp5', 'nccl', 'nvidia-ml', 'cublas', 'cusolver', 'curand'],
            library_dirs=[nvtx_lib_dir],
        ),
    ]

# Prevent Conda from injecting garbage compile flags
from distutils.sysconfig import get_config_vars
cfg_vars = get_config_vars()
for key in ('CC', 'CXX', 'LDSHARED'):
    if cfg_vars[key]:
        cfg_vars[key] = cfg_vars[key].replace('-B /root/anaconda3/compiler_compat', '')
        cfg_vars[key] = cfg_vars[key].replace('-pthread', '')
        cfg_vars[key] = cfg_vars[key].replace('-fno-strict-overflow', '')

for key, value in cfg_vars.items():
    if value and '-fno-strict-overflow' in str(value):
        cfg_vars[key] = value.replace('-fno-strict-overflow', '')

install_requires = [
    'setuptools',
    'numpy<2.0',
    'shimmy[gym-v21]',
    'gym==0.23',
    'gymnasium>=0.29.1',
    'pettingzoo>=1.24.1',
]

if not NO_TRAIN:
    install_requires += [
        'torch>=2.9',
        'psutil',
        'nvidia-ml-py',
        'rich',
        'rich_argparse',
        'imageio',
        'gpytorch',
        'scikit-learn',
        'heavyball>=2.2.0', # contains relevant fixes compared to 1.7.2 and 2.1.1
        'neptune',
        'wandb',
    ]

setup(
    version="3.0.0",
    packages=find_namespace_packages() + find_packages() + c_extension_paths + ['pufferlib/extensions'],
    package_data={
        "pufferlib": [RAYLIB_NAME + '/lib/libraylib.a']
    },
    include_package_data=True,
    install_requires=install_requires,
    ext_modules = c_extensions + torch_extensions,
    cmdclass=cmdclass,
    include_dirs=[numpy.get_include(), RAYLIB_NAME + '/include'],
)
