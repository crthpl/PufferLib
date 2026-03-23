# Debug command:
#    DEBUG=1 python setup.py build_ext --inplace --force

from setuptools import find_packages, find_namespace_packages, setup
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

RAYLIB_A = f'{RAYLIB_NAME}/lib/libraylib.a'

# Compile args
extra_link_args = ['-fwrapv', '-fopenmp']

system = platform.system()
if system == 'Linux':
    extra_link_args += ['-Bsymbolic-functions']
elif system == 'Darwin':
    extra_link_args += ['-framework', 'Cocoa', '-framework', 'OpenGL', '-framework', 'IOKit']
else:
    raise ValueError(f'Unsupported system: {system}')

# ============================================================================
# Static env builds: clang-compiled env → static .a → linked into _C.so
# ============================================================================

OCEAN_DIR = 'ocean'
STATIC_ENVS = [
    name for name in os.listdir(OCEAN_DIR)
    if os.path.isdir(os.path.join(OCEAN_DIR, name))
    and os.path.exists(f'ocean/{name}/binding.c')
] if not NO_OCEAN else []

def _needs_rebuild(output, sources):
    if not os.path.exists(output):
        return True
    out_mtime = os.path.getmtime(output)
    for src in sources:
        if os.path.exists(src) and os.path.getmtime(src) > out_mtime:
            return True
    return False

def _extract_obs_tensor_t(obj_path):
    import subprocess
    out = subprocess.check_output(['strings', obj_path], text=True)
    for line in out.splitlines():
        if line.endswith('Tensor'):
            return line.strip()
    raise RuntimeError(f'Could not find OBS_TENSOR_T in {obj_path}')

def _build_static_lib(env_name, force=False):
    import subprocess
    env_binding_src = f'ocean/{env_name}/binding.c'
    static_lib = f'src/libstatic_{env_name}.a'
    static_obj = f'src/libstatic_{env_name}.o'

    env_deps = [env_binding_src, 'src/vecenv.h']
    if not force and not _needs_rebuild(static_lib, env_deps):
        print(f'Static env up to date: {static_lib}')
        return static_lib, _extract_obs_tensor_t(static_obj)

    clang_cmd = [
        'clang', '-c', '-O2', '-DNDEBUG',
        '-I.', '-Isrc', f'-Iocean/{env_name}',
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
    obs_tensor_t = _extract_obs_tensor_t(static_obj)
    print(f'OBS_TENSOR_T={obs_tensor_t}')
    return static_lib, obs_tensor_t

# ============================================================================
# _C.so build: single nvcc compile + link (no torch dependency)
# ============================================================================

_BINDINGS_CU_DEPS = [
    'src/bindings.cu', 'src/pufferlib.cu',
    'src/models.cu', 'src/kernels.cu',
    'src/vecenv.h', 'src/puffernet.h',
]

def _build_notorch_C(static_lib=None, obs_tensor_t=None, force=False, precision='bf16'):
    import subprocess
    import sysconfig

    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
    nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    output = f'pufferlib/_C{ext_suffix}'

    bindings_cu = 'src/bindings.cu'
    bindings_o = 'src/bindings.o'

    precision_flag = '-DPRECISION_FLOAT' if precision == 'float' else ''

    need_compile = force or _needs_rebuild(bindings_o, _BINDINGS_CU_DEPS)
    need_link = need_compile or force or _needs_rebuild(output, [bindings_o] + ([static_lib] if static_lib else []))

    if not need_link:
        print(f'Up to date: {output}')
        return

    python_include = sysconfig.get_path('include')
    pybind_include = pybind11.get_include()
    nvcc_cmd = [
        nvcc, '-c', '-Xcompiler', '-fPIC',
        '-Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1',
        '-Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-Xcompiler=-DPLATFORM_DESKTOP',
        '-std=c++17',
        '-I.', '-Isrc',
        f'-I{python_include}',
        f'-I{pybind_include}',
        f'-I{numpy.get_include()}',
        f'-I{cuda_home}/include',
        f'-I{RAYLIB_NAME}/include',
        '-Xcompiler=-fopenmp',
    ]
    if precision_flag:
        nvcc_cmd.append(precision_flag)
    if obs_tensor_t:
        nvcc_cmd.append(f'-DOBS_TENSOR_T={obs_tensor_t}')
    if DEBUG:
        nvcc_cmd += ['-O0', '-g']
    else:
        nvcc_cmd += ['-O3']
    nvcc_cmd += [bindings_cu, '-o', bindings_o]

    if need_compile:
        print(f'nvcc: {" ".join(nvcc_cmd)}')
        subprocess.check_call(nvcc_cmd)
    else:
        print(f'nvcc: up to date ({bindings_o})')

    link_cmd = [
        'g++', '-shared', '-fPIC', '-fopenmp',
        bindings_o,
    ]
    if static_lib:
        link_cmd.append(static_lib)
    link_cmd += [RAYLIB_A]
    link_cmd += [
        f'-L{cuda_home}/lib64',
        '-lcudart', '-lnccl', '-lnvidia-ml', '-lcublas', '-lcusolver', '-lcurand', '-lcudnn',
        '-lnvToolsExt', '-lomp5',
    ]
    if DEBUG:
        link_cmd += ['-g']
    else:
        link_cmd += ['-O2']
    link_cmd += extra_link_args
    link_cmd += ['-o', output]
    print(f'link: {" ".join(link_cmd)}')
    subprocess.check_call(link_cmd)
    print(f'Built: {output}')

# ============================================================================
# Build commands
# ============================================================================

class BuildExt(build_ext):
    user_options = build_ext.user_options + [
        ('precision=', None, 'Precision: float or bf16 (default: bf16)'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.precision = 'bf16'

    def finalize_options(self):
        super().finalize_options()

    def run(self):
        _build_notorch_C(force=self.force, precision=self.precision)

cmdclass = {"build_ext": BuildExt}

def create_static_env_build_class(env_name):
    class StaticEnvBuildExt(build_ext):
        user_options = build_ext.user_options + [
            ('precision=', None, 'Precision: float or bf16 (default: bf16)'),
        ]

        def initialize_options(self):
            super().initialize_options()
            self.precision = 'bf16'

        def finalize_options(self):
            super().finalize_options()

        def run(self):
            static_lib, obs_tensor_t = _build_static_lib(env_name, force=self.force)
            _build_notorch_C(static_lib, obs_tensor_t, force=self.force, precision=self.precision)
    return StaticEnvBuildExt

for env_name in STATIC_ENVS:
    cmdclass[f"build_{env_name}"] = create_static_env_build_class(env_name)

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

setup(
    packages=find_namespace_packages() + find_packages(),
    include_package_data=True,
    ext_modules=[],
    cmdclass=cmdclass,
    include_dirs=[numpy.get_include(), RAYLIB_NAME + '/include'],
)
