# Cấu hình biên dịch Cython
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Sử dụng Extension để ép cứng tên module là core.model_inference
# Cython sẽ không tự dò tên thư mục cha có chứa dấu gạch ngang nữa
extensions = [
    Extension(
        name="core.model_inference",
        sources=["core/model_inference.pyx"],
        include_dirs=[np.get_include()]
    )
]

# Build extension đã được định nghĩa
setup(
    ext_modules=cythonize(
        extensions, 
        compiler_directives={'language_level': "3"}
    )
)