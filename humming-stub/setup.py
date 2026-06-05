from setuptools import setup, find_packages
setup(
    name="humming-stub",
    version="0.0.0",
    packages=find_packages(),
    description="Stub package to satisfy vLLM's eager `from humming.* import ...` on systems without the real humming library installed.",
)
