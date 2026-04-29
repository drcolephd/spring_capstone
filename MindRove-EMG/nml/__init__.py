try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.8"  # fallback for weird/no-SCM installs
