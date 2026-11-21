"""crit Core for Python: model files, model config and the text layer (tokenizer, chat templates, stops)."""
__version__ = "0.22.0"


def cache_home(*parts):
    """The folder every cache crit keeps lives under, and `parts` below it: CRIT_CACHE_HOME, else ~/.cache/crit. Each
    cache's own variable (CRIT_KERNEL_CACHE, CRIT_CORE_CACHE, ...) still names another folder for it. Everything under
    it can be built again, so the whole folder may be deleted at any time."""
    import os
    from pathlib import Path
    root = os.environ.get("CRIT_CACHE_HOME")
    return Path(root).expanduser().joinpath(*parts) if root else Path.home().joinpath(".cache", "crit", *parts)


def _no_core_files():
    """No core file from a process that uses crit: an engine, a test or a script holding tens of GB of mapped weights
    and pinned memory writes one that large when it crashes (under WSL, into %TEMP%\\wsl-crashes on the Windows disk).
    A soft limit of 1 byte, since pipe-based core handlers ignore 0. CRIT_CORE_FILES=1 keeps the system's limit."""
    import os
    if os.name == "nt" or os.environ.get("CRIT_CORE_FILES") == "1":
        return
    try:
        import resource
        hard = resource.getrlimit(resource.RLIMIT_CORE)[1]
        if hard == resource.RLIM_INFINITY or hard >= 1:
            resource.setrlimit(resource.RLIMIT_CORE, (1, hard))
    except (ImportError, OSError, ValueError):
        pass


_no_core_files()
