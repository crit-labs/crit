// The two macros a Kernel Hub kernel's torch-ext binding uses (kernel-builder generates this header in its own builds),
// so a Hub kernel's source builds with torch's extension builder on the workbench: TORCH_LIBRARY_EXPAND registers the
// ops under the extension's name and REGISTER_EXTENSION gives Python a module to import.
#pragma once

#include <Python.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                                                       \
    PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                           \
        static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
        return PyModule_Create(&module);                                                               \
    }
