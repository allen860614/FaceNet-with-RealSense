#include <Python.h>

static PyObject* hello(PyObject* self, PyObject* args) {
    printf("Hello! A C function called!\n");
    Py_RETURN_NONE;
}

static PyMethodDef HelloMethods[] = {
    {"hello", hello, METH_VARARGS, "A hello function."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC inithello(void) {
     (void) Py_InitModule("hello", HelloMethods);
}