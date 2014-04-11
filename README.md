# GPU based Multi-View Deconvolution using cuFFT

This package aims at providing an easy-to-use API for calling GPU deconvolution routines from the Fiji/SPIM_Registration plugin. It is written in C++ and offers C-style API of plain methods to call from any client binary or library. 

# Building the Library

## Operating Systems

We currently support the following operating systems:

* OSX 10.8 or higher (experimental)
* Fedora Linux 19 or higher
* CentOS Linux 6.3 or higher

## Dependencies

* CUDA 5 or later (the nvcc compiler, libcudart and libcufft) must be available.
* cmake versions later than 2.8 ( FindCUDA.cmake sources are needed )
* boost 1.40 or later
* fftw 3.1 or later

## Install:
To compile the shared library libMultiViewNative.so, invoke
```bash
$ mkdir build
$ cd build
```

If a specific installation destination is to be used
```bash
$ cmake -DCMAKE_INSTALL_PREFIX=/path/to/lib ..
$ make
$ make install
```

or else (will install library in system default library paths, administrator rights required)
```bash
$ cmake .. 
$ make
$ make install
```


# Using the Library

## Fiji related

In order to call the GPU methods from Java, simply create a directory called lib/linux64 under the fiji root directory. Create a symlink to libMultiViewNative.so here.

## License

see LICENSE

## Contact

For questions, bug reports, improvement suggestions or else, use the github issue tracker of this repo.

## Known Issues

On OS X, CUDA is (at the time of writing) INCOMPATIBLE with the installed llvm based compilers used by default (e.g. /usr/bin/c++). To resolve this, the installation procedure uses an llvm based gcc version for compiling and testing.

## Disclaimer

The library at hand is an evolving project. It is designed as the native implementation of http://fiji.sc/Multi-View_Deconvolution. The ultimate goal is to implement the performance sensitive parts of the Multi-View_Deconvolution in this library.

