# myFFT
arbitrary size 1D and 2D fft / ifft

This code combines Cooley Tuckey (Radix 2 decimation in time) and Bluestein fft algorithms to take the Fourier (and inverse Fourier) transforms of arbitrary size, 1D and 2D, real and complex arrays.

fft_test.cpp contains 6 examples that demonstrate 1D and 2D fft/ifft applications.

The code contains some optimizations such as unrolling decimation in time loops up to N=128 and using decimation in time algorithm for arbitrary size arrays as many times as possible when the size is divisable by a power of 2

The repository also contains a mex files that can be compiled and called from Matlab. This makes it easy to compare the code outputs with Matlab's fft/ifft outputs. 
