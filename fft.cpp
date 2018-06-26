
/**
* @file fft.cpp
* @author Melih Altun @2015-2018
**/

#include "fft.h"
#define FFT_VERSION 1.06

//Version 1.02: removed all malloc/free calls from cooley-tuckey recursive function. The less dynamic allocation the code has, the faster it operates.
//Version 1.03: Separeted fft_complex into wrapper, core and power of 2, functions. For 2d fft, all memory allocation is done once before row-wise fft and a second time before column-wise fft.
//              Not doing mallocs for each row and column greatly reduces number of allocations and total allocated memory.
//Version 1.04: Added register variables and complexMultiplication, complexReciprocal functions to minimize repeated operations. Increased base recursion case to N=2.
//              Added unrolled loops for N={4,8,16,32,64,128} for speed up.
//Version 1.05: Added base cases up to N=8 for speed up.
//              Removed unrolled loops for N={4,8} since they are covered by base cases now.
//Version 1.06: Moved base cases to fft_core. Instead of using bluestein for all non power of two sizes, this version applies Radix 2 DIT algo as long as N % 2 == 0.
//              It switches to bluestein when this condition no longer holds.

#define UNROLL_CT 128  //unroll Cooley Tuckey half fft combination for loops up to N = 128. Reduce number or undef this to reduce code size.

// Buffers used by Bluestein algorithm.
typedef struct bluestein_buffers_ {
	floating *wr;
	floating *wi;
	floating *yr;
	floating *yi;
	floating *fyr;
	floating *fyi;
	floating *vr;
	floating *vi;
	floating *fvr;
	floating *fvi;
	floating *gr;
	floating *gi;
	floating *fgr;
	floating *fgi;
}bluestein_buffers;

/* Prototypes for internal functions*/
//two fft algorithms are used in this code as internal functions: cooley_tuckey radix 2 and bluestein
static void cooley_tuckey(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize);

static void bluestein(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize);

static void fft_core(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize);

static bool compute_fftSize(size_t N, size_t *fftSize, bluestein_buffers *buffers);

static int init_bluestein_buffers(bluestein_buffers *buffers, size_t N, size_t fftSize);

static int clear_bluestein_buffers(bluestein_buffers *buffers);

static bool power_of_2(size_t N);

static void primeFactorizer(size_t N, int *factors);

static void complexMultiplication(floating z1r, floating z1i, floating z2r, floating z2i, floating *yr, floating *yi);

static void complexReciprocal(floating zr, floating zi, floating *yr, floating *yi);


/* Initializer for fft instances
Parameters: (inputs) fft object and fft size*/
void set_fft_instance(fft_instance *inst, size_t size)
{
	inst->dftSize = size;
	inst->Re = (floating*)calloc(size, sizeof(floating));
	inst->Im = (floating*)calloc(size, sizeof(floating));
#if defined(MAGNITUDE)
	inst->abs = (floating*)calloc(size, sizeof(floating));
#endif
#if defined(PHASE)
	inst->angle = (floating*)calloc(size, sizeof(floating));
#endif
}


/* Same initializer for 2D */
void set_fft2_instance(fft2_instance *inst, size_t N, size_t M)
{
	inst->rows = N;
	inst->cols = M;
	inst->Re = (floating*)calloc(M*N, sizeof(floating));
	inst->Im = (floating*)calloc(M*N, sizeof(floating));
#if defined(MAGNITUDE)
	inst->abs = (floating*)calloc(M*N, sizeof(floating));
#endif
#if defined(PHASE)
	inst->angle = (floating*)calloc(M*N, sizeof(floating));
#endif
}


/*fft for real inputs
calls complex fft with imaginary part of the input set to zero
Parameters: (output) fft object containing fourier transformed input, (inputs) input x as an array of real numbers,
algo cycle counter if input x is a circular buffer, fft size */
void fft_real(fft_instance *inst, floating xr[], int clk, size_t N)
{
	floating *xi;
	xi = (floating*)calloc(N, sizeof(floating));
	fft_complex(inst, xr, xi, clk, N);
	free(xi);
}


/*fft for complex inputs
Parameters: (output) fft object containing fourier transformed input, (inputs) input xr as an array of real numbers,
input xi as an array of imaginary numbers (x = xr + j xi), algo cycle counter if input x is a circular buffer, fft size */
void fft_complex(fft_instance *inst, floating xr[], floating xi[], int clk, size_t N)
{
	bool powerOf2;
	size_t fftSize = N;
	bluestein_buffers buffers;
	floating *xxr, *xxi;
	unsigned int circIndx;

	if (N < 1)
		return;

	xxr = (floating*)malloc(sizeof(floating)* N);
	xxi = (floating*)malloc(sizeof(floating)* N);

	//copy circular buffer to a new one with start index alligned to 0
	circIndx = clk % N;
	if (circIndx == 0) {
		memcpy(xxr, xr, sizeof(floating)*N);
		memcpy(xxi, xi, sizeof(floating)*N);
	}
	else {
		memcpy(xxr, &xr[circIndx], (N - circIndx) * sizeof(floating));
		memcpy(xxi, &xi[circIndx], (N - circIndx) * sizeof(floating));
		memcpy(&xxr[N - circIndx], xr, circIndx * sizeof(floating));
		memcpy(&xxi[N - circIndx], xi, circIndx * sizeof(floating));
	}
	// It is possible to have a non-circular buffer version of this function and eliminate this rotation
	// Still, temporary copies of inputs will be needed since cooley-tuckey function modifies them.

	powerOf2 = compute_fftSize(N, &fftSize, &buffers);

	fft_core(inst->Re, inst->Im, xxr, xxi, &buffers, N, fftSize);  // call fft with adjusted buffers and calculated fft size

	//release dynamic memory
	if (!powerOf2)
		clear_bluestein_buffers(&buffers);

	free(xxr);
	free(xxi);
}


/*inverse fft with real output
calls complex ifft and returns magnitude instead of real + imaginary parts of the output
Parameters: (output) absolute value of x = ifft(X) , (inputs) fft object containing fourier transform X = fft(x),
algo cycle counter if input x is a circular buffer, fft size */
void ifft_real(floating xr[], fft_instance *inst, int clk, size_t N)
{
	register int i;
	floating *xi;
	xi = (floating*)malloc(sizeof(floating)* N);
	ifft_complex(xr, xi, inst, clk, N);
	for (i = 0; i < (int)N; i++)
		xr[i] = sqrt(xr[i] * xr[i] + xi[i] * xi[i]);  //get magnitude from real and imaginary parts
	free(xi);  //release memory
}


/*inverse fft with complex output
returns real + imaginary parts of the output
Parameters: (output) real and imaginary parts of x = ifft(X) , (inputs) fft object containing fourier transform X = fft(x),
algo cycle counter if input x is a circular buffer, fft size */
void ifft_complex(floating xr[], floating xi[], fft_instance *inst, int clk, size_t N)
{
	fft_instance inst2;
	int circIndx;
	register int i;

	if (N < 1)
		return;

	set_fft_instance(&inst2, N);

	fft_complex(&inst2, inst->Im, inst->Re, 0, N);  // ifft(Xr,Xi) = fft(Xi,Xr) / N

	circIndx = clk % N;
	if (circIndx == 0) {
		memcpy(xr, inst2.Im, sizeof(floating)* N);
		memcpy(xi, inst2.Re, sizeof(floating)* N);
	}
	else {	//re-arrange if circular buffer is used
		memcpy(xr, &inst2.Im[N-circIndx], circIndx * sizeof(floating));
		memcpy(xi, &inst2.Re[N-circIndx], circIndx * sizeof(floating));
		memcpy(&xr[circIndx], inst2.Im, (N - circIndx) * sizeof(floating));
		memcpy(&xi[circIndx], inst2.Re, (N - circIndx) * sizeof(floating));
	}

	for (i = 0; i < (int)N; i++) {
		xr[i] /= (floating)N;
		xi[i] /= (floating)N;
	}

	//clear dynamic memory
	delete_fft_instance(&inst2);
}


/*2D fft for real inputs
calls complex fft with imaginary part of the input image set to zero
Parameters: (output) fft object containing fourier transformed image, (inputs) input x as real image reshaped into a 1D array,
image row count, image column count */
void fft2_real(fft2_instance *inst, floating xr[], size_t N, size_t M)
{
	floating *xi;
	xi = (floating*)calloc(N*M, sizeof(floating));
	fft2_complex(inst, xr, xi, N, M);
	free(xi);
}


/*2D fft for complex inputs
Parameters: (output) fft object containing fourier transformed image, (inputs) input xr is the real part of a complex image
input xi is the imaginary part of the complex image such that (x = xr + j xi), image row count, image column count */
void fft2_complex(fft2_instance *inst, floating xr[], floating xi[], size_t N, size_t M)
{
	fft_instance row_fft, col_fft;
	floating *rowVec_r = NULL, *colVec_r = NULL, *rowVec_i = NULL, *colVec_i = NULL, *row_fft_img_r = NULL, *row_fft_img_i = NULL;
	bluestein_buffers buffers;
	bool powerOf2;
	size_t fftSize;
	register int i, j;

	if (N < 1 || M < 1)  //this can lead to a crash
		return;

	set_fft_instance(&row_fft, M);
	set_fft_instance(&col_fft, N);

	rowVec_r = (floating*)malloc(M * sizeof(floating));
	colVec_r = (floating*)malloc(N * sizeof(floating));
	rowVec_i = (floating*)malloc(M * sizeof(floating));
	colVec_i = (floating*)malloc(N * sizeof(floating));

	row_fft_img_r = (floating*)malloc(N * M * sizeof(floating));
	row_fft_img_i = (floating*)malloc(N * M * sizeof(floating));

	//Calculate 2D FFT by row column decomposition
	powerOf2 = compute_fftSize(M, &fftSize, &buffers);

	for (i = 0; i < (int)N; i++) {
		memcpy(rowVec_r, &xr[i*M], M*sizeof(floating));
		memcpy(rowVec_i, &xi[i*M], M*sizeof(floating)); //row-wise for loops replaced by memcpy

		fft_core(row_fft.Re, row_fft.Im, rowVec_r, rowVec_i, &buffers, M, fftSize);

		memcpy(&row_fft_img_r[i*M], row_fft.Re, M*sizeof(floating));
		memcpy(&row_fft_img_i[i*M], row_fft.Im, M*sizeof(floating));
	}
	if (!powerOf2)
		clear_bluestein_buffers(&buffers);

	//row-wise FFT done. column-wise FFT is next
	powerOf2 = compute_fftSize(N, &fftSize, &buffers);

	for (j = 0; j < (int)M; j++) {
		for (i = 0; i < (int)N; i++) {
			colVec_r[i] = row_fft_img_r[lin_index(i, j, M)];  //a pointer can be used later instead of indexing
			colVec_i[i] = row_fft_img_i[lin_index(i, j, M)];  //speed up will not be significant though.
		}
		fft_core(col_fft.Re, col_fft.Im, colVec_r, colVec_i, &buffers, N, fftSize);

		for (i = 0; i < (int)N; i++) {
			inst->Re[lin_index(i, j, M)] = col_fft.Re[i];
			inst->Im[lin_index(i, j, M)] = col_fft.Im[i];
#if defined(MAGNITUDE)
			inst->abs[lin_index(i, j, M)] = col_fft.abs[i];
#endif
#if defined(PHASE)
			inst->angle[lin_index(i, j, M)] = col_fft.angle[i];
#endif
		}
	}
	if (!powerOf2)
		clear_bluestein_buffers(&buffers);

	//2D FFT complete at this point

	//clean up memory
	delete_fft_instance(&row_fft);
	delete_fft_instance(&col_fft);

	free(rowVec_r);
	free(rowVec_i);
	free(colVec_r);
	free(colVec_i);
	free(row_fft_img_r);
	free(row_fft_img_i);
}


/*inverse 2D fft with real output
calls complex ifft and returns magnitude instead of real + imaginary parts of the output
Parameters: (output) absolute value of x = ifft(X) , (inputs) fft image object containing fourier transform X = fft(x), image row count, image col count */
void ifft2_real(floating xr[], fft2_instance *inst, size_t N, size_t M)
{
	register int i;
	floating *xi;
	xi = (floating*)malloc(sizeof(floating)* M * N);
	ifft2_complex(xr, xi, inst, N, M);
	for (i = 0; i < (int)(N*M); i++)
		xr[i] = sqrt(xr[i] * xr[i] + xi[i] * xi[i]);  //get magnitude from real and imaginary parts
	free(xi);  //release memory
}


/*inverse 2D fft with complex output
returns real + imaginary parts of the output
Parameters: (output) real and imaginary parts of x = ifft(X) , (inputs) fft image object containing fourier transform X = fft(x), image row count, image col count */
void ifft2_complex(floating xr[], floating xi[], fft2_instance *inst, size_t N, size_t M)
{
	fft_instance row_fft, col_fft;
	floating *rowVec_r = NULL, *colVec_r = NULL, *rowVec_i = NULL, *colVec_i = NULL, *row_fft_img_r = NULL, *row_fft_img_i = NULL;
	floating Mf = (floating)M, Nf = (floating)N;
	bluestein_buffers buffers;
	bool powerOf2;
	size_t fftSize;
	register int i, j;

	rowVec_r = (floating*)malloc(M * sizeof(floating));
	colVec_r = (floating*)malloc(N * sizeof(floating));
	rowVec_i = (floating*)malloc(M * sizeof(floating));
	colVec_i = (floating*)malloc(N * sizeof(floating));

	row_fft_img_r = (floating*)malloc(N * M * sizeof(floating));
	row_fft_img_i = (floating*)malloc(N * M * sizeof(floating));

	set_fft_instance(&row_fft, M);
	set_fft_instance(&col_fft, N);

	//Calculate 2D iFFT by row column decomposition
	powerOf2 = compute_fftSize(M, &fftSize, &buffers);

	for (i = 0; i < (int)N; i++) {
		memcpy(rowVec_r, &inst->Im[i*M], M*sizeof(floating));
		memcpy(rowVec_i, &inst->Re[i*M], M*sizeof(floating));

		fft_core(row_fft.Re, row_fft.Im, rowVec_r, rowVec_i, &buffers, M, fftSize);

		for (j = 0; j < (int)M; j++) {
			row_fft_img_r[lin_index(i, j, M)] = row_fft.Im[j] / Mf;
			row_fft_img_i[lin_index(i, j, M)] = row_fft.Re[j] / Mf;
		}
	}
	if (!powerOf2)
		clear_bluestein_buffers(&buffers);

	//row-wise iFFT done. column-wise iFFT is next
	powerOf2 = compute_fftSize(N, &fftSize, &buffers);

	for (j = 0; j < (int)M; j++) {
		for (i = 0; i < (int)N; i++) {
			colVec_i[i] = row_fft_img_r[lin_index(i, j, M)];
			colVec_r[i] = row_fft_img_i[lin_index(i, j, M)];
		}
		fft_core(col_fft.Re, col_fft.Im, colVec_r, colVec_i, &buffers, N, fftSize);
		for (i = 0; i < (int)N; i++) {
			xr[lin_index(i, j, M)] = col_fft.Im[i] / Nf;
			xi[lin_index(i, j, M)] = col_fft.Re[i] / Nf;
		}
	}
	if (!powerOf2)
		clear_bluestein_buffers(&buffers);

	//2D iFFT complete at this point

	//clean up memory
	delete_fft_instance(&row_fft);
	delete_fft_instance(&col_fft);

	free(rowVec_r);
	free(rowVec_i);
	free(colVec_r);
	free(colVec_i);
	free(row_fft_img_r);
	free(row_fft_img_i);
}


/* fft instance garbage collection */
void delete_fft_instance(fft_instance *fft)
{
	if (fft->Re != NULL) {
		free(fft->Re);
		fft->Re = NULL;
	}
	if (fft->Im != NULL) {
		free(fft->Im);
		fft->Im = NULL;
	}
#if defined (MAGNITUDE)
	if (fft->abs != NULL) {
		free(fft->abs);
		fft->abs = NULL;
	}
#endif
#if defined (PHASE)
	if (fft->angle != NULL) {
		free(fft->angle);
		fft->angle = NULL;
	}
#endif
}


/* 2D fft instance garbage collection */
void delete_fft2_instance(fft2_instance *fft)
{
	if (fft->Re != NULL) {
		free(fft->Re);
		fft->Re = NULL;
	}
	if (fft->Im != NULL) {
		free(fft->Im);
		fft->Im = NULL;
	}
#if defined (MAGNITUDE)
	if (fft->abs != NULL) {
		free(fft->abs);
		fft->abs = NULL;
	}
#endif
#if defined (PHASE)
	if (fft->angle != NULL) {
		free(fft->angle);
		fft->angle = NULL;
	}
#endif
}


// Internal functions

/* Cooley - Tuckey algorithm. Recursive radix 2 decimation in time. Inputs have to have even number of elements
   Algo time complexity is O(n log n)  */
static void cooley_tuckey(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize)
{
	register int k, N2;
	register floating ti0, tr0, ti1, tr1, arg, arg_k, cosArg, sinArg, cos_xr, sin_xr, cos_xi, sin_xi;

	// X(k) = Sum[n = 0->N-1] x(n) exp(-j 2 pi n k / N )

	// X(k) = Sum[n = 0->N/2-1] x(2*n) exp(-j 4 pi n k / N) + exp(-j 2 pi k / N) Sum[n = 0->N/2-1] x(2*n+1) exp(-j 4 pi n k / N)

	// X(k) = E(k) + exp(-j 2 pi k / N) O(k), where E(k) is the FT of even indices and O(k) is the FT of odd indices

	// E(k + N/2) = E(k) and O(k + N/2) = O(k), since DFT is periodic

	// Furher algebra yields:
	// X(k) =  E(k) + exp(-j 2 pi k / N) O(k)
	// X(k+N/2) =  E(k) - exp(-j 2 pi k / N) O(k), for k 0<= k < N/2

	//https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

		N2 = (int)N / 2;

	for (k = 0; k < N2; k++) {  //separate odd and even indices
		Xr[k] = xr[2 * k];
		Xr[k + N2] = xr[2 * k + 1];
		Xi[k] = xi[2 * k];
		Xi[k + N2] = xi[2 * k + 1];
	}

	//Divide et impera
	fft_core(xr, xi, Xr, Xi, buffers, N2, fftSize);
	fft_core(&xr[N2], &xi[N2], &Xr[N2], &Xi[N2], buffers, N2, fftSize);
	//swap X and x buffers each time and re-use them so that, no additional buffers are created

#if defined (UNROLL_CT)
	//following if blocks contain hard coded twiddle factors for speed up. output will not change if they are removed.
	//this is code space vs. performance trade off. Each removed if block will reduce code size and increase computation time.
#if UNROLL_CT >= 16
	if (N == 16) { //unrolled for loop for size 16 combination
		tr0 = xr[0];
		ti0 = xi[0];
		tr1 = xr[8];
		ti1 = xi[8];

		Xr[0] = tr0 + tr1;
		Xi[0] = ti0 + ti1;
		Xr[8] = tr0 - tr1;
		Xi[8] = ti0 - ti1;

		tr0 = xr[1];
		ti0 = xi[1];
		tr1 = xr[9];
		ti1 = xi[9];

		Xr[1] = tr0 + 0.9238795325112867*tr1 + 0.3826834323650898*ti1;
		Xi[1] = ti0 + 0.9238795325112867*ti1 - 0.3826834323650898*tr1;
		Xr[9] = tr0 - 0.9238795325112867*tr1 - 0.3826834323650898*ti1;
		Xi[9] = ti0 - 0.9238795325112867*ti1 + 0.3826834323650898*tr1;

		tr0 = xr[2];
		ti0 = xi[2];
		tr1 = xr[10];
		ti1 = xi[10];

		Xr[2] = tr0 + 0.7071067811865476*tr1 + 0.7071067811865475*ti1;
		Xi[2] = ti0 + 0.7071067811865476*ti1 - 0.7071067811865475*tr1;
		Xr[10] = tr0 - 0.7071067811865476*tr1 - 0.7071067811865475*ti1;
		Xi[10] = ti0 - 0.7071067811865476*ti1 + 0.7071067811865475*tr1;

		tr0 = xr[3];
		ti0 = xi[3];
		tr1 = xr[11];
		ti1 = xi[11];

		Xr[3] = tr0 + 0.3826834323650898*tr1 + 0.9238795325112867*ti1;
		Xi[3] = ti0 + 0.3826834323650898*ti1 - 0.9238795325112867*tr1;
		Xr[11] = tr0 - 0.3826834323650898*tr1 - 0.9238795325112867*ti1;
		Xi[11] = ti0 - 0.3826834323650898*ti1 + 0.9238795325112867*tr1;

		tr0 = xr[4];
		ti0 = xi[4];
		tr1 = xr[12];
		ti1 = xi[12];

		Xr[4] = tr0 + ti1;
		Xi[4] = ti0 - tr1;
		Xr[12] = tr0 - ti1;
		Xi[12] = ti0 + tr1;

		tr0 = xr[5];
		ti0 = xi[5];
		tr1 = xr[13];
		ti1 = xi[13];

		Xr[5] = tr0 - 0.3826834323650897*tr1 + 0.9238795325112867*ti1;
		Xi[5] = ti0 - 0.3826834323650897*ti1 - 0.9238795325112867*tr1;
		Xr[13] = tr0 + 0.3826834323650897*tr1 - 0.9238795325112867*ti1;
		Xi[13] = ti0 + 0.3826834323650897*ti1 + 0.9238795325112867*tr1;

		tr0 = xr[6];
		ti0 = xi[6];
		tr1 = xr[14];
		ti1 = xi[14];

		Xr[6] = tr0 - 0.7071067811865475*tr1 + 0.7071067811865476*ti1;
		Xi[6] = ti0 - 0.7071067811865475*ti1 - 0.7071067811865476*tr1;
		Xr[14] = tr0 + 0.7071067811865475*tr1 - 0.7071067811865476*ti1;
		Xi[14] = ti0 + 0.7071067811865475*ti1 + 0.7071067811865476*tr1;

		tr0 = xr[7];
		ti0 = xi[7];
		tr1 = xr[15];
		ti1 = xi[15];

		Xr[7] = tr0 - 0.9238795325112867*tr1 + 0.3826834323650899*ti1;
		Xi[7] = ti0 - 0.9238795325112867*ti1 - 0.3826834323650899*tr1;
		Xr[15] = tr0 + 0.9238795325112867*tr1 - 0.3826834323650899*ti1;
		Xi[15] = ti0 + 0.9238795325112867*ti1 + 0.3826834323650899*tr1;
		return;
	}
#endif
#if UNROLL_CT >= 32
	if (N == 32) { //unrolled for loop for size 32 combination
		tr0 = xr[0];
		ti0 = xi[0];
		tr1 = xr[16];
		ti1 = xi[16];

		Xr[0] = tr0 + tr1;
		Xi[0] = ti0 + ti1;
		Xr[16] = tr0 - tr1;
		Xi[16] = ti0 - ti1;

		tr0 = xr[1];
		ti0 = xi[1];
		tr1 = xr[17];
		ti1 = xi[17];

		Xr[1] = tr0 + 0.9807852804032304*tr1 + 0.1950903220161283*ti1;
		Xi[1] = ti0 + 0.9807852804032304*ti1 - 0.1950903220161283*tr1;
		Xr[17] = tr0 - 0.9807852804032304*tr1 - 0.1950903220161283*ti1;
		Xi[17] = ti0 - 0.9807852804032304*ti1 + 0.1950903220161283*tr1;

		tr0 = xr[2];
		ti0 = xi[2];
		tr1 = xr[18];
		ti1 = xi[18];

		Xr[2] = tr0 + 0.9238795325112867*tr1 + 0.3826834323650898*ti1;
		Xi[2] = ti0 + 0.9238795325112867*ti1 - 0.3826834323650898*tr1;
		Xr[18] = tr0 - 0.9238795325112867*tr1 - 0.3826834323650898*ti1;
		Xi[18] = ti0 - 0.9238795325112867*ti1 + 0.3826834323650898*tr1;

		tr0 = xr[3];
		ti0 = xi[3];
		tr1 = xr[19];
		ti1 = xi[19];

		Xr[3] = tr0 + 0.8314696123025452*tr1 + 0.5555702330196022*ti1;
		Xi[3] = ti0 + 0.8314696123025452*ti1 - 0.5555702330196022*tr1;
		Xr[19] = tr0 - 0.8314696123025452*tr1 - 0.5555702330196022*ti1;
		Xi[19] = ti0 - 0.8314696123025452*ti1 + 0.5555702330196022*tr1;

		tr0 = xr[4];
		ti0 = xi[4];
		tr1 = xr[20];
		ti1 = xi[20];

		Xr[4] = tr0 + 0.7071067811865476*tr1 + 0.7071067811865475*ti1;
		Xi[4] = ti0 + 0.7071067811865476*ti1 - 0.7071067811865475*tr1;
		Xr[20] = tr0 - 0.7071067811865476*tr1 - 0.7071067811865475*ti1;
		Xi[20] = ti0 - 0.7071067811865476*ti1 + 0.7071067811865475*tr1;

		tr0 = xr[5];
		ti0 = xi[5];
		tr1 = xr[21];
		ti1 = xi[21];

		Xr[5] = tr0 + 0.5555702330196023*tr1 + 0.8314696123025452*ti1;
		Xi[5] = ti0 + 0.5555702330196023*ti1 - 0.8314696123025452*tr1;
		Xr[21] = tr0 - 0.5555702330196023*tr1 - 0.8314696123025452*ti1;
		Xi[21] = ti0 - 0.5555702330196023*ti1 + 0.8314696123025452*tr1;

		tr0 = xr[6];
		ti0 = xi[6];
		tr1 = xr[22];
		ti1 = xi[22];

		Xr[6] = tr0 + 0.3826834323650898*tr1 + 0.9238795325112867*ti1;
		Xi[6] = ti0 + 0.3826834323650898*ti1 - 0.9238795325112867*tr1;
		Xr[22] = tr0 - 0.3826834323650898*tr1 - 0.9238795325112867*ti1;
		Xi[22] = ti0 - 0.3826834323650898*ti1 + 0.9238795325112867*tr1;

		tr0 = xr[7];
		ti0 = xi[7];
		tr1 = xr[23];
		ti1 = xi[23];

		Xr[7] = tr0 + 0.1950903220161283*tr1 + 0.9807852804032304*ti1;
		Xi[7] = ti0 + 0.1950903220161283*ti1 - 0.9807852804032304*tr1;
		Xr[23] = tr0 - 0.1950903220161283*tr1 - 0.9807852804032304*ti1;
		Xi[23] = ti0 - 0.1950903220161283*ti1 + 0.9807852804032304*tr1;

		tr0 = xr[8];
		ti0 = xi[8];
		tr1 = xr[24];
		ti1 = xi[24];

		Xr[8] = tr0 + ti1;
		Xi[8] = ti0 - tr1;
		Xr[24] = tr0 - ti1;
		Xi[24] = ti0 + tr1;

		tr0 = xr[9];
		ti0 = xi[9];
		tr1 = xr[25];
		ti1 = xi[25];

		Xr[9] = tr0 - 0.1950903220161282*tr1 + 0.9807852804032304*ti1;
		Xi[9] = ti0 - 0.1950903220161282*ti1 - 0.9807852804032304*tr1;
		Xr[25] = tr0 + 0.1950903220161282*tr1 - 0.9807852804032304*ti1;
		Xi[25] = ti0 + 0.1950903220161282*ti1 + 0.9807852804032304*tr1;

		tr0 = xr[10];
		ti0 = xi[10];
		tr1 = xr[26];
		ti1 = xi[26];

		Xr[10] = tr0 - 0.3826834323650897*tr1 + 0.9238795325112867*ti1;
		Xi[10] = ti0 - 0.3826834323650897*ti1 - 0.9238795325112867*tr1;
		Xr[26] = tr0 + 0.3826834323650897*tr1 - 0.9238795325112867*ti1;
		Xi[26] = ti0 + 0.3826834323650897*ti1 + 0.9238795325112867*tr1;

		tr0 = xr[11];
		ti0 = xi[11];
		tr1 = xr[27];
		ti1 = xi[27];

		Xr[11] = tr0 - 0.5555702330196020*tr1 + 0.8314696123025454*ti1;
		Xi[11] = ti0 - 0.5555702330196020*ti1 - 0.8314696123025454*tr1;
		Xr[27] = tr0 + 0.5555702330196020*tr1 - 0.8314696123025454*ti1;
		Xi[27] = ti0 + 0.5555702330196020*ti1 + 0.8314696123025454*tr1;

		tr0 = xr[12];
		ti0 = xi[12];
		tr1 = xr[28];
		ti1 = xi[28];

		Xr[12] = tr0 - 0.7071067811865475*tr1 + 0.7071067811865476*ti1;
		Xi[12] = ti0 - 0.7071067811865475*ti1 - 0.7071067811865476*tr1;
		Xr[28] = tr0 + 0.7071067811865475*tr1 - 0.7071067811865476*ti1;
		Xi[28] = ti0 + 0.7071067811865475*ti1 + 0.7071067811865476*tr1;

		tr0 = xr[13];
		ti0 = xi[13];
		tr1 = xr[29];
		ti1 = xi[29];

		Xr[13] = tr0 - 0.8314696123025454*tr1 + 0.5555702330196022*ti1;
		Xi[13] = ti0 - 0.8314696123025454*ti1 - 0.5555702330196022*tr1;
		Xr[29] = tr0 + 0.8314696123025454*tr1 - 0.5555702330196022*ti1;
		Xi[29] = ti0 + 0.8314696123025454*ti1 + 0.5555702330196022*tr1;

		tr0 = xr[14];
		ti0 = xi[14];
		tr1 = xr[30];
		ti1 = xi[30];

		Xr[14] = tr0 - 0.9238795325112867*tr1 + 0.3826834323650899*ti1;
		Xi[14] = ti0 - 0.9238795325112867*ti1 - 0.3826834323650899*tr1;
		Xr[30] = tr0 + 0.9238795325112867*tr1 - 0.3826834323650899*ti1;
		Xi[30] = ti0 + 0.9238795325112867*ti1 + 0.3826834323650899*tr1;

		tr0 = xr[15];
		ti0 = xi[15];
		tr1 = xr[31];
		ti1 = xi[31];

		Xr[15] = tr0 - 0.9807852804032304*tr1 + 0.1950903220161286*ti1;
		Xi[15] = ti0 - 0.9807852804032304*ti1 - 0.1950903220161286*tr1;
		Xr[31] = tr0 + 0.9807852804032304*tr1 - 0.1950903220161286*ti1;
		Xi[31] = ti0 + 0.9807852804032304*ti1 + 0.1950903220161286*tr1;

		return;
	}
#endif
#if UNROLL_CT >= 64
	if (N == 64) { //unrolled for loop for size 64 combination
		tr0 = xr[0];
		ti0 = xi[0];
		tr1 = xr[32];
		ti1 = xi[32];

		Xr[0] = tr0 + tr1;
		Xi[0] = ti0 + ti1;
		Xr[32] = tr0 - tr1;
		Xi[32] = ti0 - ti1;

		tr0 = xr[1];
		ti0 = xi[1];
		tr1 = xr[33];
		ti1 = xi[33];

		Xr[1] = tr0 + 0.9951847266721969*tr1 + 0.0980171403295606*ti1;
		Xi[1] = ti0 + 0.9951847266721969*ti1 - 0.0980171403295606*tr1;
		Xr[33] = tr0 - 0.9951847266721969*tr1 - 0.0980171403295606*ti1;
		Xi[33] = ti0 - 0.9951847266721969*ti1 + 0.0980171403295606*tr1;

		tr0 = xr[2];
		ti0 = xi[2];
		tr1 = xr[34];
		ti1 = xi[34];

		Xr[2] = tr0 + 0.9807852804032304*tr1 + 0.1950903220161283*ti1;
		Xi[2] = ti0 + 0.9807852804032304*ti1 - 0.1950903220161283*tr1;
		Xr[34] = tr0 - 0.9807852804032304*tr1 - 0.1950903220161283*ti1;
		Xi[34] = ti0 - 0.9807852804032304*ti1 + 0.1950903220161283*tr1;

		tr0 = xr[3];
		ti0 = xi[3];
		tr1 = xr[35];
		ti1 = xi[35];

		Xr[3] = tr0 + 0.9569403357322088*tr1 + 0.2902846772544623*ti1;
		Xi[3] = ti0 + 0.9569403357322088*ti1 - 0.2902846772544623*tr1;
		Xr[35] = tr0 - 0.9569403357322088*tr1 - 0.2902846772544623*ti1;
		Xi[35] = ti0 - 0.9569403357322088*ti1 + 0.2902846772544623*tr1;

		tr0 = xr[4];
		ti0 = xi[4];
		tr1 = xr[36];
		ti1 = xi[36];

		Xr[4] = tr0 + 0.9238795325112867*tr1 + 0.3826834323650898*ti1;
		Xi[4] = ti0 + 0.9238795325112867*ti1 - 0.3826834323650898*tr1;
		Xr[36] = tr0 - 0.9238795325112867*tr1 - 0.3826834323650898*ti1;
		Xi[36] = ti0 - 0.9238795325112867*ti1 + 0.3826834323650898*tr1;

		tr0 = xr[5];
		ti0 = xi[5];
		tr1 = xr[37];
		ti1 = xi[37];

		Xr[5] = tr0 + 0.8819212643483551*tr1 + 0.4713967368259976*ti1;
		Xi[5] = ti0 + 0.8819212643483551*ti1 - 0.4713967368259976*tr1;
		Xr[37] = tr0 - 0.8819212643483551*tr1 - 0.4713967368259976*ti1;
		Xi[37] = ti0 - 0.8819212643483551*ti1 + 0.4713967368259976*tr1;

		tr0 = xr[6];
		ti0 = xi[6];
		tr1 = xr[38];
		ti1 = xi[38];

		Xr[6] = tr0 + 0.8314696123025452*tr1 + 0.5555702330196022*ti1;
		Xi[6] = ti0 + 0.8314696123025452*ti1 - 0.5555702330196022*tr1;
		Xr[38] = tr0 - 0.8314696123025452*tr1 - 0.5555702330196022*ti1;
		Xi[38] = ti0 - 0.8314696123025452*ti1 + 0.5555702330196022*tr1;

		tr0 = xr[7];
		ti0 = xi[7];
		tr1 = xr[39];
		ti1 = xi[39];

		Xr[7] = tr0 + 0.7730104533627370*tr1 + 0.6343932841636455*ti1;
		Xi[7] = ti0 + 0.7730104533627370*ti1 - 0.6343932841636455*tr1;
		Xr[39] = tr0 - 0.7730104533627370*tr1 - 0.6343932841636455*ti1;
		Xi[39] = ti0 - 0.7730104533627370*ti1 + 0.6343932841636455*tr1;

		tr0 = xr[8];
		ti0 = xi[8];
		tr1 = xr[40];
		ti1 = xi[40];

		Xr[8] = tr0 + 0.7071067811865476*tr1 + 0.7071067811865475*ti1;
		Xi[8] = ti0 + 0.7071067811865476*ti1 - 0.7071067811865475*tr1;
		Xr[40] = tr0 - 0.7071067811865476*tr1 - 0.7071067811865475*ti1;
		Xi[40] = ti0 - 0.7071067811865476*ti1 + 0.7071067811865475*tr1;

		tr0 = xr[9];
		ti0 = xi[9];
		tr1 = xr[41];
		ti1 = xi[41];

		Xr[9] = tr0 + 0.6343932841636455*tr1 + 0.7730104533627369*ti1;
		Xi[9] = ti0 + 0.6343932841636455*ti1 - 0.7730104533627369*tr1;
		Xr[41] = tr0 - 0.6343932841636455*tr1 - 0.7730104533627369*ti1;
		Xi[41] = ti0 - 0.6343932841636455*ti1 + 0.7730104533627369*tr1;

		tr0 = xr[10];
		ti0 = xi[10];
		tr1 = xr[42];
		ti1 = xi[42];

		Xr[10] = tr0 + 0.5555702330196023*tr1 + 0.8314696123025452*ti1;
		Xi[10] = ti0 + 0.5555702330196023*ti1 - 0.8314696123025452*tr1;
		Xr[42] = tr0 - 0.5555702330196023*tr1 - 0.8314696123025452*ti1;
		Xi[42] = ti0 - 0.5555702330196023*ti1 + 0.8314696123025452*tr1;

		tr0 = xr[11];
		ti0 = xi[11];
		tr1 = xr[43];
		ti1 = xi[43];

		Xr[11] = tr0 + 0.4713967368259978*tr1 + 0.8819212643483549*ti1;
		Xi[11] = ti0 + 0.4713967368259978*ti1 - 0.8819212643483549*tr1;
		Xr[43] = tr0 - 0.4713967368259978*tr1 - 0.8819212643483549*ti1;
		Xi[43] = ti0 - 0.4713967368259978*ti1 + 0.8819212643483549*tr1;

		tr0 = xr[12];
		ti0 = xi[12];
		tr1 = xr[44];
		ti1 = xi[44];

		Xr[12] = tr0 + 0.3826834323650898*tr1 + 0.9238795325112867*ti1;
		Xi[12] = ti0 + 0.3826834323650898*ti1 - 0.9238795325112867*tr1;
		Xr[44] = tr0 - 0.3826834323650898*tr1 - 0.9238795325112867*ti1;
		Xi[44] = ti0 - 0.3826834323650898*ti1 + 0.9238795325112867*tr1;

		tr0 = xr[13];
		ti0 = xi[13];
		tr1 = xr[45];
		ti1 = xi[45];

		Xr[13] = tr0 + 0.2902846772544623*tr1 + 0.9569403357322089*ti1;
		Xi[13] = ti0 + 0.2902846772544623*ti1 - 0.9569403357322089*tr1;
		Xr[45] = tr0 - 0.2902846772544623*tr1 - 0.9569403357322089*ti1;
		Xi[45] = ti0 - 0.2902846772544623*ti1 + 0.9569403357322089*tr1;

		tr0 = xr[14];
		ti0 = xi[14];
		tr1 = xr[46];
		ti1 = xi[46];

		Xr[14] = tr0 + 0.1950903220161283*tr1 + 0.9807852804032304*ti1;
		Xi[14] = ti0 + 0.1950903220161283*ti1 - 0.9807852804032304*tr1;
		Xr[46] = tr0 - 0.1950903220161283*tr1 - 0.9807852804032304*ti1;
		Xi[46] = ti0 - 0.1950903220161283*ti1 + 0.9807852804032304*tr1;

		tr0 = xr[15];
		ti0 = xi[15];
		tr1 = xr[47];
		ti1 = xi[47];

		Xr[15] = tr0 + 0.0980171403295608*tr1 + 0.9951847266721968*ti1;
		Xi[15] = ti0 + 0.0980171403295608*ti1 - 0.9951847266721968*tr1;
		Xr[47] = tr0 - 0.0980171403295608*tr1 - 0.9951847266721968*ti1;
		Xi[47] = ti0 - 0.0980171403295608*ti1 + 0.9951847266721968*tr1;

		tr0 = xr[16];
		ti0 = xi[16];
		tr1 = xr[48];
		ti1 = xi[48];

		Xr[16] = tr0 + ti1;
		Xi[16] = ti0 - tr1;
		Xr[48] = tr0 - ti1;
		Xi[48] = ti0 + tr1;

		tr0 = xr[17];
		ti0 = xi[17];
		tr1 = xr[49];
		ti1 = xi[49];

		Xr[17] = tr0 - 0.0980171403295606*tr1 + 0.9951847266721969*ti1;
		Xi[17] = ti0 - 0.0980171403295606*ti1 - 0.9951847266721969*tr1;
		Xr[49] = tr0 + 0.0980171403295606*tr1 - 0.9951847266721969*ti1;
		Xi[49] = ti0 + 0.0980171403295606*ti1 + 0.9951847266721969*tr1;

		tr0 = xr[18];
		ti0 = xi[18];
		tr1 = xr[50];
		ti1 = xi[50];

		Xr[18] = tr0 - 0.1950903220161282*tr1 + 0.9807852804032304*ti1;
		Xi[18] = ti0 - 0.1950903220161282*ti1 - 0.9807852804032304*tr1;
		Xr[50] = tr0 + 0.1950903220161282*tr1 - 0.9807852804032304*ti1;
		Xi[50] = ti0 + 0.1950903220161282*ti1 + 0.9807852804032304*tr1;

		tr0 = xr[19];
		ti0 = xi[19];
		tr1 = xr[51];
		ti1 = xi[51];

		Xr[19] = tr0 - 0.2902846772544622*tr1 + 0.9569403357322089*ti1;
		Xi[19] = ti0 - 0.2902846772544622*ti1 - 0.9569403357322089*tr1;
		Xr[51] = tr0 + 0.2902846772544622*tr1 - 0.9569403357322089*ti1;
		Xi[51] = ti0 + 0.2902846772544622*ti1 + 0.9569403357322089*tr1;

		tr0 = xr[20];
		ti0 = xi[20];
		tr1 = xr[52];
		ti1 = xi[52];

		Xr[20] = tr0 - 0.3826834323650897*tr1 + 0.9238795325112867*ti1;
		Xi[20] = ti0 - 0.3826834323650897*ti1 - 0.9238795325112867*tr1;
		Xr[52] = tr0 + 0.3826834323650897*tr1 - 0.9238795325112867*ti1;
		Xi[52] = ti0 + 0.3826834323650897*ti1 + 0.9238795325112867*tr1;

		tr0 = xr[21];
		ti0 = xi[21];
		tr1 = xr[53];
		ti1 = xi[53];

		Xr[21] = tr0 - 0.4713967368259977*tr1 + 0.8819212643483551*ti1;
		Xi[21] = ti0 - 0.4713967368259977*ti1 - 0.8819212643483551*tr1;
		Xr[53] = tr0 + 0.4713967368259977*tr1 - 0.8819212643483551*ti1;
		Xi[53] = ti0 + 0.4713967368259977*ti1 + 0.8819212643483551*tr1;

		tr0 = xr[22];
		ti0 = xi[22];
		tr1 = xr[54];
		ti1 = xi[54];

		Xr[22] = tr0 - 0.5555702330196020*tr1 + 0.8314696123025454*ti1;
		Xi[22] = ti0 - 0.5555702330196020*ti1 - 0.8314696123025454*tr1;
		Xr[54] = tr0 + 0.5555702330196020*tr1 - 0.8314696123025454*ti1;
		Xi[54] = ti0 + 0.5555702330196020*ti1 + 0.8314696123025454*tr1;

		tr0 = xr[23];
		ti0 = xi[23];
		tr1 = xr[55];
		ti1 = xi[55];

		Xr[23] = tr0 - 0.6343932841636454*tr1 + 0.7730104533627371*ti1;
		Xi[23] = ti0 - 0.6343932841636454*ti1 - 0.7730104533627371*tr1;
		Xr[55] = tr0 + 0.6343932841636454*tr1 - 0.7730104533627371*ti1;
		Xi[55] = ti0 + 0.6343932841636454*ti1 + 0.7730104533627371*tr1;

		tr0 = xr[24];
		ti0 = xi[24];
		tr1 = xr[56];
		ti1 = xi[56];

		Xr[24] = tr0 - 0.7071067811865475*tr1 + 0.7071067811865476*ti1;
		Xi[24] = ti0 - 0.7071067811865475*ti1 - 0.7071067811865476*tr1;
		Xr[56] = tr0 + 0.7071067811865475*tr1 - 0.7071067811865476*ti1;
		Xi[56] = ti0 + 0.7071067811865475*ti1 + 0.7071067811865476*tr1;

		tr0 = xr[25];
		ti0 = xi[25];
		tr1 = xr[57];
		ti1 = xi[57];

		Xr[25] = tr0 - 0.7730104533627370*tr1 + 0.6343932841636455*ti1;
		Xi[25] = ti0 - 0.7730104533627370*ti1 - 0.6343932841636455*tr1;
		Xr[57] = tr0 + 0.7730104533627370*tr1 - 0.6343932841636455*ti1;
		Xi[57] = ti0 + 0.7730104533627370*ti1 + 0.6343932841636455*tr1;

		tr0 = xr[26];
		ti0 = xi[26];
		tr1 = xr[58];
		ti1 = xi[58];

		Xr[26] = tr0 - 0.8314696123025454*tr1 + 0.5555702330196022*ti1;
		Xi[26] = ti0 - 0.8314696123025454*ti1 - 0.5555702330196022*tr1;
		Xr[58] = tr0 + 0.8314696123025454*tr1 - 0.5555702330196022*ti1;
		Xi[58] = ti0 + 0.8314696123025454*ti1 + 0.5555702330196022*tr1;

		tr0 = xr[27];
		ti0 = xi[27];
		tr1 = xr[59];
		ti1 = xi[59];

		Xr[27] = tr0 - 0.8819212643483549*tr1 + 0.4713967368259979*ti1;
		Xi[27] = ti0 - 0.8819212643483549*ti1 - 0.4713967368259979*tr1;
		Xr[59] = tr0 + 0.8819212643483549*tr1 - 0.4713967368259979*ti1;
		Xi[59] = ti0 + 0.8819212643483549*ti1 + 0.4713967368259979*tr1;

		tr0 = xr[28];
		ti0 = xi[28];
		tr1 = xr[60];
		ti1 = xi[60];

		Xr[28] = tr0 - 0.9238795325112867*tr1 + 0.3826834323650899*ti1;
		Xi[28] = ti0 - 0.9238795325112867*ti1 - 0.3826834323650899*tr1;
		Xr[60] = tr0 + 0.9238795325112867*tr1 - 0.3826834323650899*ti1;
		Xi[60] = ti0 + 0.9238795325112867*ti1 + 0.3826834323650899*tr1;

		tr0 = xr[29];
		ti0 = xi[29];
		tr1 = xr[61];
		ti1 = xi[61];

		Xr[29] = tr0 - 0.9569403357322088*tr1 + 0.2902846772544624*ti1;
		Xi[29] = ti0 - 0.9569403357322088*ti1 - 0.2902846772544624*tr1;
		Xr[61] = tr0 + 0.9569403357322088*tr1 - 0.2902846772544624*ti1;
		Xi[61] = ti0 + 0.9569403357322088*ti1 + 0.2902846772544624*tr1;

		tr0 = xr[30];
		ti0 = xi[30];
		tr1 = xr[62];
		ti1 = xi[62];

		Xr[30] = tr0 - 0.9807852804032304*tr1 + 0.1950903220161286*ti1;
		Xi[30] = ti0 - 0.9807852804032304*ti1 - 0.1950903220161286*tr1;
		Xr[62] = tr0 + 0.9807852804032304*tr1 - 0.1950903220161286*ti1;
		Xi[62] = ti0 + 0.9807852804032304*ti1 + 0.1950903220161286*tr1;

		tr0 = xr[31];
		ti0 = xi[31];
		tr1 = xr[63];
		ti1 = xi[63];

		Xr[31] = tr0 - 0.9951847266721968*tr1 + 0.0980171403295608*ti1;
		Xi[31] = ti0 - 0.9951847266721968*ti1 - 0.0980171403295608*tr1;
		Xr[63] = tr0 + 0.9951847266721968*tr1 - 0.0980171403295608*ti1;
		Xi[63] = ti0 + 0.9951847266721968*ti1 + 0.0980171403295608*tr1;
		return;
	}
#endif
#if UNROLL_CT >= 128
	if (N == 128) {  //unrolled for loop for size 128 combination
		tr0 = xr[0];
		ti0 = xi[0];
		tr1 = xr[64];
		ti1 = xi[64];

		Xr[0] = tr0 + tr1;
		Xi[0] = ti0 + ti1;
		Xr[64] = tr0 - tr1;
		Xi[64] = ti0 - ti1;

		tr0 = xr[1];
		ti0 = xi[1];
		tr1 = xr[65];
		ti1 = xi[65];

		Xr[1] = tr0 + 0.9987954562051724*tr1 + 0.0490676743274180*ti1;
		Xi[1] = ti0 + 0.9987954562051724*ti1 - 0.0490676743274180*tr1;
		Xr[65] = tr0 - 0.9987954562051724*tr1 - 0.0490676743274180*ti1;
		Xi[65] = ti0 - 0.9987954562051724*ti1 + 0.0490676743274180*tr1;

		tr0 = xr[2];
		ti0 = xi[2];
		tr1 = xr[66];
		ti1 = xi[66];

		Xr[2] = tr0 + 0.9951847266721969*tr1 + 0.0980171403295606*ti1;
		Xi[2] = ti0 + 0.9951847266721969*ti1 - 0.0980171403295606*tr1;
		Xr[66] = tr0 - 0.9951847266721969*tr1 - 0.0980171403295606*ti1;
		Xi[66] = ti0 - 0.9951847266721969*ti1 + 0.0980171403295606*tr1;

		tr0 = xr[3];
		ti0 = xi[3];
		tr1 = xr[67];
		ti1 = xi[67];

		Xr[3] = tr0 + 0.9891765099647810*tr1 + 0.1467304744553618*ti1;
		Xi[3] = ti0 + 0.9891765099647810*ti1 - 0.1467304744553618*tr1;
		Xr[67] = tr0 - 0.9891765099647810*tr1 - 0.1467304744553618*ti1;
		Xi[67] = ti0 - 0.9891765099647810*ti1 + 0.1467304744553618*tr1;

		tr0 = xr[4];
		ti0 = xi[4];
		tr1 = xr[68];
		ti1 = xi[68];

		Xr[4] = tr0 + 0.9807852804032304*tr1 + 0.1950903220161283*ti1;
		Xi[4] = ti0 + 0.9807852804032304*ti1 - 0.1950903220161283*tr1;
		Xr[68] = tr0 - 0.9807852804032304*tr1 - 0.1950903220161283*ti1;
		Xi[68] = ti0 - 0.9807852804032304*ti1 + 0.1950903220161283*tr1;

		tr0 = xr[5];
		ti0 = xi[5];
		tr1 = xr[69];
		ti1 = xi[69];

		Xr[5] = tr0 + 0.9700312531945440*tr1 + 0.2429801799032639*ti1;
		Xi[5] = ti0 + 0.9700312531945440*ti1 - 0.2429801799032639*tr1;
		Xr[69] = tr0 - 0.9700312531945440*tr1 - 0.2429801799032639*ti1;
		Xi[69] = ti0 - 0.9700312531945440*ti1 + 0.2429801799032639*tr1;

		tr0 = xr[6];
		ti0 = xi[6];
		tr1 = xr[70];
		ti1 = xi[70];

		Xr[6] = tr0 + 0.9569403357322088*tr1 + 0.2902846772544623*ti1;
		Xi[6] = ti0 + 0.9569403357322088*ti1 - 0.2902846772544623*tr1;
		Xr[70] = tr0 - 0.9569403357322088*tr1 - 0.2902846772544623*ti1;
		Xi[70] = ti0 - 0.9569403357322088*ti1 + 0.2902846772544623*tr1;

		tr0 = xr[7];
		ti0 = xi[7];
		tr1 = xr[71];
		ti1 = xi[71];

		Xr[7] = tr0 + 0.9415440651830208*tr1 + 0.3368898533922201*ti1;
		Xi[7] = ti0 + 0.9415440651830208*ti1 - 0.3368898533922201*tr1;
		Xr[71] = tr0 - 0.9415440651830208*tr1 - 0.3368898533922201*ti1;
		Xi[71] = ti0 - 0.9415440651830208*ti1 + 0.3368898533922201*tr1;

		tr0 = xr[8];
		ti0 = xi[8];
		tr1 = xr[72];
		ti1 = xi[72];

		Xr[8] = tr0 + 0.9238795325112867*tr1 + 0.3826834323650898*ti1;
		Xi[8] = ti0 + 0.9238795325112867*ti1 - 0.3826834323650898*tr1;
		Xr[72] = tr0 - 0.9238795325112867*tr1 - 0.3826834323650898*ti1;
		Xi[72] = ti0 - 0.9238795325112867*ti1 + 0.3826834323650898*tr1;

		tr0 = xr[9];
		ti0 = xi[9];
		tr1 = xr[73];
		ti1 = xi[73];

		Xr[9] = tr0 + 0.9039892931234433*tr1 + 0.4275550934302821*ti1;
		Xi[9] = ti0 + 0.9039892931234433*ti1 - 0.4275550934302821*tr1;
		Xr[73] = tr0 - 0.9039892931234433*tr1 - 0.4275550934302821*ti1;
		Xi[73] = ti0 - 0.9039892931234433*ti1 + 0.4275550934302821*tr1;

		tr0 = xr[10];
		ti0 = xi[10];
		tr1 = xr[74];
		ti1 = xi[74];

		Xr[10] = tr0 + 0.8819212643483551*tr1 + 0.4713967368259976*ti1;
		Xi[10] = ti0 + 0.8819212643483551*ti1 - 0.4713967368259976*tr1;
		Xr[74] = tr0 - 0.8819212643483551*tr1 - 0.4713967368259976*ti1;
		Xi[74] = ti0 - 0.8819212643483551*ti1 + 0.4713967368259976*tr1;

		tr0 = xr[11];
		ti0 = xi[11];
		tr1 = xr[75];
		ti1 = xi[75];

		Xr[11] = tr0 + 0.8577286100002721*tr1 + 0.5141027441932217*ti1;
		Xi[11] = ti0 + 0.8577286100002721*ti1 - 0.5141027441932217*tr1;
		Xr[75] = tr0 - 0.8577286100002721*tr1 - 0.5141027441932217*ti1;
		Xi[75] = ti0 - 0.8577286100002721*ti1 + 0.5141027441932217*tr1;

		tr0 = xr[12];
		ti0 = xi[12];
		tr1 = xr[76];
		ti1 = xi[76];

		Xr[12] = tr0 + 0.8314696123025452*tr1 + 0.5555702330196022*ti1;
		Xi[12] = ti0 + 0.8314696123025452*ti1 - 0.5555702330196022*tr1;
		Xr[76] = tr0 - 0.8314696123025452*tr1 - 0.5555702330196022*ti1;
		Xi[76] = ti0 - 0.8314696123025452*ti1 + 0.5555702330196022*tr1;

		tr0 = xr[13];
		ti0 = xi[13];
		tr1 = xr[77];
		ti1 = xi[77];

		Xr[13] = tr0 + 0.8032075314806449*tr1 + 0.5956993044924334*ti1;
		Xi[13] = ti0 + 0.8032075314806449*ti1 - 0.5956993044924334*tr1;
		Xr[77] = tr0 - 0.8032075314806449*tr1 - 0.5956993044924334*ti1;
		Xi[77] = ti0 - 0.8032075314806449*ti1 + 0.5956993044924334*tr1;

		tr0 = xr[14];
		ti0 = xi[14];
		tr1 = xr[78];
		ti1 = xi[78];

		Xr[14] = tr0 + 0.7730104533627370*tr1 + 0.6343932841636455*ti1;
		Xi[14] = ti0 + 0.7730104533627370*ti1 - 0.6343932841636455*tr1;
		Xr[78] = tr0 - 0.7730104533627370*tr1 - 0.6343932841636455*ti1;
		Xi[78] = ti0 - 0.7730104533627370*ti1 + 0.6343932841636455*tr1;

		tr0 = xr[15];
		ti0 = xi[15];
		tr1 = xr[79];
		ti1 = xi[79];

		Xr[15] = tr0 + 0.7409511253549592*tr1 + 0.6715589548470183*ti1;
		Xi[15] = ti0 + 0.7409511253549592*ti1 - 0.6715589548470183*tr1;
		Xr[79] = tr0 - 0.7409511253549592*tr1 - 0.6715589548470183*ti1;
		Xi[79] = ti0 - 0.7409511253549592*ti1 + 0.6715589548470183*tr1;

		tr0 = xr[16];
		ti0 = xi[16];
		tr1 = xr[80];
		ti1 = xi[80];

		Xr[16] = tr0 + 0.7071067811865476*tr1 + 0.7071067811865475*ti1;
		Xi[16] = ti0 + 0.7071067811865476*ti1 - 0.7071067811865475*tr1;
		Xr[80] = tr0 - 0.7071067811865476*tr1 - 0.7071067811865475*ti1;
		Xi[80] = ti0 - 0.7071067811865476*ti1 + 0.7071067811865475*tr1;

		tr0 = xr[17];
		ti0 = xi[17];
		tr1 = xr[81];
		ti1 = xi[81];

		Xr[17] = tr0 + 0.6715589548470184*tr1 + 0.7409511253549591*ti1;
		Xi[17] = ti0 + 0.6715589548470184*ti1 - 0.7409511253549591*tr1;
		Xr[81] = tr0 - 0.6715589548470184*tr1 - 0.7409511253549591*ti1;
		Xi[81] = ti0 - 0.6715589548470184*ti1 + 0.7409511253549591*tr1;

		tr0 = xr[18];
		ti0 = xi[18];
		tr1 = xr[82];
		ti1 = xi[82];

		Xr[18] = tr0 + 0.6343932841636455*tr1 + 0.7730104533627369*ti1;
		Xi[18] = ti0 + 0.6343932841636455*ti1 - 0.7730104533627369*tr1;
		Xr[82] = tr0 - 0.6343932841636455*tr1 - 0.7730104533627369*ti1;
		Xi[82] = ti0 - 0.6343932841636455*ti1 + 0.7730104533627369*tr1;

		tr0 = xr[19];
		ti0 = xi[19];
		tr1 = xr[83];
		ti1 = xi[83];

		Xr[19] = tr0 + 0.5956993044924335*tr1 + 0.8032075314806448*ti1;
		Xi[19] = ti0 + 0.5956993044924335*ti1 - 0.8032075314806448*tr1;
		Xr[83] = tr0 - 0.5956993044924335*tr1 - 0.8032075314806448*ti1;
		Xi[83] = ti0 - 0.5956993044924335*ti1 + 0.8032075314806448*tr1;

		tr0 = xr[20];
		ti0 = xi[20];
		tr1 = xr[84];
		ti1 = xi[84];

		Xr[20] = tr0 + 0.5555702330196023*tr1 + 0.8314696123025452*ti1;
		Xi[20] = ti0 + 0.5555702330196023*ti1 - 0.8314696123025452*tr1;
		Xr[84] = tr0 - 0.5555702330196023*tr1 - 0.8314696123025452*ti1;
		Xi[84] = ti0 - 0.5555702330196023*ti1 + 0.8314696123025452*tr1;

		tr0 = xr[21];
		ti0 = xi[21];
		tr1 = xr[85];
		ti1 = xi[85];

		Xr[21] = tr0 + 0.5141027441932217*tr1 + 0.8577286100002721*ti1;
		Xi[21] = ti0 + 0.5141027441932217*ti1 - 0.8577286100002721*tr1;
		Xr[85] = tr0 - 0.5141027441932217*tr1 - 0.8577286100002721*ti1;
		Xi[85] = ti0 - 0.5141027441932217*ti1 + 0.8577286100002721*tr1;

		tr0 = xr[22];
		ti0 = xi[22];
		tr1 = xr[86];
		ti1 = xi[86];

		Xr[22] = tr0 + 0.4713967368259978*tr1 + 0.8819212643483549*ti1;
		Xi[22] = ti0 + 0.4713967368259978*ti1 - 0.8819212643483549*tr1;
		Xr[86] = tr0 - 0.4713967368259978*tr1 - 0.8819212643483549*ti1;
		Xi[86] = ti0 - 0.4713967368259978*ti1 + 0.8819212643483549*tr1;

		tr0 = xr[23];
		ti0 = xi[23];
		tr1 = xr[87];
		ti1 = xi[87];

		Xr[23] = tr0 + 0.4275550934302822*tr1 + 0.9039892931234433*ti1;
		Xi[23] = ti0 + 0.4275550934302822*ti1 - 0.9039892931234433*tr1;
		Xr[87] = tr0 - 0.4275550934302822*tr1 - 0.9039892931234433*ti1;
		Xi[87] = ti0 - 0.4275550934302822*ti1 + 0.9039892931234433*tr1;

		tr0 = xr[24];
		ti0 = xi[24];
		tr1 = xr[88];
		ti1 = xi[88];

		Xr[24] = tr0 + 0.3826834323650898*tr1 + 0.9238795325112867*ti1;
		Xi[24] = ti0 + 0.3826834323650898*ti1 - 0.9238795325112867*tr1;
		Xr[88] = tr0 - 0.3826834323650898*tr1 - 0.9238795325112867*ti1;
		Xi[88] = ti0 - 0.3826834323650898*ti1 + 0.9238795325112867*tr1;

		tr0 = xr[25];
		ti0 = xi[25];
		tr1 = xr[89];
		ti1 = xi[89];

		Xr[25] = tr0 + 0.3368898533922201*tr1 + 0.9415440651830208*ti1;
		Xi[25] = ti0 + 0.3368898533922201*ti1 - 0.9415440651830208*tr1;
		Xr[89] = tr0 - 0.3368898533922201*tr1 - 0.9415440651830208*ti1;
		Xi[89] = ti0 - 0.3368898533922201*ti1 + 0.9415440651830208*tr1;

		tr0 = xr[26];
		ti0 = xi[26];
		tr1 = xr[90];
		ti1 = xi[90];

		Xr[26] = tr0 + 0.2902846772544623*tr1 + 0.9569403357322089*ti1;
		Xi[26] = ti0 + 0.2902846772544623*ti1 - 0.9569403357322089*tr1;
		Xr[90] = tr0 - 0.2902846772544623*tr1 - 0.9569403357322089*ti1;
		Xi[90] = ti0 - 0.2902846772544623*ti1 + 0.9569403357322089*tr1;

		tr0 = xr[27];
		ti0 = xi[27];
		tr1 = xr[91];
		ti1 = xi[91];

		Xr[27] = tr0 + 0.2429801799032640*tr1 + 0.9700312531945440*ti1;
		Xi[27] = ti0 + 0.2429801799032640*ti1 - 0.9700312531945440*tr1;
		Xr[91] = tr0 - 0.2429801799032640*tr1 - 0.9700312531945440*ti1;
		Xi[91] = ti0 - 0.2429801799032640*ti1 + 0.9700312531945440*tr1;

		tr0 = xr[28];
		ti0 = xi[28];
		tr1 = xr[92];
		ti1 = xi[92];

		Xr[28] = tr0 + 0.1950903220161283*tr1 + 0.9807852804032304*ti1;
		Xi[28] = ti0 + 0.1950903220161283*ti1 - 0.9807852804032304*tr1;
		Xr[92] = tr0 - 0.1950903220161283*tr1 - 0.9807852804032304*ti1;
		Xi[92] = ti0 - 0.1950903220161283*ti1 + 0.9807852804032304*tr1;

		tr0 = xr[29];
		ti0 = xi[29];
		tr1 = xr[93];
		ti1 = xi[93];

		Xr[29] = tr0 + 0.1467304744553618*tr1 + 0.9891765099647810*ti1;
		Xi[29] = ti0 + 0.1467304744553618*ti1 - 0.9891765099647810*tr1;
		Xr[93] = tr0 - 0.1467304744553618*tr1 - 0.9891765099647810*ti1;
		Xi[93] = ti0 - 0.1467304744553618*ti1 + 0.9891765099647810*tr1;

		tr0 = xr[30];
		ti0 = xi[30];
		tr1 = xr[94];
		ti1 = xi[94];

		Xr[30] = tr0 + 0.0980171403295608*tr1 + 0.9951847266721968*ti1;
		Xi[30] = ti0 + 0.0980171403295608*ti1 - 0.9951847266721968*tr1;
		Xr[94] = tr0 - 0.0980171403295608*tr1 - 0.9951847266721968*ti1;
		Xi[94] = ti0 - 0.0980171403295608*ti1 + 0.9951847266721968*tr1;

		tr0 = xr[31];
		ti0 = xi[31];
		tr1 = xr[95];
		ti1 = xi[95];

		Xr[31] = tr0 + 0.0490676743274181*tr1 + 0.9987954562051724*ti1;
		Xi[31] = ti0 + 0.0490676743274181*ti1 - 0.9987954562051724*tr1;
		Xr[95] = tr0 - 0.0490676743274181*tr1 - 0.9987954562051724*ti1;
		Xi[95] = ti0 - 0.0490676743274181*ti1 + 0.9987954562051724*tr1;

		tr0 = xr[32];
		ti0 = xi[32];
		tr1 = xr[96];
		ti1 = xi[96];

		Xr[32] = tr0 + ti1;
		Xi[32] = ti0 - tr1;
		Xr[96] = tr0 - ti1;
		Xi[96] = ti0 + tr1;

		tr0 = xr[33];
		ti0 = xi[33];
		tr1 = xr[97];
		ti1 = xi[97];

		Xr[33] = tr0 - 0.0490676743274180*tr1 + 0.9987954562051724*ti1;
		Xi[33] = ti0 - 0.0490676743274180*ti1 - 0.9987954562051724*tr1;
		Xr[97] = tr0 + 0.0490676743274180*tr1 - 0.9987954562051724*ti1;
		Xi[97] = ti0 + 0.0490676743274180*ti1 + 0.9987954562051724*tr1;

		tr0 = xr[34];
		ti0 = xi[34];
		tr1 = xr[98];
		ti1 = xi[98];

		Xr[34] = tr0 - 0.0980171403295606*tr1 + 0.9951847266721969*ti1;
		Xi[34] = ti0 - 0.0980171403295606*ti1 - 0.9951847266721969*tr1;
		Xr[98] = tr0 + 0.0980171403295606*tr1 - 0.9951847266721969*ti1;
		Xi[98] = ti0 + 0.0980171403295606*ti1 + 0.9951847266721969*tr1;

		tr0 = xr[35];
		ti0 = xi[35];
		tr1 = xr[99];
		ti1 = xi[99];

		Xr[35] = tr0 - 0.1467304744553616*tr1 + 0.9891765099647810*ti1;
		Xi[35] = ti0 - 0.1467304744553616*ti1 - 0.9891765099647810*tr1;
		Xr[99] = tr0 + 0.1467304744553616*tr1 - 0.9891765099647810*ti1;
		Xi[99] = ti0 + 0.1467304744553616*ti1 + 0.9891765099647810*tr1;

		tr0 = xr[36];
		ti0 = xi[36];
		tr1 = xr[100];
		ti1 = xi[100];

		Xr[36] = tr0 - 0.1950903220161282*tr1 + 0.9807852804032304*ti1;
		Xi[36] = ti0 - 0.1950903220161282*ti1 - 0.9807852804032304*tr1;
		Xr[100] = tr0 + 0.1950903220161282*tr1 - 0.9807852804032304*ti1;
		Xi[100] = ti0 + 0.1950903220161282*ti1 + 0.9807852804032304*tr1;

		tr0 = xr[37];
		ti0 = xi[37];
		tr1 = xr[101];
		ti1 = xi[101];

		Xr[37] = tr0 - 0.2429801799032639*tr1 + 0.9700312531945440*ti1;
		Xi[37] = ti0 - 0.2429801799032639*ti1 - 0.9700312531945440*tr1;
		Xr[101] = tr0 + 0.2429801799032639*tr1 - 0.9700312531945440*ti1;
		Xi[101] = ti0 + 0.2429801799032639*ti1 + 0.9700312531945440*tr1;

		tr0 = xr[38];
		ti0 = xi[38];
		tr1 = xr[102];
		ti1 = xi[102];

		Xr[38] = tr0 - 0.2902846772544622*tr1 + 0.9569403357322089*ti1;
		Xi[38] = ti0 - 0.2902846772544622*ti1 - 0.9569403357322089*tr1;
		Xr[102] = tr0 + 0.2902846772544622*tr1 - 0.9569403357322089*ti1;
		Xi[102] = ti0 + 0.2902846772544622*ti1 + 0.9569403357322089*tr1;

		tr0 = xr[39];
		ti0 = xi[39];
		tr1 = xr[103];
		ti1 = xi[103];

		Xr[39] = tr0 - 0.3368898533922199*tr1 + 0.9415440651830208*ti1;
		Xi[39] = ti0 - 0.3368898533922199*ti1 - 0.9415440651830208*tr1;
		Xr[103] = tr0 + 0.3368898533922199*tr1 - 0.9415440651830208*ti1;
		Xi[103] = ti0 + 0.3368898533922199*ti1 + 0.9415440651830208*tr1;

		tr0 = xr[40];
		ti0 = xi[40];
		tr1 = xr[104];
		ti1 = xi[104];

		Xr[40] = tr0 - 0.3826834323650897*tr1 + 0.9238795325112867*ti1;
		Xi[40] = ti0 - 0.3826834323650897*ti1 - 0.9238795325112867*tr1;
		Xr[104] = tr0 + 0.3826834323650897*tr1 - 0.9238795325112867*ti1;
		Xi[104] = ti0 + 0.3826834323650897*ti1 + 0.9238795325112867*tr1;

		tr0 = xr[41];
		ti0 = xi[41];
		tr1 = xr[105];
		ti1 = xi[105];

		Xr[41] = tr0 - 0.4275550934302819*tr1 + 0.9039892931234435*ti1;
		Xi[41] = ti0 - 0.4275550934302819*ti1 - 0.9039892931234435*tr1;
		Xr[105] = tr0 + 0.4275550934302819*tr1 - 0.9039892931234435*ti1;
		Xi[105] = ti0 + 0.4275550934302819*ti1 + 0.9039892931234435*tr1;

		tr0 = xr[42];
		ti0 = xi[42];
		tr1 = xr[106];
		ti1 = xi[106];

		Xr[42] = tr0 - 0.4713967368259977*tr1 + 0.8819212643483551*ti1;
		Xi[42] = ti0 - 0.4713967368259977*ti1 - 0.8819212643483551*tr1;
		Xr[106] = tr0 + 0.4713967368259977*tr1 - 0.8819212643483551*ti1;
		Xi[106] = ti0 + 0.4713967368259977*ti1 + 0.8819212643483551*tr1;

		tr0 = xr[43];
		ti0 = xi[43];
		tr1 = xr[107];
		ti1 = xi[107];

		Xr[43] = tr0 - 0.5141027441932216*tr1 + 0.8577286100002721*ti1;
		Xi[43] = ti0 - 0.5141027441932216*ti1 - 0.8577286100002721*tr1;
		Xr[107] = tr0 + 0.5141027441932216*tr1 - 0.8577286100002721*ti1;
		Xi[107] = ti0 + 0.5141027441932216*ti1 + 0.8577286100002721*tr1;

		tr0 = xr[44];
		ti0 = xi[44];
		tr1 = xr[108];
		ti1 = xi[108];

		Xr[44] = tr0 - 0.5555702330196020*tr1 + 0.8314696123025454*ti1;
		Xi[44] = ti0 - 0.5555702330196020*ti1 - 0.8314696123025454*tr1;
		Xr[108] = tr0 + 0.5555702330196020*tr1 - 0.8314696123025454*ti1;
		Xi[108] = ti0 + 0.5555702330196020*ti1 + 0.8314696123025454*tr1;

		tr0 = xr[45];
		ti0 = xi[45];
		tr1 = xr[109];
		ti1 = xi[109];

		Xr[45] = tr0 - 0.5956993044924334*tr1 + 0.8032075314806449*ti1;
		Xi[45] = ti0 - 0.5956993044924334*ti1 - 0.8032075314806449*tr1;
		Xr[109] = tr0 + 0.5956993044924334*tr1 - 0.8032075314806449*ti1;
		Xi[109] = ti0 + 0.5956993044924334*ti1 + 0.8032075314806449*tr1;

		tr0 = xr[46];
		ti0 = xi[46];
		tr1 = xr[110];
		ti1 = xi[110];

		Xr[46] = tr0 - 0.6343932841636454*tr1 + 0.7730104533627371*ti1;
		Xi[46] = ti0 - 0.6343932841636454*ti1 - 0.7730104533627371*tr1;
		Xr[110] = tr0 + 0.6343932841636454*tr1 - 0.7730104533627371*ti1;
		Xi[110] = ti0 + 0.6343932841636454*ti1 + 0.7730104533627371*tr1;

		tr0 = xr[47];
		ti0 = xi[47];
		tr1 = xr[111];
		ti1 = xi[111];

		Xr[47] = tr0 - 0.6715589548470184*tr1 + 0.7409511253549590*ti1;
		Xi[47] = ti0 - 0.6715589548470184*ti1 - 0.7409511253549590*tr1;
		Xr[111] = tr0 + 0.6715589548470184*tr1 - 0.7409511253549590*ti1;
		Xi[111] = ti0 + 0.6715589548470184*ti1 + 0.7409511253549590*tr1;

		tr0 = xr[48];
		ti0 = xi[48];
		tr1 = xr[112];
		ti1 = xi[112];

		Xr[48] = tr0 - 0.7071067811865475*tr1 + 0.7071067811865476*ti1;
		Xi[48] = ti0 - 0.7071067811865475*ti1 - 0.7071067811865476*tr1;
		Xr[112] = tr0 + 0.7071067811865475*tr1 - 0.7071067811865476*ti1;
		Xi[112] = ti0 + 0.7071067811865475*ti1 + 0.7071067811865476*tr1;

		tr0 = xr[49];
		ti0 = xi[49];
		tr1 = xr[113];
		ti1 = xi[113];

		Xr[49] = tr0 - 0.7409511253549589*tr1 + 0.6715589548470186*ti1;
		Xi[49] = ti0 - 0.7409511253549589*ti1 - 0.6715589548470186*tr1;
		Xr[113] = tr0 + 0.7409511253549589*tr1 - 0.6715589548470186*ti1;
		Xi[113] = ti0 + 0.7409511253549589*ti1 + 0.6715589548470186*tr1;

		tr0 = xr[50];
		ti0 = xi[50];
		tr1 = xr[114];
		ti1 = xi[114];

		Xr[50] = tr0 - 0.7730104533627370*tr1 + 0.6343932841636455*ti1;
		Xi[50] = ti0 - 0.7730104533627370*ti1 - 0.6343932841636455*tr1;
		Xr[114] = tr0 + 0.7730104533627370*tr1 - 0.6343932841636455*ti1;
		Xi[114] = ti0 + 0.7730104533627370*ti1 + 0.6343932841636455*tr1;

		tr0 = xr[51];
		ti0 = xi[51];
		tr1 = xr[115];
		ti1 = xi[115];

		Xr[51] = tr0 - 0.8032075314806448*tr1 + 0.5956993044924335*ti1;
		Xi[51] = ti0 - 0.8032075314806448*ti1 - 0.5956993044924335*tr1;
		Xr[115] = tr0 + 0.8032075314806448*tr1 - 0.5956993044924335*ti1;
		Xi[115] = ti0 + 0.8032075314806448*ti1 + 0.5956993044924335*tr1;

		tr0 = xr[52];
		ti0 = xi[52];
		tr1 = xr[116];
		ti1 = xi[116];

		Xr[52] = tr0 - 0.8314696123025454*tr1 + 0.5555702330196022*ti1;
		Xi[52] = ti0 - 0.8314696123025454*ti1 - 0.5555702330196022*tr1;
		Xr[116] = tr0 + 0.8314696123025454*tr1 - 0.5555702330196022*ti1;
		Xi[116] = ti0 + 0.8314696123025454*ti1 + 0.5555702330196022*tr1;

		tr0 = xr[53];
		ti0 = xi[53];
		tr1 = xr[117];
		ti1 = xi[117];

		Xr[53] = tr0 - 0.8577286100002720*tr1 + 0.5141027441932218*ti1;
		Xi[53] = ti0 - 0.8577286100002720*ti1 - 0.5141027441932218*tr1;
		Xr[117] = tr0 + 0.8577286100002720*tr1 - 0.5141027441932218*ti1;
		Xi[117] = ti0 + 0.8577286100002720*ti1 + 0.5141027441932218*tr1;

		tr0 = xr[54];
		ti0 = xi[54];
		tr1 = xr[118];
		ti1 = xi[118];

		Xr[54] = tr0 - 0.8819212643483549*tr1 + 0.4713967368259979*ti1;
		Xi[54] = ti0 - 0.8819212643483549*ti1 - 0.4713967368259979*tr1;
		Xr[118] = tr0 + 0.8819212643483549*tr1 - 0.4713967368259979*ti1;
		Xi[118] = ti0 + 0.8819212643483549*ti1 + 0.4713967368259979*tr1;

		tr0 = xr[55];
		ti0 = xi[55];
		tr1 = xr[119];
		ti1 = xi[119];

		Xr[55] = tr0 - 0.9039892931234433*tr1 + 0.4275550934302820*ti1;
		Xi[55] = ti0 - 0.9039892931234433*ti1 - 0.4275550934302820*tr1;
		Xr[119] = tr0 + 0.9039892931234433*tr1 - 0.4275550934302820*ti1;
		Xi[119] = ti0 + 0.9039892931234433*ti1 + 0.4275550934302820*tr1;

		tr0 = xr[56];
		ti0 = xi[56];
		tr1 = xr[120];
		ti1 = xi[120];

		Xr[56] = tr0 - 0.9238795325112867*tr1 + 0.3826834323650899*ti1;
		Xi[56] = ti0 - 0.9238795325112867*ti1 - 0.3826834323650899*tr1;
		Xr[120] = tr0 + 0.9238795325112867*tr1 - 0.3826834323650899*ti1;
		Xi[120] = ti0 + 0.9238795325112867*ti1 + 0.3826834323650899*tr1;

		tr0 = xr[57];
		ti0 = xi[57];
		tr1 = xr[121];
		ti1 = xi[121];

		Xr[57] = tr0 - 0.9415440651830207*tr1 + 0.3368898533922203*ti1;
		Xi[57] = ti0 - 0.9415440651830207*ti1 - 0.3368898533922203*tr1;
		Xr[121] = tr0 + 0.9415440651830207*tr1 - 0.3368898533922203*ti1;
		Xi[121] = ti0 + 0.9415440651830207*ti1 + 0.3368898533922203*tr1;

		tr0 = xr[58];
		ti0 = xi[58];
		tr1 = xr[122];
		ti1 = xi[122];

		Xr[58] = tr0 - 0.9569403357322088*tr1 + 0.2902846772544624*ti1;
		Xi[58] = ti0 - 0.9569403357322088*ti1 - 0.2902846772544624*tr1;
		Xr[122] = tr0 + 0.9569403357322088*tr1 - 0.2902846772544624*ti1;
		Xi[122] = ti0 + 0.9569403357322088*ti1 + 0.2902846772544624*tr1;

		tr0 = xr[59];
		ti0 = xi[59];
		tr1 = xr[123];
		ti1 = xi[123];

		Xr[59] = tr0 - 0.9700312531945440*tr1 + 0.2429801799032641*ti1;
		Xi[59] = ti0 - 0.9700312531945440*ti1 - 0.2429801799032641*tr1;
		Xr[123] = tr0 + 0.9700312531945440*tr1 - 0.2429801799032641*ti1;
		Xi[123] = ti0 + 0.9700312531945440*ti1 + 0.2429801799032641*tr1;

		tr0 = xr[60];
		ti0 = xi[60];
		tr1 = xr[124];
		ti1 = xi[124];

		Xr[60] = tr0 - 0.9807852804032304*tr1 + 0.1950903220161286*ti1;
		Xi[60] = ti0 - 0.9807852804032304*ti1 - 0.1950903220161286*tr1;
		Xr[124] = tr0 + 0.9807852804032304*tr1 - 0.1950903220161286*ti1;
		Xi[124] = ti0 + 0.9807852804032304*ti1 + 0.1950903220161286*tr1;

		tr0 = xr[61];
		ti0 = xi[61];
		tr1 = xr[125];
		ti1 = xi[125];

		Xr[61] = tr0 - 0.9891765099647810*tr1 + 0.1467304744553618*ti1;
		Xi[61] = ti0 - 0.9891765099647810*ti1 - 0.1467304744553618*tr1;
		Xr[125] = tr0 + 0.9891765099647810*tr1 - 0.1467304744553618*ti1;
		Xi[125] = ti0 + 0.9891765099647810*ti1 + 0.1467304744553618*tr1;

		tr0 = xr[62];
		ti0 = xi[62];
		tr1 = xr[126];
		ti1 = xi[126];

		Xr[62] = tr0 - 0.9951847266721968*tr1 + 0.0980171403295608*ti1;
		Xi[62] = ti0 - 0.9951847266721968*ti1 - 0.0980171403295608*tr1;
		Xr[126] = tr0 + 0.9951847266721968*tr1 - 0.0980171403295608*ti1;
		Xi[126] = ti0 + 0.9951847266721968*ti1 + 0.0980171403295608*tr1;

		tr0 = xr[63];
		ti0 = xi[63];
		tr1 = xr[127];
		ti1 = xi[127];

		Xr[63] = tr0 - 0.9987954562051724*tr1 + 0.0490676743274180*ti1;
		Xi[63] = ti0 - 0.9987954562051724*ti1 - 0.0490676743274180*tr1;
		Xr[127] = tr0 + 0.9987954562051724*tr1 - 0.0490676743274180*ti1;
		Xi[127] = ti0 + 0.9987954562051724*ti1 + 0.0490676743274180*tr1;
		return;
	}
#endif
#endif

	arg = 2.0 * PI / N;

	for (k = 0; k < N2; k++) {  //combine half ffts into a full fft
		/*
		// set of operations in open form:
		Xr[k] = xr[k] + (floating)cos(2*PI * k / N) * xr[k + N / 2] + (floating)sin(2*PI * k / N) * xi[k + N / 2];
		Xi[k] = xi[k] + (floating)cos(2*PI * k / N) * xi[k + N / 2] - (floating)sin(2*PI * k / N) * xr[k + N / 2];

		Xr[k + N / 2] = xr[k] - (floating)cos(2*PI * k / N) * xr[k + N / 2] - (floating)sin(2*PI * k / N) * xi[k + N / 2];
		Xi[k + N / 2] = xi[k] - (floating)cos(2*PI * k / N) * xi[k + N / 2] + (floating)sin(2*PI * k / N) * xr[k + N / 2];
		*/

		tr0 = xr[k];
		ti0 = xi[k];

		arg_k = arg * (floating)k;

		cosArg = (floating)cos(arg_k);
		sinArg = (floating)sin(arg_k);

		cos_xr = cosArg * xr[k+N2];
		sin_xi = sinArg * xi[k+N2];
		cos_xi = cosArg * xi[k+N2];
		sin_xr = sinArg * xr[k+N2];

		Xr[k] = tr0 + cos_xr + sin_xi;
		Xi[k] = ti0 + cos_xi - sin_xr;

		Xr[k+N2] = tr0 - cos_xr - sin_xi;
		Xi[k+N2] = ti0 - cos_xi + sin_xr;
	}
}


/* Bluestein's FFT algorithm based on a special case of chirp z transform
   Algo time complexity is O(n log n). However, this is a few times slower than Cooley-Tuckey */
static void bluestein(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize)
{
	register int k;
	register floating mag2, arg, k2;

	//memset(buffers->wr, 0, (2 * N - 1) * sizeof(floating));
	//memset(buffers->wi, 0, (2 * N - 1) * sizeof(floating));
	memset(buffers->yr, 0, fftSize * sizeof(floating));
	memset(buffers->yi, 0, fftSize * sizeof(floating));
	//memset(buffers->fyr, 0, fftSize * sizeof(floating));
	//memset(buffers->fyi, 0, fftSize * sizeof(floating));
	memset(buffers->vr, 0, fftSize * sizeof(floating));
	memset(buffers->vi, 0, fftSize * sizeof(floating));
	//memset(buffers->fvr, 0, fftSize * sizeof(floating));
	//memset(buffers->fvi, 0, fftSize * sizeof(floating));
	//memset(buffers->gr, 0, fftSize * sizeof(floating));
	//memset(buffers->gi, 0, fftSize * sizeof(floating));
	memset(buffers->fgr, 0, fftSize * sizeof(floating));
	memset(buffers->fgi, 0, fftSize * sizeof(floating));

	// X(k) = Sum[n = 0->N-1] x(n) exp(-j 2 pi n k / N)

	// replace n k by -(k-n)^2/2 + n^2/2 + k^2/2

	// X(k) = w(k) Sum[n = 0->N-1] x(n) w(n) w*(k-n)

	// where w(n) = exp(-j pi n^2 / N) and w* is its complex conjugate

	//https://en.wikipedia.org/wiki/Chirp_Z-transform
	
	arg = PI / N;
	for (k = -(int)N + 1; k < (int)N; k++) { //compute phase factors w
		k2 = (floating)(k*k);
		buffers->wr[k + N - 1] = (floating)cos(arg*k2);
		buffers->wi[k + N - 1] = (floating)-sin(arg*k2);
	}

	for (k = 0; k < (int)N; k++) {  //compute y from x using phase factors.
		buffers->yr[k] = xr[k] * buffers->wr[N + k - 1] - xi[k] * buffers->wi[N + k - 1];
		buffers->yi[k] = xi[k] * buffers->wr[N + k - 1] + xr[k] * buffers->wi[N + k - 1];  // y = [xw + zeros up to next power of 2]
		//complexMultiplication(xr[k], xi[k], buffers->wr[N + k - 1], buffers->wi[N + k - 1], &(buffers->yr[k]), &(buffers->yi[k]));
	}

	cooley_tuckey(buffers->fyr, buffers->fyi, buffers->yr, buffers->yi, buffers, fftSize, fftSize);  // obtain fft of zero padded y

	for (k = 0; k < 2 * (int)N - 1; k++) {  //compute chirp filter v. v is the 1 inverse of complex conjugate of w
		mag2 = (buffers->wr[k] * buffers->wr[k]) + (buffers->wi[k] * buffers->wi[k]);
		buffers->vr[k] = buffers->wr[k] / mag2;
		buffers->vi[k] = -buffers->wi[k] / mag2;
		//complexReciprocal(buffers->wr[k], buffers->wi[k], &(buffers->vr[k]), &(buffers->vi[k]));
	}

	cooley_tuckey(buffers->fvr, buffers->fvi, buffers->vr, buffers->vi, buffers, fftSize, fftSize);  // convert v to freq. domain

	for (k = 0; k < (int)fftSize; k++) {  //filter y by v in freq. domain
		buffers->fgr[k] = buffers->fyr[k] * buffers->fvr[k] - buffers->fyi[k] * buffers->fvi[k];
		buffers->fgi[k] = buffers->fyi[k] * buffers->fvr[k] + buffers->fyr[k] * buffers->fvi[k];
		//complexMultiplication(buffers->fyr[k], buffers->fyi[k], buffers->fvr[k], buffers->fvi[k], &(buffers->fgr[k]), &(buffers->fgi[k]));
	}

	cooley_tuckey(buffers->gi, buffers->gr, buffers->fgi, buffers->fgr, buffers, fftSize, fftSize);  //convert result back to sample domain

	for (k = 0; k < (int)fftSize; k++) {
		buffers->gr[k] /= (floating)fftSize;
		buffers->gi[k] /= (floating)fftSize;
	} //at this point g is the ifft of fg

	for (k = 0; k < (int)N; k++) { //w times the convolution result is the fourier transform output
		Xr[k] = buffers->gr[N - 1 + k] * buffers->wr[N - 1 + k] - buffers->gi[N - 1 + k] * buffers->wi[N - 1 + k];
		Xi[k] = buffers->gr[N - 1 + k] * buffers->wi[N - 1 + k] + buffers->gi[N - 1 + k] * buffers->wr[N - 1 + k];
		//complexMultiplication(buffers->gr[N - 1 + k], buffers->gi[N - 1 + k], buffers->wr[N - 1 + k], buffers->wi[N - 1 + k], &Xr[k], &Xi[k]);
	}
}


/* Core fft function: calls Bluestein or Cooley-Tuckey algo depending on size of the input and calculates magnitude and phase from algo results */
static void fft_core(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize)
{
#if defined(MAGNITUDE) || defined(PHASE)
	register unsigned int i;
#endif
	register floating ti0, tr0, ti1, tr1;

	if (N == 1) { //Base case
		Xr[0] = xr[0];
		Xi[0] = xi[0];
		return;
	}
	else if (N == 2) { //Base case
		Xr[0] = xr[0] + xr[1];
		Xi[0] = xi[0] + xi[1];
		Xr[1] = xr[0] - xr[1];
		Xi[1] = xi[0] - xi[1];

		return;
	}
	else if (N == 4) { //Base case
		Xr[0] = xr[0] + xr[2] + xr[1] + xr[3];
		Xi[0] = xi[0] + xi[2] + xi[1] + xi[3];
		Xr[1] = xr[0] - xr[2] + xi[1] - xi[3];
		Xi[1] = xi[0] - xi[2] - xr[1] + xr[3];

		Xr[2] = xr[0] + xr[2] - xr[1] - xr[3];
		Xi[2] = xi[0] + xi[2] - xi[1] - xi[3];
		Xr[3] = xr[0] - xr[2] - xi[1] + xi[3];
		Xi[3] = xi[0] - xi[2] + xr[1] - xr[3];
		return;
	}
	else if (N == 8) { //Base case
		tr0 = 0.7071067811865476*(xr[1] - xr[5]);
		ti0 = 0.7071067811865476*(xi[1] - xi[5]);
		tr1 = 0.7071067811865476*(xr[3] - xr[7]);
		ti1 = 0.7071067811865476*(xi[3] - xi[7]);
		Xr[0] = (xr[0] + xr[4]) + (xr[2] + xr[6]) + ((xr[1] + xr[5]) + (xr[3] + xr[7]));
		Xi[0] = (xi[0] + xi[4]) + (xi[2] + xi[6]) + ((xi[1] + xi[5]) + (xi[3] + xi[7]));
		Xr[1] = (xr[0] - xr[4]) + (xi[2] - xi[6]) + (tr0 + ti1) + (ti0 - tr1);
		Xi[1] = (xi[0] - xi[4]) - (xr[2] - xr[6]) + (ti0 - tr1) - (tr0 + ti1);
		Xr[2] = (xr[0] + xr[4]) - (xr[2] + xr[6]) + ((xi[1] + xi[5]) - (xi[3] + xi[7]));
		Xi[2] = (xi[0] + xi[4]) - (xi[2] + xi[6]) - ((xr[1] + xr[5]) - (xr[3] + xr[7]));
		Xr[3] = (xr[0] - xr[4]) - (xi[2] - xi[6]) - (tr0 - ti1) + (ti0 + tr1);
		Xi[3] = (xi[0] - xi[4]) + (xr[2] - xr[6]) - (ti0 + tr1) - (tr0 - ti1);
		Xr[4] = (xr[0] + xr[4]) + (xr[2] + xr[6]) - ((xr[1] + xr[5]) + (xr[3] + xr[7]));
		Xi[4] = (xi[0] + xi[4]) + (xi[2] + xi[6]) - ((xi[1] + xi[5]) + (xi[3] + xi[7]));
		Xr[5] = (xr[0] - xr[4]) + (xi[2] - xi[6]) - (tr0 + ti1) - (ti0 - tr1);
		Xi[5] = (xi[0] - xi[4]) - (xr[2] - xr[6]) - (ti0 - tr1) + (tr0 + ti1);
		Xr[6] = (xr[0] + xr[4]) - (xr[2] + xr[6]) - ((xi[1] + xi[5]) - (xi[3] + xi[7]));
		Xi[6] = (xi[0] + xi[4]) - (xi[2] + xi[6]) + ((xr[1] + xr[5]) - (xr[3] + xr[7]));
		Xr[7] = (xr[0] - xr[4]) - (xi[2] - xi[6]) + (tr0 - ti1) - (ti0 + tr1);
		Xi[7] = (xi[0] - xi[4]) + (xr[2] - xr[6]) + (ti0 + tr1) + (tr0 - ti1);
		return;
	}
	else if (N % 2 == 0)
		cooley_tuckey(Xr, Xi, xr, xi, buffers, N, fftSize);   //Use Cooley-Tuckey algo if size is a multiple of 2
	else
		bluestein(Xr, Xi, xr, xi, buffers, N, fftSize);  // use Bluestein's algo othwerwise
#if defined(MAGNITUDE) || defined(PHASE)
	for (i = 0; i < N; i++) {  //calculate magnitude and phase
#if defined(MAGNITUDE)
		inst->abs[i] = (floating)sqrt(inst->Re[i] * inst->Re[i] + inst->Im[i] * inst->Im[i]);
#endif
#if defined(PHASE)
		inst->angle[i] = atan2(inst->Im[i], inst->Re[i]);
#endif
	}
#endif
}

floating Log2( floating n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( 2.0 );  
}  

/* Computes fftSize that will be used by bluestein algorithm and initializes bluestiein buffers */
static bool compute_fftSize(size_t N, size_t *fftSize, bluestein_buffers *buffers)
{
	int NN;

	NN = N;

	while (NN % 2 == 0)
		NN /= 2;

	if (NN == 1) {
		*fftSize = N;
		return true;  //power of 2
	}

	*fftSize = (size_t)pow(2, ceil( Log2(2.0 * (floating)NN - 1.0)));
	init_bluestein_buffers(buffers, NN, *fftSize);
	return false;
}


/* Inits buffers used by Bluestein algorithm. This is useful when same size buffers are used over and over by 2d fft methods */
static int init_bluestein_buffers(bluestein_buffers *buffers, size_t N, size_t fftSize)
{
	if (!buffers)
		return -1;
	buffers->wr = (floating*)malloc((2 * N - 1) * sizeof(floating));
	buffers->wi = (floating*)malloc((2 * N - 1) * sizeof(floating));
	buffers->yr = (floating*)malloc(fftSize * sizeof(floating));
	buffers->yi = (floating*)malloc(fftSize * sizeof(floating));
	buffers->fyr = (floating*)malloc(fftSize * sizeof(floating));
	buffers->fyi = (floating*)malloc(fftSize * sizeof(floating));
	buffers->vr = (floating*)malloc(fftSize * sizeof(floating));
	buffers->vi = (floating*)malloc(fftSize * sizeof(floating));
	buffers->fvr = (floating*)malloc(fftSize * sizeof(floating));
	buffers->fvi = (floating*)malloc(fftSize * sizeof(floating));
	buffers->gr = (floating*)malloc(fftSize * sizeof(floating));
	buffers->gi = (floating*)malloc(fftSize * sizeof(floating));
	buffers->fgr = (floating*)malloc(fftSize * sizeof(floating));
	buffers->fgi = (floating*)malloc(fftSize * sizeof(floating));
	return 0;
}


/* Releases buffers used by Bluestein algorithm */
static int clear_bluestein_buffers(bluestein_buffers *buffers)
{
	if (!buffers)
		return -1;
	//clear dynamic memory
	if (buffers->wr) {
		free(buffers->wr);
		buffers->wr = NULL;
	}
	if (buffers->wi) {
		free(buffers->wi);
		buffers->wi = NULL;
	}
	if (buffers->yr) {
		free(buffers->yr);
		buffers->yr = NULL;
	}
	if (buffers->yi) {
		free(buffers->yi);
		buffers->yi = NULL;
	}
	if (buffers->fyr) {
		free(buffers->fyr);
		buffers->fyr = NULL;
	}
	if (buffers->fyi) {
		free(buffers->fyi);
		buffers->fyi = NULL;
	}
	if (buffers->vr) {
		free(buffers->vr);
		buffers->vr = NULL;
	}
	if (buffers->vi) {
		free(buffers->vi);
		buffers->vi = NULL;
	}
	if (buffers->fvr) {
		free(buffers->fvr);
		buffers->fvr = NULL;
	}
	if (buffers->fvi) {
		free(buffers->fvi);
		buffers->fvi = NULL;
	}
	if (buffers->gr) {
		free(buffers->gr);
		buffers->gr = NULL;
	}
	if (buffers->gi) {
		free(buffers->gi);
		buffers->gi = NULL;
	}
	if (buffers->fgr) {
		free(buffers->fgr);
		buffers->fgr = NULL;
	}
	if (buffers->fgi) {
		free(buffers->fgi);
		buffers->fgi = NULL;
	}
	return 0;
}


/* utility function: returns whether N is a power of 2 or not*/
static bool power_of_2(size_t N)
{
	unsigned int NN, n = 1, numberOfOnes = 0, size_of_size_t = 16, i;

	if (N < 1)
		return false;

	NN = (unsigned int)N;
	//check if size is a power of 2. Max size is 2^16
	for (i = 0; i < size_of_size_t; i++) {
		if (n & NN) {
			numberOfOnes++;
			if (numberOfOnes>1) {
				return false;
			}
		}
		n *= 2;
	}
	return true;
}


/* utility function: returns prime factors of a number - max 32 factors allowed - to be used later when arbitrary size DIT algo is implemented */
static void primeFactorizer(size_t N, int *factors)
{
	int k = 0, j;
	const int maxFactorNum = 32;

	memset(factors, 0, maxFactorNum * sizeof(int));  // make sure factors array is created beforehand

	if (N < 2)
		return;

	while (N % 2 == 0 && k < maxFactorNum) {
		factors[k] = 2;
		k++;
		N = N / 2;
	}
	while (N % 3 == 0 && k < maxFactorNum) {
		factors[k] = 3;
		k++;
		N = N / 3;
	}
	j = 5;
	while (j*j <= (int)N && k < maxFactorNum) {
		while (N % j == 0 && k < maxFactorNum) {
			factors[k] = j;
			k++;
			N = N / j;
		}
		while (N % (j+2) == 0 && k < maxFactorNum) {
			factors[k] = j+2;
			k++;
			N = N / (j+2);
		}
		j += 6;
	}
	if (N > 2 && k < maxFactorNum) {
		factors[k] = N;
		k++;
	}
}


/* Utility function: z1 x z2 = y */
static void complexMultiplication(floating z1r, floating z1i, floating z2r, floating z2i, floating *yr, floating *yi)
{
	*yr = z1r * z2r - z1i * z2i;
	*yi = z1r * z2i + z1i * z2r;
}


/* Utility function: 1/z* = y */
static void complexReciprocal(floating zr, floating zi, floating *yr, floating *yi)
{
	register floating mag2 = zr*zr + zi*zi;
	*yr = zr / mag2;
	*yi = -zi / mag2;
}
