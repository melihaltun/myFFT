
#include "fft.h"
#include "C:\\Program Files\\MATLAB\\R2017b\\extern\\include\\mex.h"  //path may need to be be adjusted.

void convertMatlabImg2C(floating *out, floating *in, int N, int M)
{
	int i, j, ii, jj;
	ii = 0; jj = 0;
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			out[lin_index(ii, jj, M)] = in[lin_index(i, j, M)];
			ii++;
			if (ii >= N) {
				ii = 0;
				jj++;
			}
		}
	}
}

void convertCImg2Matlab(floating *out, floating *in, int N, int M)
{
	int i, j, ii, jj;
	ii = 0; jj = 0;
	for (j = 0; j < M; j++) {
		for (i = 0; i < N; i++) {
			out[lin_index(ii, jj, M)] = in[lin_index(i, j, M)];
			jj++;
			if (jj >= M) {
				jj = 0;
				ii++;
			}
		}
	}
}


/* The gateway routine. */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	floating *x, *xx, *yy;
	fft_instance fft1;
	fft2_instance fft2;
	int M, N, MM, NN, sz;
#if defined (MAGNITUDE) && defined (PHASE)
	if (nrhs != 3) {
		mexErrMsgTxt("Usage: [Xr, Xi, Xm, Xp] = fft_mex(x, N, M);\nOutputs: Re{FT(x)}, Im{FT(x)}, |FT(x)|, <FT(x)\nInputs: x, Row Count, Col Count\nSet Col count to 1 if input is 1D.");
		return;
	}
	if (nlhs > 4) {
		mexErrMsgTxt("Too many output arguments!\nUsage: [Xr, Xi, Xm, Xp] = fft_mex(x, N, M);\nOutputs: Re{FT(x)}, Im{FT(x)}, |FT(x)|, <FT(x)\nInputs: x, Row Count, Col Count\nSet Col count to 1 if input is 1D.");
		return;
	}
	if (nlhs < 2) {
		mexErrMsgTxt("Too few output arguments!\nUsage: [Xr, Xi, Xm, Xp] = fft_mex(x, N, M);\nOutputs: Re{FT(x)}, Im{FT(x)}, |FT(x)|, <FT(x)\nInputs: x, Row Count, Col Count\nSet Col count to 1 if input is 1D.");
		return;
	}
#else 
	if (nrhs != 3) {
		mexErrMsgTxt("Usage: [Xr, Xi] = fft_mex(x, N, M);\nOutputs: Re{FT(x)}, Im{FT(x)}\nInputs: x, Row Count, Col Count\nSet Col count to 1 if input is 1D.");
		return;
	}
	if (nlhs > 2) {
		mexErrMsgTxt("Too many output arguments!\nUsage: [Xr, Xi] = fft_mex(x, N, M);\nOutputs: Re{FT(x)}, Im{FT(x)}\nInputs: x, Row Count, Col Count\nSet Col count to 1 if input is 1D.");
		return;
	}
	if (nlhs < 2) {
		mexErrMsgTxt("Too few output arguments!\nUsage: [Xr, Xi] = fft_mex(x, N, M);\nOutputs: Re{FT(x)}, Im{FT(x)}\nInputs: x, Row Count, Col Count\nSet Col count to 1 if input is 1D.");
		return;
	}
#endif

	MM = mxGetN(prhs[0]);  //get columns
	NN = mxGetM(prhs[0]);  //get rows

	x = (floating*)mxGetData(prhs[0]);

	N = (int)mxGetScalar(prhs[1]);
	M = (int)mxGetScalar(prhs[2]);

	if (MM != M || NN != N) {
		mexErrMsgTxt("Size mismatch!");
		return;
	}

	sz = M*N;

	if (M == 1 || N == 1) {
		set_fft_instance(&fft1, sz);
		fft_real(&fft1, x, 0, sz);
		plhs[0] = mxCreateDoubleMatrix(sz, 1, mxREAL);
		memcpy(mxGetPr(plhs[0]), fft1.Re, sz * sizeof(floating));
		plhs[1] = mxCreateDoubleMatrix(sz, 1, mxREAL);
		memcpy(mxGetPr(plhs[1]), fft1.Im, sz * sizeof(floating));
#if defined (MAGNITUDE) && defined (PHASE)
		if (nlhs > 2) {
			plhs[2] = mxCreateDoubleMatrix(sz, 1, mxREAL);
			memcpy(mxGetPr(plhs[2]), fft1.abs, sz * sizeof(floating));
		}
		if (nlhs > 3) {
			plhs[3] = mxCreateDoubleMatrix(sz, 1, mxREAL);
			memcpy(mxGetPr(plhs[3]), fft1.angle, sz * sizeof(floating));
		}
#endif
		delete_fft_instance(&fft1);
	}
	else {
		xx = new floating[M*N];
		yy = new floating[M*N];
		memcpy(xx, x, M*N * sizeof(floating));
		convertMatlabImg2C(yy, xx, N, M);

		set_fft2_instance(&fft2, N, M);
		fft2_real(&fft2, yy, N, M);

		plhs[0] = mxCreateDoubleMatrix(N, M, mxREAL);
		convertCImg2Matlab(xx, fft2.Re, N, M);
		memcpy(mxGetPr(plhs[0]), xx, sz * sizeof(floating));
		plhs[1] = mxCreateDoubleMatrix(N, M, mxREAL);
		convertCImg2Matlab(xx, fft2.Im, N, M);
		memcpy(mxGetPr(plhs[1]), xx, sz * sizeof(floating));
#if defined (MAGNITUDE) && defined (PHASE)
		if (nlhs > 2) {
			plhs[2] = mxCreateDoubleMatrix(N, M, mxREAL);
			convertCImg2Matlab(xx, fft2.abs, N, M);
			memcpy(mxGetPr(plhs[2]), xx, sz * sizeof(floating));
		}
		if (nlhs > 3) {
			plhs[3] = mxCreateDoubleMatrix(N, M, mxREAL);
			convertCImg2Matlab(xx, fft2.angle, N, M);
			memcpy(mxGetPr(plhs[3]), xx, sz * sizeof(floating));
		}
#endif
		delete_fft2_instance(&fft2);
		delete[]xx;
		delete[]yy;
	}
}
