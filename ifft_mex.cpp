
#include "fft.h"
#include "C:\\Program Files\\MATLAB\\R2017b\\extern\\include\\mex.h"  //The path may need to be be adjusted.

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
	floating *xr, *xi, *Xr, *Xi, *XXr, *XXi, *YYr, *YYi, *xxr, *xxi, *yyr, *yyi;
	fft_instance fft1;
	fft2_instance fft2;
	int M, N, MM, NN, sz;

	if (nrhs != 4) {
		mexErrMsgTxt("Usage: [xr, xi] = ifft_mex(Xr, Xi, N, M);\nOutputs: Re{iFT(X)}, Im{yFT(X)}\nInputs: Re{X}, Im{X}, Row Count, Col Count\nSet Col count to 1 if input is 1D.");
		return;
	}
	if (nlhs > 2) {
		mexErrMsgTxt("Too many output arguments!\nUsage: [xr, xi] = ifft_mex(Xr, Xi, N, M);\nOutputs: Re{iFT(X)}, Im{yFT(X)}\nInputs: Re{X}, Im{X}, Row Count, Col Count\nSet Col count to 1 if input is 1D.");
		return;
	}
	if (nlhs < 2) {
		mexErrMsgTxt("Too few output arguments!\nUsage: Usage: [xr, xi] = ifft_mex(Xr, Xi, N, M);\nOutputs: Re{iFT(X)}, Im{yFT(X)}\nInputs: Re{X}, Im{X}, Row Count, Col Count\nSet Col count to 1 if input is 1D.");
		return;
	}

	MM = mxGetN(prhs[0]);  //get columns
	NN = mxGetM(prhs[0]);  //get rows

	M = mxGetN(prhs[1]);  //get columns
	N = mxGetM(prhs[1]);  //get rows

	if (MM != M || NN != N) {
		mexErrMsgTxt("Size mismatch!");
		return;
	}

	Xr = (floating*)mxGetData(prhs[0]);
	Xi = (floating*)mxGetData(prhs[1]);

	N = (int)mxGetScalar(prhs[2]);
	M = (int)mxGetScalar(prhs[3]);

	if (MM != M || NN != N) {
		mexErrMsgTxt("Size mismatch!");
		return;
	}

	sz = M*N;

	if (M == 1 || N == 1) {
		set_fft_instance(&fft1, sz);
		xr = new floating[sz];
		xi = new floating[sz];
		memcpy(fft1.Re, Xr, sz * sizeof(floating));
		memcpy(fft1.Im, Xi, sz * sizeof(floating));
		ifft_complex(xr, xi, &fft1, 0, sz);
		plhs[0] = mxCreateDoubleMatrix(sz, 1, mxREAL);
		memcpy(mxGetPr(plhs[0]), xr, sz * sizeof(floating));
		plhs[1] = mxCreateDoubleMatrix(sz, 1, mxREAL);
		memcpy(mxGetPr(plhs[1]), xi, sz * sizeof(floating));

		delete_fft_instance(&fft1);
		delete[] xi;
		delete[] xr;
	}
	else {
		XXr = new floating[M*N];
		XXi = new floating[M*N];
		YYr = new floating[M*N];
		YYi = new floating[M*N];
		memcpy(XXr, Xr, M*N * sizeof(floating));
		memcpy(XXi, Xi, M*N * sizeof(floating));
		convertMatlabImg2C(YYr, XXr, N, M);
		convertMatlabImg2C(YYi, XXi, N, M);

		set_fft2_instance(&fft2, N, M);
		memcpy(fft2.Re, YYr, sz * sizeof(floating));
		memcpy(fft2.Im, YYi, sz * sizeof(floating));

		xxr = new floating[M*N];
		xxi = new floating[M*N];
		yyr = new floating[M*N];
		yyi = new floating[M*N];

		ifft2_complex(xxr, xxi, &fft2, N, M);
		convertCImg2Matlab(yyr, xxr, N, M);
		convertCImg2Matlab(yyi, xxi, N, M);
		plhs[0] = mxCreateDoubleMatrix(N, M, mxREAL);
		memcpy(mxGetPr(plhs[0]), yyr, sz * sizeof(floating));
		plhs[1] = mxCreateDoubleMatrix(N, M, mxREAL);
		memcpy(mxGetPr(plhs[1]), yyi, sz * sizeof(floating));

		delete[] XXr;
		delete[] XXi;
		delete[] YYr;
		delete[] YYi;
		delete[] xxr;
		delete[] xxi;
		delete[] yyr;
		delete[] yyi;
	}
}
