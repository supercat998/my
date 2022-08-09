#include <iostream>
using namespace std;
#include <math.h>
#include "filter2d.h"

static DTYPE matrix(DTYPE WB[3][3], DTYPE K[9])
{
#pragma HLS PIPELINE
	int i, j;
	short int out_pix = 0;
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++)
			out_pix += WB[i][j] * K[i*3+j];

	return (DTYPE) out_pix;
}

void filter2d_accel(DTYPE* img_in, DTYPE* kernel, DTYPE* img_out, int rows, int cols)
{
    
#pragma HLS INTERFACE m_axi port=img_in offset=slave depth=16384
#pragma HLS INTERFACE m_axi port=kernel offset=slave depth=9
#pragma HLS INTERFACE m_axi port=img_out offset=slave depth=16384
#pragma HLS INTERFACE s_axilite port=rows  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=cols  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL


	DTYPE mx[N] = {0,0,0,0,0,0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=mx complete dim=0

	DTYPE filter ;
	DTYPE LineBuffer[3][WIDTH];
#pragma HLS ARRAY_PARTITION variable=LineBuffer complete dim=1

	DTYPE WindowBuffer[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
#pragma HLS ARRAY_PARTITION variable=WindowBuffer complete dim=0

	ap_uint<8> row, col;
    ap_uint<5> n;
	ap_uint<2> lb_r_i;
	ap_uint<2> top, mid, btm;//line buffer row index



    for(n = 0; n < 9; n++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=9
		mx[n] = (DTYPE) *kernel;
        kernel++;
	}


	for(col = 0; col < cols; col++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS pipeline
		LineBuffer[0][col] = (DTYPE) *img_in;
		img_in++;
	}

	for(col = 0; col < cols; col++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS pipeline
		LineBuffer[1][col] = (DTYPE) *img_in;
		img_in++;
	}


	lb_r_i = 2;
	for(row = 2; row < rows; row++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
		if(lb_r_i == 2)
		{
			top = 0; mid = 1; btm = 2;
		}
		else if(lb_r_i == 0)
		{
			top = 1; mid = 2; btm = 0;
		}
		else if(lb_r_i == 1)
		{
			top = 2; mid = 0; btm = 1;
		}

		for(col = 0; col < cols; col++)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS pipeline II=24
			if(row < rows)
			{
				LineBuffer[btm][col] = (DTYPE) *img_in;
				img_in++;
			}
			else
				LineBuffer[btm][col] = 0;
			WindowBuffer[0][0] = WindowBuffer[0][1];
			WindowBuffer[1][0] = WindowBuffer[1][1];
			WindowBuffer[2][0] = WindowBuffer[2][1];
			WindowBuffer[0][1] = WindowBuffer[0][2];
			WindowBuffer[1][1] = WindowBuffer[1][2];
			WindowBuffer[2][1] = WindowBuffer[2][2];
			WindowBuffer[0][2] = LineBuffer[top][col];
			WindowBuffer[1][2] = LineBuffer[mid][col];
			WindowBuffer[2][2] = LineBuffer[btm][col];
			if(col > 1)
            {
				filter = matrix(WindowBuffer,mx);
                *img_out = filter;
                img_out++;
            }

		}
		lb_r_i++;
		if(lb_r_i == 3) lb_r_i = 0;

	}
}
