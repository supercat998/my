#include "ap_axi_sdata.h"
#include <stdint.h>
#include "ap_int.h"

#define WIDTH 	128
#define N 	9

typedef int DTYPE;


void filter2d_accel(DTYPE* img_in, DTYPE* kernel, DTYPE* img_out, int rows, int cols);
