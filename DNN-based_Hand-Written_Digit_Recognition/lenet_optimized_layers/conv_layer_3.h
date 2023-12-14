#ifndef CONV_LAYER_3_H
#define CONV_LAYER_3_H

void convolution5_hw(float input[16][5][5], float weights[120][16][5][5], float bias[120], float output[120][1][1]);

#endif