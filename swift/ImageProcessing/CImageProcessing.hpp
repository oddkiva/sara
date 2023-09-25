#pragma once

void shakti_rgb8_to_gray32f_cpu(unsigned char* src, float* dst, int w, int h);
void shakti_gray32f_to_rgb8_cpu(float* src, unsigned char* dst, int w, int h);
