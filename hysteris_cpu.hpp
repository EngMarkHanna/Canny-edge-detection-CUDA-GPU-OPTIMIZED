#pragma once

void hysteresis_scalar(unsigned char* map, int mapW, int mapH);
void final_output(const unsigned char* map, int mapW, unsigned char* dst, int W, int H);
void hysteresis_omp_basic(unsigned char* map, int mapW, int mapH, int num_threads);
void hysteresis_omp_numa(unsigned char* map, int mapW, int mapH, int num_threads);