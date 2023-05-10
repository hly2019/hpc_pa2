// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include  <iostream>
#define b 32
#define MAX_SHARE_SIZE 32 * 1024 // real max 48KB

namespace {

__global__ void kernel(int n, int k, int *graph) {
    auto i = blockIdx.y * blockDim.y + threadIdx.y;
    auto j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < n) {
        graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
    }
}

}


__global__ void kernel_stage2(int p, int n, int* graph) { // 第p步的操作。

    int centra_left = p * b;
    int centra_right = min((p + 1) * b, n);

    __shared__ int central_block[b][b]; // 中心块
    __shared__ int block[b][b]; // 待更新块
    
    int thread_x = threadIdx.x; // 块内偏移量
    int thread_y = threadIdx.y;

    int cent_x = thread_x + centra_left; // thread_x:[0, 32)
    int cent_y = thread_y + centra_left; // thread_y: [0, 32)
    if(cent_x < n && cent_y < n) { // 缓存中心块
        central_block[thread_x][thread_y] = graph[cent_x * n + cent_y];
    }

    // 缓存待更新块
    int block_x = blockIdx.x; // 对标centra_left
    int block_y = blockIdx.y;

    if(block_x >= p) block_x += 1; // 跳过中心块
    if(block_y == 1) { // 竖着的
        int i = cent_x; // x方向和中心块一致
        int j = thread_y + block_x * b;
        if(i < n && j < n) { // 缓存中心块
            block[thread_x][thread_y] = graph[i * n + j];
        }
    } else { // 横着的
        int i = thread_x + block_x * b;
        int j = cent_y;
        if(i < n && j < n) {
            block[thread_x][thread_y] = graph[i * n + j];
        }
    }

    __syncthreads(); // 同步

    // block[thread_x * n + thread_y] = std::min(block[thread_x * n + thread_y], block[thread_x * n + thread_y])
    int range = centra_right - centra_left;
    for(int k = 0; k < range; k++) {
        block[thread_x][thread_y] = min(block[thread_x][thread_y], central_block[thread_x][k] + block[k][thread_y]);
        __syncthreads();
    }
    // 写回
    if(block_y == 1) { // 竖着的
        int i = cent_x; // x方向和中心块一致
        int j = thread_y + block_x * b;
        if(i < n && j < n) { // 缓存中心块
            graph[i * n + j] = block[thread_x][thread_y];
        }
    } else { // 横着的
        int i = thread_x + block_x * b;
        int j = cent_y;
        if(i < n && j < n) {
            graph[i * n + j] = block[thread_x][thread_y];
        }
    }    
}

__global__ void kernel_stage1(int p, int n, int* graph) {
    int centra_left = p * b;
    int centra_right = min((p + 1) * b, n);

    __shared__ int central_block[b][b]; // 中心块
    
    int thread_x = threadIdx.x; // 块内偏移量
    int thread_y = threadIdx.y;

    int cent_x = thread_x + centra_left; // thread_x:[0, 32)
    int cent_y = thread_y + centra_left; // thread_y: [0, 32)
    if(cent_x < n && cent_y < n) { // 缓存中心块
        central_block[thread_x][thread_y] = graph[cent_x * n + cent_y];
    }
    int range = centra_right - centra_left;
    __syncthreads(); // 同步
    for(int k = 0; k < range; k++) {
        central_block[thread_x][thread_y] = min(central_block[thread_x][thread_y], 
            central_block[thread_x][k] + central_block[k][thread_y]);
        __syncthreads(); // 同步
    }
    if(cent_x < n && cent_y < n) { // 缓存中心块
        graph[cent_x * n + cent_y] = central_block[thread_x][thread_y];
    }
}

__global__ void kernel_stage3(int p, int n, int* graph) {
    // 拷贝十字块，只需要当前块对应的两个即可
    __shared__ int v_cross_block[b][b]; // 位于十字块竖直方向的
    __shared__ int h_cross_block[b][b]; // 位于十字块水平方向的

    int block_x = blockIdx.x; // 对标centra_left
    int block_y = blockIdx.y;

    if(block_x >= p) block_x += 1; // 跳过中心块
    if(block_y >= p) block_y += 1; // 跳过中心块

    int centra_left = p * b;

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int v_cross_x = thread_x + centra_left;
    int v_cross_y = thread_y + block_y * b;

    if(v_cross_x < n && v_cross_y < n) {
        v_cross_block[thread_x][thread_y] = graph[v_cross_x * n + v_cross_y];
    }

    int h_cross_x = thread_x + block_x * b;
    int h_cross_y = thread_y + centra_left;

    if(h_cross_x < n && h_cross_y < n) {
        h_cross_block[thread_x][thread_y] = graph[h_cross_x * n + h_cross_y];
    }

    // 拷贝待计算块
    __shared__ int block[b][b];
    
    int i = thread_x + block_x * b;
    int j = thread_y + block_y * b;

    if(i < n && j < n) {
        block[thread_x][thread_y] = graph[i * b + j]; 
    }

    // 更新
    int thread_k = threadIdx.z;
    block[thread_x][thread_y] = min(block[thread_x][thread_y],
         h_cross_block[thread_x][thread_k] + v_cross_block[thread_k][thread_y]);
    
    // 写回
    if(i < n && j < n) {
        graph[i * b + j] = block[thread_x][thread_y]; 
    }

}

void apsp(int n, int* graph) {
    for(int p = 0; p < (n - 1) / b + 1; p++) {
        // stage 1? 不并行?
        // int left_bound = p * b;
        // int right_bound = std::min((p + 1) * b, n); // 边界情况
        // for(int k = left_bound; k < right_bound; k++) {
        //     for(int i = left_bound; i < right_bound; i++) {
        //         for(int j = left_bound; j < right_bound; j++) {
        //             graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
        //         }
        //     }
        // }
        // stage 1:
        dim3 thr_1(b, b);
        // dim3 blk_1 = 1;
        kernel_stage1<<<1, thr_1>>>(p, n, graph);
        int * test = (int*)malloc(1);
        cudaMemcpy(test, (void*)graph, sizeof(int), cudaMemcpyDeviceToHost);
        printf("test: %d\n", *test);

        // stage 2:
        dim3 thr_2(b, b);
        // dim3 blk((n - 1) / 32 + 1, (n - 1) / 32 + 1);
        dim3 blk_2((n - 1) / b, 2); // 所有待更新的block(去掉中心块)
        kernel_stage2<<<blk_2, thr_2>>>(p, n, graph);

        // // stage 3:
        dim3 thr_3(b, b, b); // TODO: 这样是否可行
        dim3 blk_3((n - 1) / b, (n - 1) / b);
        kernel_stage3<<<blk_3, thr_3>>>(p, n, graph);
    }


}

