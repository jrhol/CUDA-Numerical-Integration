#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <algorithm>

#define setThreads 1024
#define setIterations 28

double serialIntegration(double a, double delta, unsigned long long n)
{
	double integralSum = 0;

	for (unsigned long long i = 1; i <= n; i++) //Iterate through the sum
	{
		double x = (a + (i - 0.5) * delta);
		integralSum += exp(-(x * x)) * delta;
	}

	return (integralSum *= 2); //Multiply integral sum for symmetry
}

__global__ void cudaIntegration(double a, double delta, unsigned long long n, double* deviceArray, unsigned long long itemsPerThread)
{
	//Calculate Each Threads Index
	unsigned long long idx = threadIdx.x + (blockIdx.x * blockDim.x);

	// Calculate the starting index for each thread
	unsigned long long startIndex = idx * itemsPerThread;

	// Calculate the ending index for each thread, ensuring it does not exceed the array bounds
	unsigned long long endIndex = startIndex + itemsPerThread;
	if (endIndex > n) {
		endIndex = n;
	}

	//Calculate a Local Sum for each of the threads
	double threadVal = 0.0;
	for (unsigned long long i = startIndex; i < endIndex; i++) {
		double x = a + (i + 0.5) * delta;
		threadVal = exp(-(x * x)) * delta;
		deviceArray[idx] += threadVal; //Returns an Array with size of threads per block * num blocks
	}
	__syncthreads(); //Synchronise Threads so all threads have completed Computational Sums first

}

__global__ void cudaSumReduction(double* deviceArray, double* blocksums, int numBlocks)
{
	if (deviceArray == blocksums && numBlocks > 1024) //Perform this when we are reducing the final block sums but not when reducing the thread sums
	{
		// Allocate shared memory
		__shared__ double partial_Block_Sum[setThreads]; //Maximum static shared memory is 48kB so we must stride through the blocks (can be up to 65536 blocks)
		// Calculate thread ID
		int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread ID within the grid

		for (int stride = 0; stride < numBlocks; stride += 1024)
		{
			//Load elements into shared memory
			partial_Block_Sum[threadIdx.x] = deviceArray[idx + stride];
			__syncthreads();

			// Iterate of log base 2 the block dimension
			for (int s = 1; s < blockDim.x; s *= 2) {
				if (threadIdx.x % (2 * s) == 0) {
					partial_Block_Sum[threadIdx.x] += partial_Block_Sum[threadIdx.x + s];
				}
				__syncthreads();
			}

			// Let the thread 0 for this block write it's result to main memory
			if (threadIdx.x == 0) {
				blocksums[blockIdx.x] += partial_Block_Sum[0];
			}
		}
	}
	else //Perform this when we are reducing the thread sums into block sums
	{
		// Allocate shared memory
		__shared__ double partial_Block_Sum[setThreads];

		// Calculate thread ID
		int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread ID within the grid

		//Load elements into shared memory
		partial_Block_Sum[threadIdx.x] = deviceArray[idx];
		__syncthreads();

		// Iterate of log base 2 the block dimension
		for (int s = 1; s < blockDim.x; s *= 2) {
			if (threadIdx.x % (2 * s) == 0) {
				partial_Block_Sum[threadIdx.x] += partial_Block_Sum[threadIdx.x + s];
			}
			__syncthreads();
		}

		// Let the thread 0 for this block write it's result to main memory
		if (threadIdx.x == 0) {
			blocksums[blockIdx.x] = partial_Block_Sum[0];
		}
	}
}

__global__ void cudaThreadReduction(double* deviceArray, double* blocksums, int numBlocks)
{
	// Allocate shared memory
	__shared__ double partial_Block_Sum[setThreads];

	// Calculate thread ID
	int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread ID within the grid

	//Load elements into shared memory
	partial_Block_Sum[threadIdx.x] = deviceArray[idx];
	__syncthreads();

	// Iterate of log base 2 the block dimension
	for (int s = 1; s < blockDim.x; s *= 2) {
		if (threadIdx.x % (2 * s) == 0) {
			partial_Block_Sum[threadIdx.x] += partial_Block_Sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	if (threadIdx.x == 0) {
		blocksums[blockIdx.x] = partial_Block_Sum[0];
	}
}


__global__ void cudaBlockReduction(double* deviceArray, double* blocksums, int numBlocks)
{
	if (deviceArray == blocksums && numBlocks > 1024) //Perform this when we are reducing the final block sums but not when reducing the thread sums
	{
		// Allocate shared memory
		__shared__ double partial_Block_Sum[setThreads]; //Maximum static shared memory is 48kB so we must stride through the blocks (can be up to 65536 blocks)
		// Calculate thread ID
		int idx = blockIdx.x * blockDim.x + threadIdx.x; //Thread ID within the grid

		for (int stride = 0; stride < numBlocks; stride += 1024)
		{
			//Load elements into shared memory
			partial_Block_Sum[threadIdx.x] = deviceArray[idx + stride];
			__syncthreads();

			// Iterate of log base 2 the block dimension
			for (int s = 1; s < blockDim.x; s *= 2) {
				if (threadIdx.x % (2 * s) == 0) {
					partial_Block_Sum[threadIdx.x] += partial_Block_Sum[threadIdx.x + s];
				}
				__syncthreads();
			}

			// Let the thread 0 for this block write it's result to main memory
			if (threadIdx.x == 0) {
				blocksums[blockIdx.x] += partial_Block_Sum[0];
			}
		}
	}
}


int main()
{
	//Getting Device Information
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount(&count);
	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i);

		printf("\n**************** General Properties **************** \n");
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);

		printf("\n**************** Processing Properties **************** \n");
		printf("Clock rate: %d MHz\n", prop.clockRate / 1000);
		printf("Number of Multi Processors: %d\n", prop.multiProcessorCount);
		printf("Maximum number of Threads per Multi Processor: %d\n", prop.maxThreadsPerMultiProcessor);

		printf("\n**************** Memory Properties **************** \n");
		printf("Memory (at DDR) Clock rate: %d MHz\n", prop.memoryClockRate / 1000);
		printf("Memory Bus Width: %d Bits\n", prop.memoryBusWidth);
		printf("Total amount of shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
		printf("L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
		printf("Total (Available) global mem: %zu GB\n", prop.totalGlobalMem / (1024 * 1024 * 1024)); //Convert Bytes to GB

		printf("\n\n");
	}
	// End Device information


	//Creating some text Files that will be used for easy Data Analysis
	const char* SerialfilePath = "serialData.txt"; // Specify the file path
	//const char* SerialfilePath = "Serial.txt"; // Specify the file path
	// Delete the file if it already exists
	if (remove(SerialfilePath) != 0) {
		// If remove returns a non-zero value, it indicates an error
		perror("Error deleting file");
	}
	else {
		printf("File deleted successfully\n");
	}

	FILE* serialFile = fopen(SerialfilePath, "a+"); // Open the file in append mode or create if it doesn't exist

	if (serialFile == NULL) {
		fprintf(stderr, "Error opening file for appending.\n");
		return 1; // Exit with an error code
	}

	const char* cudaFilePath = "cudalData.txt"; // Specify the file path
	//const char* cudaFilePath = "Cuda.txt";
	// Delete the file if it already exists
	if (remove(cudaFilePath) != 0) {
		// If remove returns a non-zero value, it indicates an error
		perror("Error deleting file");
	}
	else {
		printf("File deleted successfully\n");
	}

	FILE* cudaFile = fopen(cudaFilePath, "a+"); // Open the file in append mode or create if it doesn't exist

	if (cudaFile == NULL) {
		fprintf(stderr, "Error opening file for appending.\n");
		return 1; // Exit with an error code
	}

	unsigned long long n = 1ULL; //The number of rectangles to Use for integration

	for (int i = 0; i < setIterations; i++) // Repeat for 32 Data Points
	{
		//Variables Used by every implementation
		//Current Value for N
		n = n * 2ULL; //Starting with 2 rectangles and repeating in multiples

		//Define A and B
		double a = 0; // Minimum value into the function
		double b = 3; // Maximum value into the function

		double delta = (b - a) / n; //Calculate Delta

		//Serial Code
		printf("\n*********************************************Serial Code**********************************************\n");

		//Start Serial Timing
		cudaDeviceSynchronize();
		double start = omp_get_wtime();

		//Serial Code
		long double integralSum = 0;
		integralSum = serialIntegration(a, delta, n);
		printf("The integral Sum is: %.16lf \n", integralSum);

		//End Serial Timing and Print Result
		cudaDeviceSynchronize();
		double end = omp_get_wtime();
		printf("\nstart = %.16g\nend = %.16g\nTotal Time is = %.16g seconds\n", start, end, end - start); // in secs

		// Write the variables to the Serial file
		fprintf(serialFile, "%.16lf,%.16lf,%.16lf\n", delta, (end - start), integralSum);

		//Start Cuda Timing (1 Block Many threads)

		//Start Cuda Timing (Many blocks 1 thread)
		//cudaIntegration << <numBlocks, BLOCK_SIZE >> > (a, delta, n, d_result);

		//4.3 CUDA Mixture Blocks and threads
		// Not starting Timing here as we only started timing for serial when function called.
		printf("\n*********************************************Cude Code 4.3 **********************************************\n");

		int threadsPerBlock = setThreads;
		int numBlocks = std::min((n + threadsPerBlock - 1) / threadsPerBlock, 65535ULL);  // Set the number of Blocks (Ensures Always At least 1 block and less than Max)
		unsigned long int itemsPerBlock = (n / numBlocks);
		unsigned long int itemsPerThread = ceil((double)itemsPerBlock / (double)threadsPerBlock); //Rounds up to make sure at least 1 item per thread

		printf("Number of N %d\n", n);
		printf("Number of Blocks %d\n", numBlocks);
		printf("Number of Threads %d\n", threadsPerBlock);
		printf("Items Per thread %d\n", itemsPerThread);
		printf("Size of Thread Value Array %d\n", threadsPerBlock * numBlocks);

		//Start Cuda Timing
		//Cuda requires additional memcpy so timing starts now
		cudaDeviceSynchronize();
		start = omp_get_wtime();

		//Allocate Host Memory
		double* hostIntegralAreas = (double*)malloc(sizeof(double) * threadsPerBlock * numBlocks);
		double* hostBlockSums = (double*)malloc(sizeof(double) * numBlocks);
		memset(hostIntegralAreas, 0, sizeof(double) * threadsPerBlock * numBlocks);// Set hostIntegralAreas to 0
		memset(hostBlockSums, 0, sizeof(double) * numBlocks);// Set hostBlockSums to 0


		//Define Device Memory
		double* deviceIntegralAreas; //Array that stores an integral sum for each of the threads
		double* deviceblockSums; //Array that stores an integral sum for each of the blocks

		//Allocate Device Memory
		cudaMalloc((void**)&deviceIntegralAreas, sizeof(double) * threadsPerBlock * numBlocks);
		cudaMalloc((void**)&deviceblockSums, sizeof(double) * numBlocks);
		cudaMemset(deviceIntegralAreas, 0, sizeof(double) * threadsPerBlock * numBlocks);// Set deviceIntegralAreas to 0
		cudaMemset(deviceblockSums, 0, sizeof(double) * numBlocks);// Set deviceblockSums to 0

		cudaIntegration << <numBlocks, threadsPerBlock >> > (a, delta, n, deviceIntegralAreas, itemsPerThread);
		cudaDeviceSynchronize();
		cudaSumReduction << <numBlocks, threadsPerBlock >> > (deviceIntegralAreas, deviceblockSums, numBlocks);
		cudaDeviceSynchronize();
		cudaSumReduction << <1, threadsPerBlock >> > (deviceblockSums, deviceblockSums, numBlocks);

		// Copy the thread array back to the host
		cudaMemcpy(hostIntegralAreas, deviceIntegralAreas, sizeof(double) * threadsPerBlock * numBlocks, cudaMemcpyDeviceToHost);
		cudaMemcpy(hostBlockSums, deviceblockSums, sizeof(double) * numBlocks, cudaMemcpyDeviceToHost);


		//for (int i = 0; i < 8; i++)
		//{
		//	printf("Value at index %d is %.16lf\n", i, hostBlockSums[i]);
		//}

		// Sum the array on the host
		//long double cudaSum = 0.0;
		//for (unsigned long long i = 0; i < (threadsPerBlock * numBlocks); ++i) {
		//	cudaSum += hostIntegralAreas[i];
		//}

		//cudaSum *= 2;

		//// Print or use the result as needed
		//printf("The Serial Addition integral Sum is: %.16lf \n", cudaSum);
		printf("*********** The Parralell Addition integral Sum is: %.16lf \n", hostBlockSums[0] * 2.0);

		//End Serial Timing and Print Result
		cudaDeviceSynchronize();
		end = omp_get_wtime();

		//Printing Time taken to complete
		printf("\nstart = %.16g\nend = %.16g\nTotal Time is = %.16g seconds\n", start, end, end - start); // in secs
		// Write the variables to the Serial file
		fprintf(cudaFile, "%.16lf,%.16lf,%.16lf\n", delta, (end - start), hostBlockSums[0] * 2.0);

		// Free memory
		free(hostIntegralAreas);
		free(hostBlockSums);
		cudaFree(deviceIntegralAreas);
		cudaFree(deviceblockSums);

	}

	fclose(serialFile);
	fclose(cudaFile);


	return 0;
}