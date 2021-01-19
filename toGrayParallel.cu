#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

void readPnm(char * fileName,
		int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height,
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height);

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}

__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels)
{
	// TODO
    // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	if (indexX < height && indexY < width)
	{
		int i = indexX * width + indexY;
		uint8_t red = inPixels[3 * i];
		uint8_t green = inPixels[3 * i + 1];
		uint8_t blue = inPixels[3 * i + 2];
		outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
	}

}

void convertRgb2Gray(uint8_t * inPixels, int width, int height,
		uint8_t * outPixels, dim3 blockSize=dim3(1))
{
	GpuTimer timer;
	timer.Start();

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printf("GPU name: %s\n", devProp.name);
	printf("GPU compute capability: %d.%d\n\n", devProp.major, devProp.minor);

	// TODO: Allocate device memories
	uint8_t *d_inPixels, *d_outPixels;
	CHECK(cudaMalloc(&d_inPixels, height * width * 3 * sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_outPixels, height * width * 3 * sizeof(uint8_t)));

	// TODO: Copy data to device memories
	CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));

	// TODO: Set grid size and call kernel (remember to check kernel error)
	dim3 gridsize((height - 1)/blockSize.x + 1, (width - 1)/blockSize.y + 1);
  printf("Grid size: %i x %i\n",gridsize.x, gridsize.y);
  printf("Block size: %i x %i\n",blockSize.x, blockSize.y);
	convertRgb2GrayKernel<<<gridsize, blockSize>>>(d_inPixels, width, height, d_outPixels);
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	// TODO: Copy result from device memories
	CHECK(cudaMemcpy(outPixels, d_outPixels, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	// TODO: Free device memories
	CHECK(cudaFree(d_inPixels));
	CHECK(cudaFree(d_outPixels));

	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time %f ms use device", time);
}

int main(int argc, char ** argv)
{
	if (argc != 3 && argc != 5)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input RGB image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Convert RGB to grayscale using device
	uint8_t * outPixels= (uint8_t *)malloc(width * height);
	dim3 blockSize(32, 32); // Default
	if (argc == 5)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}
	convertRgb2Gray(inPixels, width, height, outPixels, blockSize);

		// Write results to files
    char *name = argv[2];
	writePnm(outPixels, 1, width, height, name);

	// Free memories
	free(inPixels);
	free(outPixels);
}
