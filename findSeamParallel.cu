#include <stdio.h>
#include <stdint.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
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

void readPnmUchar3(char * fileName,
		int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);

	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
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

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

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
void writePnmUchar3(uchar3 * pixels, int width, int height,
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "P3\n%i\n%i\n255\n", width, height);

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);

	fclose(f);
}

__global__ void findSeamKernel(uint8_t* inEnergy, int width, int height,
		uint8_t* outMap, int *line)
{
	// TODO
  int rowIndex = height - 1;
  int c = threadIdx.x + blockIdx.x * blockDim.x;
  if (c < width)
  {
    outMap[rowIndex * width + c] = inEnergy[rowIndex * width + c];
    __syncthreads();
    int val1 = 255;
    int val2 = 255;
    int val3 = 255;
    int minVal;
    while (rowIndex>0)
    {
      rowIndex--;
      if (c>0)
        val1 = outMap[(rowIndex + 1) * width + c - 1];
      else val1 = 255;
      val2 = outMap[(rowIndex + 1) * width + c];
      if (c < width - 1)
        val3 = outMap[(rowIndex + 1) * width + c + 1];
      else val3 = 255;
      minVal = min(val1, min(val2, val3));
      outMap[rowIndex * width + c] = minVal + inEnergy[rowIndex * width + c];
      if (minVal == val1)
        line[rowIndex * width + c] = -1;
      else if (minVal == val2)
        line[rowIndex *  width + c] = 0;
      else line[rowIndex * width + c] = 1;
      __syncthreads();
    }
  }
}

void findSeam(uint8_t * inEnergy, int width, int height,
    uint8_t * outMap, int* line, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printf("GPU name: %s\n", devProp.name);
	printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

	// TODO
  //Host allocates memories on device
  int *d_line;
  uint8_t *d_inEnergy, *d_outMap;
  CHECK(cudaMalloc(&d_inEnergy, width * height * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_outMap, width * height * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_line, width * height * sizeof(int)));
  //Host copies data to divece memories
  CHECK(cudaMemcpy(d_inEnergy, inEnergy, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
  //Host invokes kernel function
  dim3 gridSize((width - 1)/blockSize.x + 1, (height - 1)/blockSize.y + 1);
  findSeamKernel<<<gridSize, blockSize>>>(d_inEnergy, width, height, d_outMap, d_line);
  cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
  //Host copies result form divece memories
  CHECK(cudaMemcpy(outMap, d_outMap, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(line, d_line, width * height * sizeof(int), cudaMemcpyDeviceToHost));
  //Host free device memories
  CHECK(cudaFree(d_inEnergy));
  CHECK(cudaFree(d_outMap));
  CHECK(cudaFree(d_line));
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time %f ms use device\n", time);
}
void checkSeam(uint8_t *map, uchar3 *pixels, int width, int height, int *line)
{
  int k = 0;
  for (int c = 0; c < width; c++)
    {
      if (map[c]<=map[k])
      {
        k = c;
      }
    }
  printf("%i %i\n",k, map[k]);
  pixels[k].x = 255;
  pixels[k].y = 0;
  pixels[k].z = 0;
  int r = 0;
    while (r < height)
  {
    r++;
    k +=  line[(r-1)*width+k];
    pixels[r*width+k].x = 255;
    pixels[r*width+k].y = 0;
    pixels[r*width+k].z = 0;
  }
}

int main(int argc, char ** argv)
{
	// Read input image file
	int numChannels, width, height;
  uchar3 * inPixels;
  uint8_t *gray;

	readPnm(argv[1], numChannels, width, height, gray);
  readPnmUchar3(argv[2], width, height, inPixels);
	printf("Image size (width x height): %i x %i\n\n", width, height);

		// Blur input image using device
	uint8_t * map = (uint8_t *)malloc(width * height * sizeof(uint8_t));
  int *line = (int*)malloc(width * height * sizeof(int));
	if (argc == 6)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}
  findSeam(gray, width, height, map, line, 512);
  checkSeam(map, inPixels, width, height, line);
	// Write results to files
	char * name = argv[3];
  //char * nameedge = "edge.txt";
	writePnmUchar3(inPixels, width, height, name);
  //writeMatrix(deviceOutPixels, 1, width, height, name);
	// Free memories
	free(inPixels);
	free(map);
  free(line);
}
