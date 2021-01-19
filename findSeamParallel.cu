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

__global__ void findSeamKernel(uint8_t* inPixels, int width, int height,		
		uint8_t* outPixels)
{
	// TODO

  int c = threadIdx.x + blockIdx.x * blockDim.x;
	int r = threadIdx.y + blockIdx.y * blockDim.y;// chỉ số phần tử pixle
  if (r < height && c < width) //check có cần làm hay không, thread ngoài biên thì không làm
  {
    int i = r * width + c;
    float outx = 0.0f;
    float outy = 0.0f;
    int iFilter = 0; //chỉ số phần tử ở filter
    for (int x = -1; x <= 1; x++)
      for (int y = -1; y <= 1; y++) //duyệt filter
      {
        int rx = r + x;
        int cy = c + y;
        if (rx < 0) rx = 0;
        if (rx > height - 1) rx = height - 1;
        if (cy < 0) cy = 0;
        if (cy > width - 1) cy = width - 1; // ngoài biên thì lấy phần tử gần nhất
        int k = rx * width + cy;
        outx += float(float(inPixels[k]) * float(filterX[iFilter]/4));
        outy += float(float(inPixels[k]) * float(filterY[iFilter]/4));

        iFilter++;
      }
    outPixels[i] = abs(outx) + abs(outy);
  }
}

void detectionEdge(uint8_t * inPixels, int width, int height, float *filterX, float *filterY, int filterWidth,
		uint8_t * outPixels, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	printf("GPU name: %s\n", devProp.name);
	printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

	// TODO
  //Host allocates memories on device
  uint8_t *d_inPixels, *d_outPixels;
  float *d_filterX, *d_filterY;
  CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_outPixels, width * height * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_filterX, filterWidth * filterWidth * sizeof(float)));
  CHECK(cudaMalloc(&d_filterY, filterWidth * filterWidth * sizeof(float)));
  //Host copies data to divece memories
  CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_filterX, filterX, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_filterY, filterY, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
  //Host invokes kernel function
  dim3 gridSize((width - 1)/blockSize.x + 1, (height - 1)/blockSize.y + 1);
  edgeDetectionKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_filterX, d_filterY, filterWidth, d_outPixels);
  cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
  //Host copies result form divece memories
  CHECK(cudaMemcpy(outPixels, d_outPixels, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  //Host free device memories
  CHECK(cudaFree(d_inPixels));
  CHECK(cudaFree(d_outPixels));
  CHECK(cudaFree(d_filterX));
  CHECK(cudaFree(d_filterY));
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

	// Read input image file
	int numChannels, width, height;
	uint8_t * inPixels;
	readPnm(argv[1], numChannels, width, height, inPixels);
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Set up filter xSobel
	float filterX [9]= {1, 0, -1, 2, 0, -2, 1, 0, -1};
  // Set up filter ySobel
	float filterY [9]= {1, 2, 1, 0, 0, 0, -1, -2, -1};

	// Blur input image using device
	uint8_t * deviceOutPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
	dim3 blockSize(32, 32); // Default
	if (argc == 6)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}
	detectionEdge(inPixels, width, height, filterX, filterY, 3, deviceOutPixels, blockSize);

	// Write results to files
	char * name = argv[2];
  //char * nameedge = "edge.txt";
	writePnm(deviceOutPixels, 1, width, height, name);
  //writeMatrix(deviceOutPixels, 1, width, height, name);
	// Free memories
	free(inPixels);
	free(deviceOutPixels);
}
