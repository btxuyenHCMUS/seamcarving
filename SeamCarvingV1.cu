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
		uint8_t * grayPixels)
{
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	if (indexX < height && indexY < width)
	{
		int i = indexX * width + indexY;
		float red = 0.299*inPixels[3 * i];
		float green = 0.587*inPixels[3 * i + 1];
		float blue = 0.114*inPixels[3 * i + 2];
		grayPixels[i] = red + green + blue;
	}

}

void convertRgb2Gray(uint8_t *d_inPixels, uint8_t *d_grayPixels, int width, int height, dim3 blockSize=dim3(1))
{
	//Set grid size and call kernel (remember to check kernel error)
	dim3 gridsize((height - 1)/blockSize.x + 1, (width - 1)/blockSize.y + 1);
	convertRgb2GrayKernel<<<gridsize, blockSize>>>(d_inPixels, width, height, d_grayPixels);
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}
__global__ void edgeDetectionKernel(uint8_t* grayPixels, int width, int height,
		float * filterX, float * filterY, int filterWidth, uint8_t* edgePixels)
{
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
        outx += float(float(grayPixels[k]) * float(filterX[iFilter]));
        outy += float(float(grayPixels[k]) * float(filterY[iFilter]));

        iFilter++;
      }
    edgePixels[i] = abs(outx) + abs(outy);
  }
}

void detectionEdge(uint8_t *d_grayPixels, uint8_t *d_edgePixels, int width,
  int height, float *d_filterX, float *d_filterY, int filterWidth,
		dim3 blockSize=dim3(1, 1))
{
	//Host invokes kernel function
  dim3 gridSize((width - 1)/blockSize.x + 1, (height - 1)/blockSize.y + 1);
  edgeDetectionKernel<<<gridSize, blockSize>>>(d_grayPixels, width, height, d_filterX, d_filterY, filterWidth, d_edgePixels);
  cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
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
      else val1 = 1000;
      val2 = outMap[(rowIndex + 1) * width + c];
      if (c < width - 1)
        val3 = outMap[(rowIndex + 1) * width + c + 1];
      else val3 = 1000;
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
void findSeam(uint8_t * d_edgePixels, uint8_t * d_map, int* d_seam,
  int width, int height, dim3 blockSize=dim3(1, 1))
{
	//Host invokes kernel function
  dim3 gridSize((width - 1)/blockSize.x + 1, (height - 1)/blockSize.y + 1);
  findSeamKernel<<<gridSize, blockSize>>>(d_edgePixels, width, height, d_map, d_seam);
  cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}
void checkSeam(uint8_t *map, uint8_t *pixels, int width, int height, int *line)
{
  int k = 0;
  for (int c = 0; c < width; c++)
    {
      if (map[c]<map[k])
      {
        k = c;
      }
    }
  printf("%i %i\n",k, map[k]);
  pixels[3*k] = 255;
  pixels[3*k+1] = 0;
  pixels[3*k+2] = 0;
  int r = 0;

  while (r < height)
  {
    r++;
    k +=  line[(r-1)*width+k];
    pixels[r*3*width+3*k] = 255;
    pixels[r*3*width+3*k+1] = 0;
    pixels[r*3*width+3*k+2] = 0;
  }
}

__global__ void cutSeamKernel(uint8_t*inPixels, uint8_t *outPixels, uint8_t *inEdges, uint8_t *outEdges,
  int width, int height, int k, int *seam)
{
  int c = threadIdx.x + blockIdx.x*blockDim.x;
  int r = 0;
  if (c<width)
  {
    while (r<height)
    {
					if (c + r < k)
          {
            outPixels[3 * c + r * 3 * width] = inPixels[3 * (c + r) + r * 3 * width];
            outPixels[3 * c + r * 3 * width+1] = inPixels[3 * (c + r) + r * 3 * width+1];
            outPixels[3 * c + r * 3 * width+2] = inPixels[3 * (c + r) + r * 3 * width+2];
            outEdges[c + r * width] = inEdges[(c + r) + r * width];
          }
        else
          {
            outPixels[3 * c + r * 3 *width] = inPixels[3 * (c + r + 1) + r * 3 * width];
            outPixels[3 * c + r * 3 *width+1] = inPixels[3 * (c + r + 1) + r * 3 * width+1];
            outPixels[3 * c + r * 3 *width+2] = inPixels[3 * (c + r + 1) + r * 3 * width+2];
            outEdges[c + r * width] = inEdges[(c + r + 1) + r * width];
          }

        k += seam[r * width + k];
        r++;
    }
  }
}
void cutSeam(uint8_t *map, uint8_t *inPixels, uint8_t *outPixels, uint8_t *inEdges, uint8_t *outEdges,
   int width, int height, int* seam, dim3 blockSize = dim3(1,1))
{
  int k = 0;
  for (int c = 0; c < width; c++)
    {
      if (map[c]<map[k])
      {
        k = c;
      }
    }
  dim3 gridSize((width - 1)/blockSize.x + 1, (height - 1)/blockSize.y + 1);
  cutSeamKernel<<<gridSize, blockSize>>>(inPixels, outPixels, inEdges, outEdges, width, height, k, seam);
}

int main(int argc, char ** argv)
{
  GpuTimer timer;
  timer.Start();
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  printf("GPU name: %s\n", devProp.name);
  printf("GPU compute capability: %d.%d\n\n", devProp.major, devProp.minor);

	// Read input RGB image file
	int numChannels, width, height;
	uint8_t * inPixels;

	readPnm(argv[1], numChannels, width, height, inPixels);
	if (numChannels != 3)
		return EXIT_FAILURE; // Input image must be RGB
	printf("RGB Image size (width x height): %i x %i\n\n", width, height);

  //allocates device memories
  uint8_t *d_inPixels;
  uint8_t *d_inPixels2;
  uint8_t *d_grayPixels;
  float *d_filterX;
  float *d_filterY;
  uint8_t *d_edgePixels;
  uint8_t *d_edgePixels2;
  uint8_t *d_map;
  int *d_seam;
  CHECK(cudaMalloc(&d_inPixels, height * width * 3 * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_inPixels2, height * width * 3 * sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_grayPixels, height * width * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_filterX, 9 * sizeof(float)));
  CHECK(cudaMalloc(&d_filterY, 9 * sizeof(float)));
  CHECK(cudaMalloc(&d_edgePixels, width * height * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_edgePixels2, width * height * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_map, width * height * sizeof(uint8_t)));
  CHECK(cudaMalloc(&d_seam, width * height * sizeof(int)));

  // Set up filter xSobel, ySobel
	float filterX [9]= {1, 0, -1, 2, 0, -2, 1, 0, -1};
	float filterY [9]= {1, 2, 1, 0, 0, 0, -1, -2, -1};
  //copy memories to device
  CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_filterX, filterX, 9 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_filterY, filterY, 9 * sizeof(float), cudaMemcpyHostToDevice));
	// Convert RGB to grayscale using device
	dim3 blockSize(32, 32); // Default
  uint8_t * tmp = (uint8_t *)malloc(width * height * sizeof(uint8_t));

  // converse RGB image to Gray image
  convertRgb2Gray(d_inPixels, d_grayPixels, width, height, blockSize);
  // detect edge using device
  detectionEdge(d_grayPixels, d_edgePixels, width, height, d_filterX, d_filterY, 3, blockSize);
  uint8_t *map = (uint8_t*)malloc(width*sizeof(uint8_t));
	int *seam = (int*)malloc(width*height*sizeof(int));
  int nSeams = atoi(argv[3]);
  for (int i = 0; i < nSeams; i++)
  {
    if (i%2==0)
    {

      findSeam(d_edgePixels, d_map, d_seam, width, height, 1024);
      CHECK(cudaMemcpy(map, d_map, width * sizeof(uint8_t), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(seam, d_seam, width * height * sizeof(int), cudaMemcpyDeviceToHost));
			checkSeam(map, inPixels, width, height, seam);
			writePnm(inPixels, 3, width, height, argv[4]);
      cutSeam(map, d_inPixels, d_inPixels2, d_edgePixels, d_edgePixels2, width, height, d_seam, 1024);
    }
    else
    {
      //detectionEdge(d_grayPixels2, d_edgePixels, width, height, d_filterX, d_filterY, 3, blockSize);
      findSeam(d_edgePixels, d_map, d_seam, width, height, 1024);
      CHECK(cudaMemcpy(map, d_map, width * sizeof(uint8_t), cudaMemcpyDeviceToHost));
      cutSeam(map, d_inPixels2, d_inPixels, d_edgePixels2, d_edgePixels, width, height, d_seam, 1024);
    }
    width--;
  }
  uint8_t * outPixels = (uint8_t*)malloc(width*height*3*sizeof(uint8_t));
  if (nSeams%2==0)
    {
      CHECK(cudaMemcpy(outPixels, d_inPixels, width * 3 * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }
  else
    {
      CHECK(cudaMemcpy(outPixels, d_inPixels2, width * 3 * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }
  //printf("%i",width);
  writePnm(outPixels, 3, width, height, argv[2]);

  //printf("success");

	// Free memories
	//free(map);
  free(inPixels);
  cudaFree(d_inPixels);
  cudaFree(d_inPixels2);
  cudaFree(d_grayPixels);
  cudaFree(d_filterX);
  cudaFree(d_filterY);
  cudaFree(d_edgePixels);
  cudaFree(d_edgePixels2);
  cudaFree(d_map);
  cudaFree(d_seam);
}
