#include <stdio.h>
#include <stdint.h>
#include <time.h>

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

struct CpuTimer
{
    clock_t start;
    clock_t end;

    CpuTimer()
    {
        // Constructor
    }

    ~CpuTimer()
    {
        // De-constructor
    }

    void Start()
    {
        start = clock();
    }

    void Stop()
    {
        end = clock();
    }

    double Elapsed()
    {
        return ((double) (end - start)) / CLOCKS_PER_SEC;
    }
};

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

struct EngeryPoint
{
    int val;        // Energy of current point.
    int prePos;     // Postion of pre enegry.
};

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
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

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
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

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

void setValAndPos(uint8_t * inPixels, EngeryPoint * energies, int rowImg, int colImg, int width, int height)
{
    int energy = energies[(rowImg + 1) * width + colImg].val;
    int pos_tmp = (rowImg + 1) * width + colImg;
    if (colImg - 1 >= 0)
    {
        if (energy > energies[(rowImg + 1) * width + colImg - 1].val)
        {
            energy = energies[(rowImg + 1) * width + colImg - 1].val;
            pos_tmp = (rowImg + 1) * width + colImg - 1;
        }
    }
    if (colImg + 1 < height)
    {
        if (energy > energies[(rowImg + 1) * width + colImg + 1].val)
        {
            energy = energies[(rowImg + 1) * width + colImg + 1].val;
            pos_tmp = (rowImg + 1) * width + colImg + 1;
        }
    }
    energies[rowImg * width + colImg].val = energy + inPixels[rowImg * width + colImg];
    energies[rowImg * width + colImg].prePos = pos_tmp;
}

__global__ void convertRgb2GrayKernel(uchar3 * inPixels, int width, int height, uint8_t * outPixels)
{
    int outPixelsR = blockIdx.x * blockDim.x + threadIdx.y;
	int outPixelsC = blockIdx.y * blockDim.y + threadIdx.x;
    // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue
    if (outPixelsR < height && outPixelsC < width)
    {
        uint8_t red = inPixels[outPixelsR * width + outPixelsC].x;
        uint8_t green = inPixels[outPixelsR * width + outPixelsC].y;
        uint8_t blue = inPixels[outPixelsR * width + outPixelsC].z;
        outPixels[outPixelsR * width + outPixelsC] = 0.299f*red + 0.587f*green + 0.114f*blue;
    }
}

__global__ void detectEdgeImgkernel(uint8_t * inPixels, int width, int height, float * xFilter, float * yFilter, int filterWidth, uint8_t * outPixels)
{
    extern __shared__ uint8_t s_inPixels[];
	int outPixelsR = blockIdx.x * blockDim.x + threadIdx.y;
    int outPixelsC = blockIdx.y * blockDim.y + threadIdx.x;
    
    if (outPixelsC < width && outPixelsR < height)
    {
        float xSobel = 0;
        float ySobel = 0;
        for (int filterC = 0; filterC < filterWidth; filterC++)
        {
            for (int filterR = 0; filterR < filterWidth; filterR++)
            {
                float xFilterVal = xFilter[filterR * filterWidth + filterC] / 4;
                float yFilterVal = yFilter[filterR * filterWidth + filterC] / 4;
                int inPixelsC = outPixelsC + filterC - filterWidth / 2;
                int inPixelsR = outPixelsR + filterR - filterWidth / 2;
                inPixelsC = min(max(inPixelsC, 0), width - 1);
                inPixelsR = min(max(inPixelsR, 0), height - 1);
                uint8_t inPixel = inPixels[inPixelsR * width + inPixelsC];
                xSobel += inPixel * xFilterVal;
                ySobel += inPixel * yFilterVal;
            }
        }
        outPixels[outPixelsR * width + outPixelsC] = sqrt(xSobel * xSobel + ySobel * ySobel);
    }
}

void detectEdgeImg(uint8_t * inPixels, int width, int height, uint8_t * outPixels)
{
    int filterWidth = 3;
    float xFilter[filterWidth * filterWidth] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    float yFilter[filterWidth * filterWidth] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    for (int rowImg = 0; rowImg < height; rowImg++)
    {
        for (int colImg = 0; colImg < width; colImg++)
        {
            float xSobel = 0;
            float ySobel = 0;
            for (int colFilter = 0; colFilter < filterWidth; colFilter++)
            {
                for (int rowFilter = 0; rowFilter < filterWidth; rowFilter++)
                {
                    float xFilterVal = xFilter[rowFilter * filterWidth + colFilter] / 4;
                    float yFilterVal = yFilter[rowFilter * filterWidth + colFilter] / 4;
                    int inRowImg = rowImg + rowFilter - filterWidth / 2;
                    int inColImg = colImg + colFilter - filterWidth / 2;
                    inRowImg = min(max(inRowImg, 0), height - 1);
                    inColImg = min(max(inColImg, 0), width - 1);
                    xSobel += xFilterVal * inPixels[inRowImg * width + inColImg];
                    ySobel += yFilterVal * inPixels[inRowImg * width + inColImg];
                }
            }

            outPixels[rowImg * width + colImg] = sqrt(xSobel * xSobel + ySobel * ySobel);
        }
    }
}

void findSeamCarving(uint8_t * inPixels, int width, int height, int * traces)
{
    EngeryPoint * energies = (EngeryPoint *)malloc(width * height * sizeof(EngeryPoint));
    for (int colImg = 0; colImg < width; colImg++)
    {
        energies[(height - 1) * width + colImg].val = inPixels[(height - 1) * width + colImg];
        energies[(height - 1) * width + colImg].prePos = -1;
    }
    for (int rowImg = height - 2; rowImg >= 0; rowImg--)
    {
        for (int colImg = 0; colImg < width; colImg++)
        {
            setValAndPos(inPixels, energies, rowImg, colImg, width, height);
        }
    }

    int firstEgy = energies[0].val;
    int firstPos = 0;
    int index = 0;
    for (int col = 0; col < width; col++)
    {
        if (firstEgy > energies[col].val)
        {
            firstEgy = energies[col].val;
            firstPos = col;
        }
    }

    while (index < height)
    {
        traces[index] = firstPos;
        firstPos = energies[firstPos].prePos;
        index++;
    }

    // free energy tables
    free(energies);
}

void cutSeamCarvingImg(uint8_t * inPixels, int width, int height, int * traces)
{
    for (int row = height - 1; row >= 0; row--)
    {
        for (int idx = traces[row]; idx < width * height + height - 2 - row; idx++)
        {
            inPixels[idx] = inPixels[idx + 1];
        }
    }
}

void cutSeamCarvingRGBImg(uchar3 * inPixels, int width, int height, int * traces)
{
    for (int row = height - 1; row >= 0; row--)
    {
        for (int idx = traces[row]; idx < width * height - 1; idx++)
        {
            inPixels[idx].x = inPixels[idx + 1].x;
            inPixels[idx].y = inPixels[idx + 1].y;
            inPixels[idx].z = inPixels[idx + 1].z;
        }
    }
}

void seamCarvingImg(uchar3 * inPixels, int width, int height, uchar3 * outPixels, int size)
{
    CpuTimer timer;
    timer.Start();
    int maxCol = width;
    int maxRow = height;
    uint8_t * edgeOutPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    int * traces = (int *)malloc(height * sizeof(int));
    /*---- This is block convert RGP to gray with parallel ----*/
    GpuTimer timerGpu;
    uchar3 * d_inPixels;
    uint8_t * d_outGrayPixels;
    CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
    CHECK(cudaMalloc(&d_outGrayPixels, width * height * sizeof(uint8_t)));
    CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), cudaMemcpyHostToDevice));
    // Call kernel
    dim3 blockSize(32, 32);
    dim3 gridSize((height-1)/blockSize.x + 1, (width-1)/blockSize.y + 1);
    printf("block size %ix%i, grid size %ix%i\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);
    timerGpu.Start();
    convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels, width, height, d_outGrayPixels);
    timerGpu.Stop();
    float timekernel = timerGpu.Elapsed();
	printf("ConvertRGP2Gray kernel time: %f ms\n", timekernel);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    CHECK(cudaFree(d_inPixels));
    /*---- Ended block doing convert RGP to gray with parallel ---*/
    /*---- This is block processing detect Edge Img parallel ----*/
    int filterWidth = 3;
    float xFilter[filterWidth * filterWidth] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    float yFilter[filterWidth * filterWidth] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    float * d_xFilter, * d_yFilter;
    uint8_t * d_outEdgePixels;
    CHECK(cudaMalloc(&d_outEdgePixels, width * height * sizeof(uint8_t)));
    CHECK(cudaMalloc(&d_xFilter, filterWidth * filterWidth * sizeof(float)));
    CHECK(cudaMalloc(&d_yFilter, filterWidth * filterWidth * sizeof(float)));
    CHECK(cudaMemcpy(d_xFilter, xFilter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_yFilter, yFilter, filterWidth * filterWidth * sizeof(float), cudaMemcpyHostToDevice));
    timerGpu.Start();
    detectEdgeImgkernel<<<gridSize, blockSize>>>(d_outGrayPixels, width, height, d_xFilter, d_yFilter, filterWidth, d_outEdgePixels);
    timerGpu.Stop();
    timekernel = timerGpu.Elapsed();
	printf("DetectEgdeImg kernel time: %f ms\n", timekernel);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(edgeOutPixels, d_outEdgePixels, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_xFilter));
    CHECK(cudaFree(d_yFilter));
    CHECK(cudaFree(d_outEdgePixels));
    CHECK(cudaFree(d_outGrayPixels));
    /*---- Ended block doing detect Edge Img parallel ----*/
    for (int loop = 0; loop < size; loop++)
    {
        findSeamCarving(edgeOutPixels, maxCol, maxRow, traces);
        cutSeamCarvingImg(edgeOutPixels, maxCol, maxRow, traces);
        cutSeamCarvingRGBImg(inPixels, maxCol, maxRow, traces);
        maxCol--;
    }

    for (int idx = 0; idx < maxCol * height; idx++)
    {
        outPixels[idx].x = inPixels[idx].x;
        outPixels[idx].y = inPixels[idx].y;
        outPixels[idx].z = inPixels[idx].z;
    }

    // Free memories
    free(traces);
    free(edgeOutPixels);
    timer.Stop();
    double time = timer.Elapsed();
    printf("Processing time (use host): %f s\n\n", time);
}

int main(int argc, char ** argv)
{
    if (argc != 3 && argc != 4)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
    }

    int size = 1;
    if (argc == 4)
    {
        size = atoi(argv[3]);
    }
    
    // Read input image file
	int width, height;
    uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
    printf("Image size (width x height): %i x %i\n\n", width, height);
    if (width < size)
    {
        printf("The width of image less than size!\n");
        return EXIT_FAILURE;
    }
    
    // Seam Carving input image using device
    uchar3 * seamCarvingOutPixels = (uchar3 *)malloc((width - size) * height * sizeof(uchar3));
    seamCarvingImg(inPixels, width, height, seamCarvingOutPixels, size);

    // Write results to files
    char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(seamCarvingOutPixels, width - size, height, concatStr(outFileNameBase, "_device.pnm"));

    // Free memories
    free(seamCarvingOutPixels);
    free(inPixels);
}