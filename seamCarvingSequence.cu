#include <stdio.h>
#include <stdint.h>

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

void setValAndPostionEnergy(uint8_t * inPixels, int *** energyTable, int rowImg, int colImg, int width, int height, int &energy, int &position)
{
    int energy_tmp = energyTable[rowImg + 1][colImg][0];
    int position_tmp = (rowImg + 1) * width + colImg;
    if (colImg - 1 >= 0)
    {
        if (energy_tmp > energyTable[rowImg + 1][colImg - 1][0])
        {
            energy_tmp = energyTable[rowImg + 1][colImg - 1][0];
            position_tmp = (rowImg + 1) * width + colImg - 1;
        }
    }
    if (colImg + 1 < height)
    {
        if (energy_tmp > energyTable[rowImg + 1][colImg + 1][0])
        {
            energy_tmp = energyTable[rowImg + 1][colImg + 1][0];
            position_tmp = (rowImg + 1) * width + colImg + 1;
        }
    }
    energy = energy_tmp + inPixels[rowImg * width + colImg];
    position = position_tmp;
}

void convertRgb2Gray(uchar3 * inPixels, int width, int height, uint8_t * &outPixels)
{
    // Reminder: gray = 0.299*red + 0.587*green + 0.114*blue
    for (int rowImg = 0; rowImg < height; rowImg++)
    {
        for (int colImg = 0; colImg < width; colImg++)
        {
            int idx = rowImg * width + colImg;
            uint8_t red = inPixels[idx].x;
            uint8_t green = inPixels[idx].y;
            uint8_t blue = inPixels[idx].z;
            outPixels[idx] = 0.299f*red + 0.587f*green + 0.114f*blue;
        }
    }
}

void detectEdgeImg(uint8_t * inPixels, int width, int height, uint8_t * &outPixels)
{
    int filterWidth = 3;
    int threshold = 16; // Set threshold for filter noise with 6.25%
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

            outPixels[rowImg * width + colImg] = abs(xSobel) + abs(ySobel);
            if (outPixels[rowImg * width + colImg] < threshold)
            {
                outPixels[rowImg * width + colImg] = 0;
            }
        }
    }
}

void findSeamCarving(uint8_t * inPixels, int width, int height, int * &traces)
{
    int *** energyTable = (int ***)malloc(height * sizeof(int **));
    for (int row = 0; row < height; row++)
    {
        energyTable[row] = (int **)malloc(width * sizeof(int *));
        for (int col = 0; col < width; col++)
        {
            energyTable[row][col] = (int *)malloc(2 * sizeof(int));
        }
    }
    for (int colImg = 0; colImg < width; colImg++)
    {
        energyTable[height - 1][colImg][0] = inPixels[(height - 1) * width + colImg];
        energyTable[height - 1][colImg][1] = -1;
    }
    for (int rowImg = height - 2; rowImg >= 0; rowImg--)
    {
        for (int colImg = 0; colImg < width; colImg++)
        {
            setValAndPostionEnergy(inPixels, energyTable, rowImg, colImg, width, height, energyTable[rowImg][colImg][0], energyTable[rowImg][colImg][1]);
        }
    }

    int minEnergy = energyTable[0][0][0];
    int minPostion = 0;
    int index = 0;
    for (int col = 0; col < width; col++)
    {
        if (minEnergy > energyTable[0][col][0])
        {
            minEnergy = energyTable[0][col][0];
            minPostion = col;
        }
    }

    while (index < height)
    {
        traces[index] = minPostion;
        minPostion = energyTable[minPostion / width][minPostion % width][1];
        index++;
    }

    // free energy tables
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            free(energyTable[row][col]);
        }
        free(energyTable[row]);
    }
    free(energyTable);
}

void cutSeamCarvingImg(uint8_t * inPixels, int width, int height, int * traces)
{
    for (int row = height - 1; row >= 0; row--)
    {
        for (int idx = traces[row]; idx < width * height - 1; idx++)
        {
            inPixels[idx] = inPixels[idx + 1];
        }
    }
}

void seamCarvingImg(uchar3 * inPixels, int width, int height, uchar3 * &outPixels, int size)
{
    int maxCol = width;
    int maxRow = height;
    uint8_t * grayOutPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    uint8_t * edgeOutPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    int ** traces = (int **)malloc(size * sizeof(int *));
    for (int loop = 0; loop < size; loop++)
    {
        traces[loop] = (int *)malloc(height * sizeof(int));
    }
    convertRgb2Gray(inPixels, width, height, grayOutPixels);
    detectEdgeImg(grayOutPixels, width, height, edgeOutPixels);
    for (int loop = 0; loop < size; loop++)
    {
        findSeamCarving(edgeOutPixels, maxCol, maxRow, traces[loop]);
        cutSeamCarvingImg(edgeOutPixels, maxCol, maxRow, traces[loop]);
        maxCol--;
    }

    for (int rowImg = 0; rowImg < height; rowImg++)
    {
        for (int colImg = 0; colImg < width; colImg++)
        {
            outPixels[rowImg * width + colImg].x = inPixels[rowImg * width + colImg].x;
            outPixels[rowImg * width + colImg].y = inPixels[rowImg * width + colImg].y;
            outPixels[rowImg * width + colImg].z = inPixels[rowImg * width + colImg].z;
        }
    }

    for (int loop = 0; loop < size; loop++)
    {
        for (int rowImg = 0; rowImg < height; rowImg++)
        {
            int col = traces[loop][rowImg] % (width - loop) + loop;
            int idx = rowImg * width + col;
            outPixels[idx].x = 255;
            outPixels[idx].y = 0;
            outPixels[idx].z = 0;
        }
    }

    // Free memories
    for (int loop = 0; loop < size; loop++)
    {
        free(traces[loop]);
    }
    free(traces);
    free(edgeOutPixels);
    free(grayOutPixels);
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
    
    // Seam Carving input image using host
    uchar3 * seamCarvingOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
    seamCarvingImg(inPixels, width, height, seamCarvingOutPixels, size);

    // Write results to files
    char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(seamCarvingOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));

    // Free memories
    free(seamCarvingOutPixels);
}