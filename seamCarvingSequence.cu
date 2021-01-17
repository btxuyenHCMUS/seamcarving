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

void writePnm(uint8_t * pixels, int numChannels, int width, int height, char * fileName)
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

    for (int i = 0; i < width * height; i++)
        fprintf(f, "%hhu\n", pixels[i]);

    fclose(f);
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

void detectEdgImg(uint8_t * inPixels, int width, int height, uint8_t * &outPixels)
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

void seamCarvingImg()
{

}

char * concatStr(const char * s1, const char * s2)
{
	char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
	strcpy(result, s1);
	strcat(result, s2);
	return result;
}

int main(int argc, char ** argv)
{
    if (argc != 3)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
    }
    
    // Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
    printf("Image size (width x height): %i x %i\n\n", width, height);
    
    // Seam Carving input image using host
    // TODO:
    uint8_t * grayOutPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    uint8_t * edgeOutPixels = (uint8_t *)malloc(width * height * sizeof(uint8_t));
    convertRgb2Gray(inPixels, width, height, grayOutPixels);
    detectEdgImg(grayOutPixels, width, height, edgeOutPixels);

    // Write results to files
    char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
    writePnm(edgeOutPixels, 1, width, height, concatStr(outFileNameBase, "_host.pnm"));

    // Free memories
    free(edgeOutPixels);
    free(grayOutPixels);
}