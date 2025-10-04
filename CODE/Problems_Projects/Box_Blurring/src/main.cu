#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iostream>
using namespace std;

void savePGM(unsigned char **matrix, int width, int height, const string &filename)
{
	ofstream file(filename, ios::binary);
	file << "P5\n"
		 << width << " " << height << "\n255\n";

	for (int i = 0; i < height; i++)
		file.write(reinterpret_cast<char *>(matrix[i]), width);

	file.close();
}

inline int clamp(int val, int minVal, int maxVal)
{
	if (val < minVal)
		return minVal;
	if (val > maxVal)
		return maxVal;
	return val;
}

void applyKernel(unsigned char **src, unsigned char **dst, int width, int height,
				 int kernel[10][10], int kSize)
{
	int offset = kSize / 2;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int sum = 0;

			// Apply kernel
			for (int ky = 0; ky < kSize; ky++)
			{
				for (int kx = 0; kx < kSize; kx++)
				{
					int ny = y + ky - offset;
					int nx = x + kx - offset;

					// Clamp to image boundaries
					ny = clamp(ny, 0, height - 1);
					nx = clamp(nx, 0, width - 1);

					sum += src[ny][nx] * kernel[ky][kx];
				}
			}

			// Normalize (sum of kernel values)
			int kernelSum = 0;
			for (int ky = 0; ky < kSize; ky++)
				for (int kx = 0; kx < kSize; kx++)
					kernelSum += kernel[ky][kx];

			if (kernelSum == 0)
				kernelSum = 1;
			dst[y][x] = sum / kernelSum;
		}
	}
}

int main()
{

	/*==========================================================================
	# Image Loading Part
	==========================================================================*/

	/* Load the pgm Image */
	ifstream file("images/globalProfilePic_LW_GS.pgm", ios::binary);
	string format;
	int width, height, maxVal;

	/* Read the Image header */
	file >> format >> width >> height >> maxVal;
	/* Skip a line */
	file.ignore(1);

	if (format != "P5")
	{
		cout << "Only binary PGM (P5) supported !" << endl;
		return 1;
	}
	else
	{
		cout << "Image Supported !!" << endl;
	}

	/* Allocate dynamic matrix for image pixel value */
	unsigned char **matrix = new unsigned char *[height];
	for (int i = 0; i < height; i++)
		matrix[i] = new unsigned char[width];

	unsigned char **blurred = new unsigned char *[height];
	for (int i = 0; i < height; i++)
		blurred[i] = new unsigned char[width];

	/* Read the actual Pixel values */
	for (int i = 0; i < height; i++)
		file.read(reinterpret_cast<char *>(matrix[i]), width);

	cout << "Image loaded into 2D array (" << height << "x" << width << ")" << endl;

	/*==========================================================================
	# Image Loading Part
	==========================================================================*/

	int kernel[10][10] = {
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

	applyKernel(matrix, blurred, width, height, kernel, 10);

	savePGM(blurred, width, height, "outputImage/blurred.pgm");

	/*==========================================================================
	# Cleaning Up Memory
	==========================================================================*/
	/* Free Memory */
	for (int i = 0; i < height; i++)
		delete[] matrix[i];
	delete[] matrix;

	for (int i = 0; i < height; i++)
		delete[] blurred[i];
	delete[] blurred;

	return 0;
}