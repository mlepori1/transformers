/**
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>

#include <png.h>

#include <GL/glew.h>

#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

png_byte* readPng(const char* filename, int& width, int& height);


void writePng(const char* filename, png_byte* image, png_bytep* row_pointers,
        int width, int height);


vector< vector<float> > readCsv(const char* filename);


void writeCsv(const char* filename, const vector<string>& colNames,
        const vector< vector<VectorXf> >& data);


const vector<string> readImagePaths(const char* filename);


void saveImage(const char* image_fn);
