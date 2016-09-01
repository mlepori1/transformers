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


struct configParams
{
    float w0;
    float dt;
    float q;
    float r;

    bool observe_image;
    bool observe_position;
    bool observe_velocity;
    bool observe_orientation;

    string u_path;
    string x_path;
    string y_path;

    bool save_rgb;
    bool save_depth;
};

png_byte* readPng(const char* filename, int& width, int& height);


void writePng(const char* filename, png_byte* image, const int width,
        const int height, const int color_type);


VectorXf toVector(png_byte* image_bytes, int num_bytes);


void toByteArray(VectorXf image, png_byte* image_bytes);


vector<VectorXf> readCsv(const char* filename);


void writeCsv(const char* filename, const vector<string>& col_names,
        const vector<VectorXf>& data);


const vector<string> readImagePaths(const char* filename);


void writeImagePaths(const char* filename, vector<string> image_paths);


void parseConfigFile(const char* config_fn, configParams & params);
