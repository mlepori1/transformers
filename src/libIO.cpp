/**
 *
 */


#include "libIO.h"

using namespace std;
using namespace Eigen;


png_byte* readPng(const char* filename, int& width, int& height)
{
    /*
     * Copied with minor edits from https://gist.github.com/niw/5963798
     */

    FILE *fp = fopen(filename, "rb");

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    width      = png_get_image_width(png, info);
    height     = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    // Read any color_type into 8-bit depth, RGBA format
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if (bit_depth == 16)
        png_set_strip_16(png);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if (png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_types don't have an alpha channel, so fill it with 0xff
    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    // Form row pointers and read image
    unsigned int bytes_per_row = png_get_rowbytes(png, info);
    png_byte* image = new png_byte[bytes_per_row * height];
    png_bytep* row_pointers = new png_bytep[height];
    for (int row = 0; row < height; row++)
        row_pointers[row] = image + row * bytes_per_row;
    png_read_image(png, row_pointers);
    delete[] row_pointers;

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
    png = NULL;
    info = NULL;

    return image;
}


void writePng(const char* filename, png_byte* image, png_bytep* row_pointers,
        int width, int height)
{
    /*
     * Copied with minor edits from https://gist.github.com/niw/5963798
     */

    FILE *fp = fopen(filename, "wb");
    if (!fp) abort();
            
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    // Output is 8-bit depth, RGB format.
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
            PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
   
    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    //png_set_filler(png, 0, PNG_FILLER_AFTER);
    
    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    if (png && info)
        png_destroy_write_struct(&png, &info);
     
    fclose(fp);
}


vector< vector<float> > readCsv(const char* filename)
{
    ifstream ifs(filename, ifstream::in);
    string line;
    getline(ifs, line);

    // Initialize one column for each column name in the CSV file
    stringstream lineStream(line);
    string entry;
    vector< vector<float> > cols;
    while (getline(lineStream, entry, ','))
    {
        vector<float> col;
        cols.push_back(col);
    }

    // Read the CSV data as floats
    while (getline(ifs, line))
    {
        stringstream lineStream(line);
        string entry;
        int i = 0;
        while (getline(lineStream, entry, ','))
        {
            cols[i].push_back(stof(entry));
            i++;
        }
    }
    ifs.close();

    return cols;
}


void writeCsv(const char* filename, const vector<string>& colNames,
        const vector< vector<VectorXf> >& data)
{
    ofstream ofs(filename, ofstream::out);
    vector<string>::const_iterator it;
    for (it = colNames.begin(); it != colNames.end(); ++it)
    {
        ofs << *it << ",";
    }
    ofs << endl;

    IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "\n");
    // FIXME: assert all columns have the same number of entries
    for (int i = 0; i < data.size(); ++i)
    {
        for (int j = 0; j < data[i].size(); ++j)
        {
            ofs << data[i][j].transpose() << endl;
        }
    }

    ofs.close();
}


const vector<string> readImagePaths(const char* filename)
{
    ifstream ifs(filename, ifstream::in);
    string line;
    getline(ifs, line);

    vector<string> imagePaths;
    // Read the CSV data as floats
    while (getline(ifs, line))
    {
        imagePaths.push_back(line);
    }
    ifs.close();

    return imagePaths;
}


void saveImage(const char* image_fn)
{
    // Save image
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int w = viewport[2];
    int h = viewport[3];

    glFinish();

    unsigned int bytes_per_row = 3 * w;
    png_byte* out_image = new png_byte[h * bytes_per_row];
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, out_image);

    png_bytep* row_pointers = new png_bytep[h];
    for (int row = 0; row < h; row++)
        row_pointers[h - 1 - row] = out_image + row * bytes_per_row;
    writePng(image_fn, out_image, row_pointers, w, h);
    delete[] row_pointers;
    delete[] out_image;
}
