/**
 *
 */

#include "libIO.h"
#include "assert.h"

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

    // Read any color_type into 8-bit depth, RGB format
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

    if (color_type & PNG_COLOR_MASK_ALPHA)
        png_set_strip_alpha(png);

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

    //cout << bytes_per_row << endl;

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
    png = NULL;
    info = NULL;

    return image;
}

Map<VectorXf> toVector(png_byte* image_bytes, int num_bytes)
{
    // Convert to vector
    float* image_floats = new float[num_bytes];
    memcpy(image_floats, image_bytes, num_bytes);
    Map<VectorXf> image(image_floats, num_bytes);

    delete[] image_bytes;
    delete[] image_floats;
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


vector<VectorXf> readCsv(const char* filename)
{
    ifstream ifs(filename, ifstream::in);
    string line;
    getline(ifs, line);

    // Count the number of columns in the CSV file
    stringstream lineStream(line);
    string entry;
    int num_cols = 0;
    while (getline(lineStream, entry, ','))
    {
        ++num_cols;
    }

    // Read the CSV data as floats
    vector<VectorXf> data;
    while (getline(ifs, line))
    {
        stringstream lineStream(line);
        string entry;
        int cur_col = 0;
        VectorXf row(num_cols);
        while (getline(lineStream, entry, ','))
        {
            row(cur_col) = stof(entry);
            ++cur_col;
        }

        // If we haven't read num_cols by now, something's wrong
        assert(cur_col == num_cols);

        data.push_back(row);
    }

    ifs.close();

    return data;
}


void writeCsv(const char* filename, const vector<string>& col_names,
        const vector<VectorXf>& data)
{
    ofstream ofs(filename, ofstream::out);
    vector<string>::const_iterator str_it;
    for (str_it = col_names.begin(); str_it != col_names.end(); ++str_it)
    {
        ofs << *str_it << ",";
    }
    ofs << endl;

    // FIXME: assert all vectors have the same number of entries
    const static IOFormat csv_format(StreamPrecision, DontAlignCols, ",", "\n");
    vector<VectorXf>::const_iterator vec_it;
    for (vec_it = data.begin(); vec_it != data.end(); ++vec_it)
    {
        ofs << vec_it->transpose().format(csv_format) << endl;
    }

    ofs.close();
}


const vector<string> readImagePaths(const char* filename)
{
    ifstream ifs(filename, ifstream::in);
    string line;

    vector<string> image_paths;
    while (getline(ifs, line))
    {
        image_paths.push_back(line);
    }

    ifs.close();

    return image_paths;
}


void writeImagePaths(const char* filename, vector<string> image_paths)
{
    ofstream ofs(filename, ofstream::out);

    vector<string>::const_iterator it;
    for (it = image_paths.begin(); it != image_paths.end(); ++it)
    {
        ofs << *it << endl;
    }

    ofs.close();
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
