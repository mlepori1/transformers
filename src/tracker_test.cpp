// Link statically with GLEW
//#define GLEW_STATIC

// Headers
#include "blockmodel.h"

#include "Eigen/Dense"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <png.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <chrono>

// Shader macro
// FIXME: this should be an inline function
#define GLSL(src) "#version 150 core\n" #src

using namespace Eigen;
using namespace std;

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

void writePng(const char* filename, png_byte* image, png_bytep* row_pointers, int width, int height)
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

    // Output is 8-bit depth, RGBA format.
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGBA,
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

void writeCsv(const char* filename, const vector<string>& colNames, const vector< vector<Vector3f> >& data)
{
    ofstream ofs(filename, ofstream::out);
    vector<string>::const_iterator it;
    for (it = colNames.begin(); it != colNames.end(); ++it)
    {
        ofs << *it << ",";
    }
    ofs << endl;
   
    // FIXME: assert all columns have the same number of entries
    for (int i = 0; i < data[0].size(); ++i)
    {
        vector< vector<Vector3f> >::const_iterator it;
        for (it = data.begin(); it != data.end(); ++it)
        {
            ofs << (*it)[i][0] << ",";
            ofs << (*it)[i][1] << ",";
            ofs << (*it)[i][2] << "\t";//",";
        }
        ofs << endl;
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


// Vertex shader
const GLchar* vertexShaderSrc = GLSL(
    in vec3 pos;
    in vec3 normal;

    out vec3 fColor;
    out vec3 fPos;
    out vec3 fNormal;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 proj;

    void main()
    {
        fPos = vec3(model * vec4(pos, 1.0f));
        gl_Position = proj * view * vec4(fPos, 1.0);
        fNormal = normal;
    }
);

// Fragment shader
const GLchar* fragmentShaderSrc = GLSL(
    in vec3 fColor;
    in vec3 fNormal;
    in vec3 fPos;

    out vec4 outColor;

    uniform vec3 objectColor;
    uniform vec3 lightColor;
    uniform vec3 lightPos;

    void main()
    {
        vec3 norm = normalize(fNormal);
        vec3 lightDir = normalize(lightPos - fPos);
        // FIXME: This seems fishy
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        float ambientStrength = 0.3f;
        vec3 ambient = ambientStrength * lightColor;

        vec3 result = (ambient + diffuse) * objectColor;
        outColor = vec4(result, 1.0);
    }
);

GLuint createShader(GLenum type, const GLchar* src)
{
    // Create and compile the vertex shader
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    // Check shader compilation status and compile logs
    GLint status;
    char buffer[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    glGetShaderInfoLog(shader, 512, NULL, buffer);
    printf("Shader compile status: %d\nLog: %s\n", status, buffer);

    return shader;
}

void APIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const void *userParam)
{
    printf("entered glDebugOutput");

    // ignore non-significant error/warning codes
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return; 
    
    cout << "---------------" << endl;
    cout << "Debug message (" << id << "): " <<  message << endl;
    
    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             cout << "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   cout << "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: cout << "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     cout << "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     cout << "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           cout << "Source: Other"; break;
    } cout << endl;
        
    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               cout << "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: cout << "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  cout << "Type: Undefined Behaviour"; break; 
        case GL_DEBUG_TYPE_PORTABILITY:         cout << "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         cout << "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              cout << "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          cout << "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           cout << "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               cout << "Type: Other"; break;
    } cout << endl;
        
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         cout << "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       cout << "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          cout << "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: cout << "Severity: notification"; break;
    } cout << endl;
    cout << endl;
}

void saveImage(const char* image_fn)
{
    // Save image
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int w = viewport[2];
    int h = viewport[3];
    glFinish();
    unsigned int bytes_per_row = 4 * w;
    png_byte* out_image = new png_byte[h * bytes_per_row];
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, out_image);
    png_bytep* row_pointers = new png_bytep[h];
    for (int row = 0; row < h; row++)
        row_pointers[h - 1 - row] = out_image + row * bytes_per_row;
    writePng(image_fn, out_image, row_pointers, w, h);
    delete[] row_pointers;
    delete[] out_image;
}

int main()
{
    auto t_start = chrono::high_resolution_clock::now();

    // Initialize a GLFW window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    GLFWwindow* window = glfwCreateWindow(160, 120, "OpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    glewInit();

    // Debug output
    GLint flags;
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS); 
        glDebugMessageCallback(glDebugOutput, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    }

    // Compile shaders and create shader programs
    GLuint vertexShader = createShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    glLinkProgram(shaderProgram);

    // Check linking status and logs
    GLint status;
    char buffer[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
    glGetShaderInfoLog(shaderProgram, 512, NULL, buffer);
    printf("Program link status: %d\nLog: %s\n", status, buffer);

    glUseProgram(shaderProgram);

    // Create Vertex Array Objects
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Create Vertex Buffer Objects and copy the vertex data to them
    GLuint vbo;
    glGenBuffers(1, &vbo);

    // Define vertices
    GLfloat w = 31.8;
    GLfloat l = 31.8;
    GLfloat h = 19.2;
    GLfloat vertices[] = {
    // square block
    //  position in R3, normal vector in R3
        -w/2, -l/2, -h/2, -1.0,  0.0,  0.0,
        -w/2,  l/2, -h/2, -1.0,  0.0,  0.0,
        -w/2, -l/2,  h/2, -1.0,  0.0,  0.0,
        -w/2,  l/2,  h/2, -1.0,  0.0,  0.0,
        -w/2,  l/2, -h/2, -1.0,  0.0,  0.0,
        -w/2, -l/2,  h/2, -1.0,  0.0,  0.0,

         w/2, -l/2, -h/2,  1.0,  0.0,  0.0,
         w/2,  l/2, -h/2,  1.0,  0.0,  0.0,
         w/2, -l/2,  h/2,  1.0,  0.0,  0.0,
         w/2,  l/2,  h/2,  1.0,  0.0,  0.0,
         w/2,  l/2, -h/2,  1.0,  0.0,  0.0,
         w/2, -l/2,  h/2,  1.0,  0.0,  0.0,

        -w/2, -l/2, -h/2,  0.0, -1.0,  0.0,
         w/2, -l/2, -h/2,  0.0, -1.0,  0.0,
        -w/2, -l/2,  h/2,  0.0, -1.0,  0.0,
         w/2, -l/2,  h/2,  0.0, -1.0,  0.0,
         w/2, -l/2, -h/2,  0.0, -1.0,  0.0,
        -w/2, -l/2,  h/2,  0.0, -1.0,  0.0,

        -w/2,  l/2, -h/2,  0.0,  1.0,  0.0,
         w/2,  l/2, -h/2,  0.0,  1.0,  0.0,
        -w/2,  l/2,  h/2,  0.0,  1.0,  0.0,
         w/2,  l/2,  h/2,  0.0,  1.0,  0.0,
         w/2,  l/2, -h/2,  0.0,  1.0,  0.0,
        -w/2,  l/2,  h/2,  0.0,  1.0,  0.0,

        -w/2, -l/2, -h/2,  0.0,  0.0, -1.0,
        -w/2,  l/2, -h/2,  0.0,  0.0, -1.0,
         w/2, -l/2, -h/2,  0.0,  0.0, -1.0,
         w/2,  l/2, -h/2,  0.0,  0.0, -1.0,
        -w/2,  l/2, -h/2,  0.0,  0.0, -1.0,
         w/2, -l/2, -h/2,  0.0,  0.0, -1.0,

        -w/2, -l/2,  h/2,  0.0,  0.0,  1.0,
        -w/2,  l/2,  h/2,  0.0,  0.0,  1.0,
         w/2, -l/2,  h/2,  0.0,  0.0,  1.0,
         w/2,  l/2,  h/2,  0.0,  0.0,  1.0,
        -w/2,  l/2,  h/2,  0.0,  0.0,  1.0,
         w/2, -l/2,  h/2,  0.0,  0.0,  1.0,
    //  Rectangular block
    //  position in R3, normal vector in R3
        -w/2, -l, -h/2, -1.0,  0.0,  0.0,
        -w/2,  l, -h/2, -1.0,  0.0,  0.0,
        -w/2, -l,  h/2, -1.0,  0.0,  0.0,
        -w/2,  l,  h/2, -1.0,  0.0,  0.0,
        -w/2,  l, -h/2, -1.0,  0.0,  0.0,
        -w/2, -l,  h/2, -1.0,  0.0,  0.0,

         w/2, -l, -h/2,  1.0,  0.0,  0.0,
         w/2,  l, -h/2,  1.0,  0.0,  0.0,
         w/2, -l,  h/2,  1.0,  0.0,  0.0,
         w/2,  l,  h/2,  1.0,  0.0,  0.0,
         w/2,  l, -h/2,  1.0,  0.0,  0.0,
         w/2, -l,  h/2,  1.0,  0.0,  0.0,

        -w/2, -l, -h/2,  0.0, -1.0,  0.0,
         w/2, -l, -h/2,  0.0, -1.0,  0.0,
        -w/2, -l,  h/2,  0.0, -1.0,  0.0,
         w/2, -l,  h/2,  0.0, -1.0,  0.0,
         w/2, -l, -h/2,  0.0, -1.0,  0.0,
        -w/2, -l,  h/2,  0.0, -1.0,  0.0,

        -w/2,  l, -h/2,  0.0,  1.0,  0.0,
         w/2,  l, -h/2,  0.0,  1.0,  0.0,
        -w/2,  l,  h/2,  0.0,  1.0,  0.0,
         w/2,  l,  h/2,  0.0,  1.0,  0.0,
         w/2,  l, -h/2,  0.0,  1.0,  0.0,
        -w/2,  l,  h/2,  0.0,  1.0,  0.0,

        -w/2, -l, -h/2,  0.0,  0.0, -1.0,
        -w/2,  l, -h/2,  0.0,  0.0, -1.0,
         w/2, -l, -h/2,  0.0,  0.0, -1.0,
         w/2,  l, -h/2,  0.0,  0.0, -1.0,
        -w/2,  l, -h/2,  0.0,  0.0, -1.0,
         w/2, -l, -h/2,  0.0,  0.0, -1.0,

        -w/2, -l,  h/2,  0.0,  0.0,  1.0,
        -w/2,  l,  h/2,  0.0,  0.0,  1.0,
         w/2, -l,  h/2,  0.0,  0.0,  1.0,
         w/2,  l,  h/2,  0.0,  0.0,  1.0,
        -w/2,  l,  h/2,  0.0,  0.0,  1.0,
         w/2, -l,  h/2,  0.0,  0.0,  1.0
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Specify the layout of the vertex data
    GLint posAttrib = glGetAttribLocation(shaderProgram, "pos");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), 0);

    GLint normAttrib = glGetAttribLocation(shaderProgram, "normal");
    glEnableVertexAttribArray(normAttrib);
    glVertexAttribPointer(normAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

    // Set up lighting uniforms
    GLint uniLightColor = glGetUniformLocation(shaderProgram, "lightColor");
    GLint uniLightPos = glGetUniformLocation(shaderProgram, "lightPos");
    glUniform3f(uniLightColor, 1.0f, 1.0f, 1.0f);
    glUniform3f(uniLightPos, 0.0f, 0.0f, 2000.0f);

    // Set up projection
    // Camera located 600mm above the origin, pointing along -z.
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, -50.0f, 600.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    GLint uniView = glGetUniformLocation(shaderProgram, "view");
    glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

    // I calculated field of view from camera intrinsic parameters
    // I set the near and far clipping planes ~arbitrarily~
    float field_of_view = 47.9975;  // degrees
    float near_plane = 100;     // mm
    float far_plane = 1000;     // mm
    glm::mat4 proj = glm::perspective(glm::radians(field_of_view), 800.0f / 600.0f, near_plane, far_plane);
    GLint uniProj = glGetUniformLocation(shaderProgram, "proj");
    glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

    // Generate vectors to hold simulation data
    vector<string> col_names;
    col_names.push_back("position (XYZ)");
    col_names.push_back("velocity (XYZ)");
    col_names.push_back("acceleration (XYZ)");
    col_names.push_back("orientation (XYZ)");
    col_names.push_back("angular velocity (XYZ)");
    vector< vector<Vector3f> > data;
    for (int i = 0; i < col_names.size(); ++i)
    {
        vector<Vector3f> col;
        data.push_back(col);
    }

    // Define block colors
    Vector3f red(1.0f, 0.0f, 0.0f);
    Vector3f green(0.0f, 1.0f, 0.0f);
    Vector3f blue(0.0f, 0.0f, 1.0f);
    Vector3f yellow(1.0f, 1.0f, 0.0f);

    // Initialize block model(s)
    Vector3f center(0.0f, 0.0f, 0.0f);
    float angle = 0.0f;
    float dt = 1.0 / 100.0;
    float rate = glm::radians(180.0f);  // pi radians per second
    float r = 100.0;
    Vector3f s0 = center + Vector3f(r * cos(angle), r * sin(angle), 0.0f);
    Vector3f v0(-r * rate * sin(angle), r * rate * cos(angle), 0.0f);
    Vector3f theta0(0.0f, 0.0f, 0.0f);
    Vector3f a_g(0.0f, 0.0f, 9810.0f);   // millimeters / second^2
    BlockModel block(s0, v0, theta0, red, a_g);

    const vector<string> imagePaths = readImagePaths("../working/gl-render/png-files.txt");
    int num_frames = imagePaths.size();
    int field_width = ceil(log(float(num_frames)) / log(10.0f)) + 1;

    GLint uniModel = glGetUniformLocation(shaderProgram, "model");
    GLint uniObjectColor = glGetUniformLocation(shaderProgram, "objectColor");

    int frame = 0;
    while (!glfwWindowShouldClose(window))
    //for (int frame = 0; frame < num_frames; frame++)
    {
        /*
        // Read the current frame
        int frame_width, frame_height;
        png_byte* image = readPng(imagePaths[frame].c_str(), frame_width, frame_height);
        */

        // Enable depth rendering, clear the screen to black and clear depth buffer
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // State at current time
        Vector3f s_t = block.getPosition();
        Vector3f v_t = block.getVelocity();
        Vector3f theta_t = block.getOrientation();

        // Input at current time
        //auto t_now = chrono::high_resolution_clock::now();
        //float time = chrono::duration_cast< chrono::duration<float> >(t_now - t_start).count();
        Vector3f a_t(-r * rate * rate * cos(rate * frame * dt),
                     -r * rate * rate * sin(rate * frame * dt),
                     0.0f);
        Vector3f omega_t(0.0f, 0.0f, 0.0f);
        
        // Store data
        data[0].push_back(s_t);
        data[1].push_back(v_t);
        data[2].push_back(a_t);
        data[3].push_back(theta_t);
        data[4].push_back(omega_t);

        // Render scene
        block.draw(uniModel, uniObjectColor, 0);

        // Save the image
        string frame_str = to_string(frame);
        string fn = "../working/gl-render/" + frame_str.insert(0, field_width - frame_str.size(), '0') + ".png";
        saveImage(fn.c_str());

        // Take a step in state space
        block.updateState(a_t + a_g, omega_t, dt);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        ++frame;
    }

    writeCsv("../working/gl-data/debug_output.csv", col_names, data);

    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    glfwTerminate();

    return 0;
}
