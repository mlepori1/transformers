// Link statically with GLEW
//#define GLEW_STATIC

// Headers
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

std::vector< std::vector<float> > readCsv(const char* filename)
{
    std::ifstream ifs(filename, std::ifstream::in);
    std::string line;
    std::getline(ifs, line);

    // Initialize one column for each column name in the CSV file
    std::stringstream lineStream(line);
    std::string entry;
    std::vector< std::vector<float> > cols;
    while (std::getline(lineStream, entry, ','))
    {
        std::vector<float> col;
        cols.push_back(col);
    }

    // Read the CSV data as floats
    while (std::getline(ifs, line))
    {
        std::stringstream lineStream(line);
        std::string entry;
        int i = 0;
        while (std::getline(lineStream, entry, ','))
        {
            // DEBUG
            //std::cout << entry << "  ";

            cols[i].push_back(std::stof(entry));
            i++;
        }
        // DEBUG
        //std::cout <<std::endl;
    }
    ifs.close();

    return cols;
}

void writeCsv(const char* filename, const std::vector<std::string>& colNames, const std::vector< std::vector<float> >& data)
{
    std::ofstream ofs(filename, std::ofstream::out);
    std::vector<std::string>::const_iterator it;
    for (it = colNames.begin(); it != colNames.end(); ++it)
    {
        ofs << *it << ",";
    }
    ofs << std::endl;
   
    // FIXME: assert all columns have the same number of entries
    for (int i = 0; i < data[0].size(); ++i)
    {
        std::vector< std::vector<float> >::const_iterator it;
        for (it = data.begin(); it != data.end(); ++it)
        {
            ofs << (*it)[i] << ",";
        }
        ofs << std::endl;
    }

    ofs.close();
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
    
    std::cout << "---------------" << std::endl;
    std::cout << "Debug message (" << id << "): " <<  message << std::endl;
    
    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
    } std::cout << std::endl;
        
    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break; 
        case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
    } std::cout << std::endl;
        
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
    } std::cout << std::endl;
    std::cout << std::endl;
}

int main()
{
    auto t_start = std::chrono::high_resolution_clock::now();

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

    /*
    // Transform feedback buffer for debugging
    const GLchar* feedbackVaryings[] = { "transformFeedback" };
    glTransformFeedbackVaryings(shaderProgram, 1, feedbackVaryings, GL_INTERLEAVED_ATTRIBS);
    */

    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // Create Vertex Array Objects
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Create Vertex Buffer Objects and copy the vertex data to them
    GLuint vbo;
    glGenBuffers(1, &vbo);

    // Define vertices
    GLfloat w  = 31.8;
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

    /*
    // Create transform feedback buffer
    GLuint tbo;
    glGenBuffers(1, &tbo);
    glBindBuffer(GL_ARRAY_BUFFER, tbo);
    glBufferData(GL_ARRAY_BUFFER, 4 * 16 * sizeof(GLfloat), NULL, GL_STATIC_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo);
    GLfloat feedback[4 * 16];
    glBindBuffer(GL_ARRAY_BUFFER, vboSquare);
    */

    // Set up projection
    // Camera located 600mm above the origin, pointing along -z.
    // Downward direction is... +z??
    glm::mat4 view = glm::lookAt(
        glm::vec3(50.0f, 50.0f, 600.0f),
        //glm::vec3(100.0f, 100.0f, 100.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    GLint uniView = glGetUniformLocation(shaderProgram, "view");
    glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

    /*
    printf("View matrix:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", view[i][j]);
        }
        printf("\n");
    }
    */
    
    // I calculated field of view from camera intrinsic parameters
    // I set the near and far clipping planes ~arbitrarily~
    float field_of_view = 47.9975;  // degrees
    float near_plane = 100;     // mm
    float far_plane = 1000;     // mm
    glm::mat4 proj = glm::perspective(glm::radians(field_of_view), 800.0f / 600.0f, near_plane, far_plane);
    GLint uniProj = glGetUniformLocation(shaderProgram, "proj");
    glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

    /*
    printf("Projection matrix:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", proj[i][j]);
        }
        printf("\n");
    }
    */

    GLint uniModel = glGetUniformLocation(shaderProgram, "model");
    GLint uniObjectColor = glGetUniformLocation(shaderProgram, "objectColor");

    glm::vec3 blockColors[4] = {
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f),
        glm::vec3(1.0f, 1.0f, 0.0f)
    };  // rgb
    glm::vec3 squareTranslations[4] = {
        glm::vec3(-150.0f, 0.0f, 0.0f),
        glm::vec3( -50.0f, 0.0f, 0.0f),
        glm::vec3(  50.0f, 0.0f, 0.0f),
        glm::vec3( 150.0f, 0.0f, 0.0f)
    };  // mm
    glm::vec3 rectTranslations[4] = {
        glm::vec3(0.0f, -150.0f, 0.0f),
        glm::vec3(0.0f,  -50.0f, 0.0f),
        glm::vec3(0.0f,   50.0f, 0.0f),
        glm::vec3(0.0f,  150.0f, 0.0f)
    };  // mm
    glm::vec3 squareRotations[4] = {
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 0.0f)
    };  // degrees
    glm::vec3 rectRotations[4] = {
        glm::vec3(0.0f, 0.0f, 90.0f),
        glm::vec3(0.0f, 0.0f,  0.0f),
        glm::vec3(0.0f, 0.0f,  0.0f),
        glm::vec3(0.0f, 0.0f, 90.0f)
    };  // degrees
   
    std::vector< std::vector<float> > data;
    for (int i = 0; i < 8 * 6; ++i)
    {
        std::vector<float> col;
        data.push_back(col);
    }
    std::vector<std::string> names(data.size(), "X");

    int num_frames = 100;
    for (int frame = 0; frame < num_frames; frame++)
    //while (!glfwWindowShouldClose(window))
    {
        glEnable(GL_DEPTH_TEST);
        // Clear the screen to black and clear depth buffer
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Calculate the transformation
        auto t_now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast< std::chrono::duration<float> >(t_now - t_start).count();
        float theta = time * glm::radians(180.0f);
        float r = 10;

        // Draw square blocks
        for (int i = 0; i < 4; i++)
        {
            // Store kinematic parameters
            data[i * 12 + 0].push_back(squareTranslations[i].x);
            data[i * 12 + 1].push_back(squareTranslations[i].y);
            data[i * 12 + 2].push_back(squareTranslations[i].z);
            data[i * 12 + 3].push_back(squareRotations[i].x);
            data[i * 12 + 4].push_back(squareRotations[i].y);
            data[i * 12 + 5].push_back(squareRotations[i].z);

            glm::mat4 model;
            model = glm::translate(model, squareTranslations[i]);
            model = glm::rotate(model, glm::radians(squareRotations[i].x), glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::rotate(model, glm::radians(squareRotations[i].y), glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::rotate(model, glm::radians(squareRotations[i].z), glm::vec3(0.0f, 0.0f, 1.0f));
            glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

            glm::vec3 color = blockColors[i];
            glUniform3f(uniObjectColor, color.r, color.g, color.b);

            //glBeginTransformFeedback(GL_LINES);
            glDrawArrays(GL_TRIANGLES, 0, 6 * 6);
            //glEndTransformFeedback();

            // Update kinematic parameters
            r = (-2 * (i / 2) + 1) * length(squareTranslations[i]);
            squareTranslations[i] = glm::vec3(r * cos(theta), r * sin(theta), 0.0);
        }

        // Draw rectangular blocks
        for (int i = 0; i < 4; i++)
        {
            // Store kinematic parameters
            data[i * 12 + 6].push_back(rectTranslations[i].x);
            data[i * 12 + 7].push_back(rectTranslations[i].y);
            data[i * 12 + 8].push_back(rectTranslations[i].z);
            data[i * 12 + 9].push_back(rectRotations[i].x);
            data[i * 12 + 10].push_back(rectRotations[i].y);
            data[i * 12 + 11].push_back(rectRotations[i].z);

            glm::mat4 model;
            model = glm::translate(model, rectTranslations[i]);
            model = glm::rotate(model, glm::radians(rectRotations[i].x), glm::vec3(1.0f, 0.0f, 0.0f));
            model = glm::rotate(model, glm::radians(rectRotations[i].y), glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::rotate(model, glm::radians(rectRotations[i].z), glm::vec3(0.0f, 0.0f, 1.0f));
            glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

            glm::vec3 color = blockColors[i];
            glUniform3f(uniObjectColor, color.r, color.g, color.b);

            glDrawArrays(GL_TRIANGLES, 6 * 6, 6 * 6);

            // Update kinematic parameters
            r = (-2 * (i / 2) + 1) * length(rectTranslations[i]);
            rectTranslations[i] = glm::vec3(r * cos(theta + 90), r * sin(theta + 90), 0.0);
        }

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
        std::string fn = "../working/gl-render/" + std::to_string(frame) + ".png";
        writePng(fn.c_str(), out_image, row_pointers, w, h);
        delete[] row_pointers;
        delete[] out_image;

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        /*
        if (!written)
        {
            // Fetch and print results
            printf("Transform feedback:\n");
            glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(feedback), feedback);
            for (int j = 0; j < 16; j++) {
                for (int i = 0; i < 4; i++) {
                    printf("%f  ", feedback[4 * j + i]);
                }
                printf("\n");
            }

            written = true;
        }
        */
    }

    writeCsv("../working/gl-data/ground-truth.csv", names, data);

    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    glDeleteBuffers(1, &vbo);
    //glDeleteBuffers(1, &tbo);

    glDeleteVertexArrays(1, &vao);

    glfwTerminate();

    return 0;
}
