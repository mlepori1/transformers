// Link statically with GLEW
//#define GLEW_STATIC

// Headers
#include <stdlib.h>
#include <stdio.h>
#include <png.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>

// Shader macro
#define GLSL(src) "#version 150 core\n" #src

png_byte* readPng(char* filename, int& width, int& height)
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

void writePng(char *filename, png_byte* image, png_bytep* row_pointers, int width, int height)
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

// Vertex shader
const GLchar* vertexShaderSrc = GLSL(
    in vec3 pos;
    in vec3 color;

    out vec3 vColor;

    void main()
    {
        gl_Position = vec4(pos, 1.0);
        vColor = color;
    }
);

// Geometry shader
const GLchar* geometryShaderSrc = GLSL(
    layout(points) in;
    layout(line_strip, max_vertices = 16) out;

    in vec3 vColor[];
    out vec3 fColor;
    out vec4 transformFeedback;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 proj;

    void main()
    {
        fColor = vColor[0];

        /*
         Block vertices in metric units
         2x2 SQUARE BRICK DIMENSIONS:
           Height: 19.2 mm
           Width:  31.8 mm
           Length: 31.8 mm
        float width  = 31.8;
        float length = 31.8;
        float height = 19.2;
        */

        float width  = 31.8;
        float length = 31.8;
        float height = 19.2;

        vec4 offset = proj * view * gl_in[0].gl_Position;

        // +X direction is "North", -X direction is "South"
        // +Y direction is "Up",    -Y direction is "Down"
        // +Z direction is "East",  -Z direction is "West"
        //                                     N/S   U/D   E/W
        vec4 NEU = proj * view * model * vec4( width / 2,  length / 2,  height / 2, 0.0);
        vec4 NED = proj * view * model * vec4( width / 2, -length / 2,  height / 2, 0.0);
        vec4 NWU = proj * view * model * vec4( width / 2,  length / 2, -height / 2, 0.0);
        vec4 NWD = proj * view * model * vec4( width / 2, -length / 2, -height / 2, 0.0);
        vec4 SEU = proj * view * model * vec4(-width / 2,  length / 2,  height / 2, 0.0);
        vec4 SED = proj * view * model * vec4(-width / 2, -length / 2,  height / 2, 0.0);
        vec4 SWU = proj * view * model * vec4(-width / 2,  length / 2, -height / 2, 0.0);
        vec4 SWD = proj * view * model * vec4(-width / 2, -length / 2, -height / 2, 0.0);

        // Create a cube centered on the given point.
        gl_Position = offset + NED;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + NWD;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + SWD;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + SED;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + SEU;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + SWU;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + NWU;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + NEU;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + NED;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + SED;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + SEU;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + NEU;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + NWU;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + NWD;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + SWD;
        transformFeedback = gl_Position;
        EmitVertex();

        gl_Position = offset + SWU;
        transformFeedback = gl_Position;
        EmitVertex();

        EndPrimitive();
    }
);

// Fragment shader
const GLchar* fragmentShaderSrc = GLSL(
    in vec3 fColor;

    out vec4 outColor;

    void main()
    {
        outColor = vec4(fColor, 1.0);
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

int main()
{
    auto t_start = std::chrono::high_resolution_clock::now();

    // Initialize a GLFW window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    glewInit();


    // Compile shaders and create shader programs
    GLuint vertexShader = createShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint geometryShader = createShader(GL_GEOMETRY_SHADER, geometryShaderSrc);
    GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, geometryShader);
    glAttachShader(shaderProgram, fragmentShader);

    // Transform feedback buffer for debugging
    const GLchar* feedbackVaryings[] = { "transformFeedback" };
    glTransformFeedbackVaryings(shaderProgram, 1, feedbackVaryings, GL_INTERLEAVED_ATTRIBS);

    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // Create Vertex Array Objects
    GLuint vaoCube;
    glGenVertexArrays(1, &vaoCube);

    // Create Vertex Buffer Objects and copy the vertex data to them
    GLuint vboCube;
    glGenBuffers(1, &vboCube);

    // Define vertices
    GLfloat cubeCenters[] = {
    //  X, Y, Z, R, G, B
         0.5f,  0.5f, 0.5f, 1.0f, 0.0f, 0.0f
    };

    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeCenters), cubeCenters, GL_STATIC_DRAW);

    glBindVertexArray(vaoCube);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);

    // Specify the layout of the vertex data
    GLint posAttrib = glGetAttribLocation(shaderProgram, "pos");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), 0);

    GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
    glEnableVertexAttribArray(colAttrib);
    glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

    // Create transform feedback buffer
    GLuint tbo;
    glGenBuffers(1, &tbo);
    glBindBuffer(GL_ARRAY_BUFFER, tbo);
    glBufferData(GL_ARRAY_BUFFER, 4 * 16 * sizeof(GLfloat), NULL, GL_STATIC_READ);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, tbo);
    GLfloat feedback[4 * 16];

    glBindBuffer(GL_ARRAY_BUFFER, vboCube);

    // Set up projection
    // Camera located 600mm above the origin, pointing along -z.
    // Downward direction is... +z??
    glm::mat4 view = glm::lookAt(
        glm::vec3(50.0f, 50.0f, 600.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    GLint uniView = glGetUniformLocation(shaderProgram, "view");
    glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

    printf("View matrix:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", view[i][j]);
        }
        printf("\n");
    }
    
    // I calculated field of view from camera intrinsic parameters
    // I set the near and far clipping planes ~arbitrarily~
    float field_of_view = 47.9975;  // degrees
    float near_plane = 100;     // mm
    float far_plane = 1000;     // mm
    glm::mat4 proj = glm::perspective(glm::radians(field_of_view), 800.0f / 600.0f, near_plane, far_plane);
    //glm::mat4 proj = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 1.0f, 10.0f);
    GLint uniProj = glGetUniformLocation(shaderProgram, "proj");
    glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

    printf("Projection matrix:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f ", proj[i][j]);
        }
        printf("\n");
    }

    GLint uniModel = glGetUniformLocation(shaderProgram, "model");

    bool written = false;
    while (!glfwWindowShouldClose(window))
    {
        glEnable(GL_DEPTH_TEST);
        // Clear the screen to black and clear depth buffer
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Calculate the transformation
        auto t_now = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration_cast< std::chrono::duration<float> >(t_now - t_start).count();

        glm::mat4 model;
        model = glm::rotate(model, time * glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

        // Draw cube
        glBeginTransformFeedback(GL_LINES);
        glDrawArrays(GL_POINTS, 0, 1);
        glEndTransformFeedback();

        // Save image
        if (!written)
        {
            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT, viewport);
            int w = viewport[2];
            int h = viewport[3];
            printf("Viewport width: %d\n", w);
            printf("Viewport height: %d\n", h);
            glFinish();
            unsigned int bytes_per_row = 4 * w;
            png_byte* out_image = new png_byte[h * bytes_per_row];
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, out_image);
            png_bytep* row_pointers = new png_bytep[h];
            for (int row = 0; row < h; row++)
                row_pointers[h - 1 - row] = out_image + row * bytes_per_row;
            writePng("test.png", out_image, row_pointers, w, h);
            delete[] row_pointers;
            delete[] out_image;

            //written = true;
        }

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

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
    }

    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    glDeleteBuffers(1, &vboCube);
    glDeleteBuffers(1, &tbo);

    glDeleteVertexArrays(1, &vaoCube);

    glfwTerminate();

    return 0;
}
