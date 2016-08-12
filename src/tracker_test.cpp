// Link statically with GLEW
//#define GLEW_STATIC

// Headers
#include "blockmodel.h"
#include "libIO.h"

#include "Eigen/Dense"

#include <stdlib.h>
#include <stdio.h>
#include <string>

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
    col_names.push_back("orientation (XYZ)");
    col_names.push_back("acceleration (XYZ)");
    col_names.push_back("angular velocity (XYZ)");
    vector< vector<VectorXf> > data;
    for (int i = 0; i < 2; ++i)
    {
        vector<VectorXf> col;
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

    // We are extremely confident about our initial condition
    VectorXf x0(s0.size() + v0.size() + theta0.size());
    x0 << s0, v0, theta0;
    MatrixXf K0 = MatrixXf::Zero(x0.size(), x0.size());
    BlockModel block(x0, K0, red, a_g);

    /*
    const vector<string> imagePaths = readImagePaths("../working/gl-render/png-files.txt");
    int num_frames = imagePaths.size();
    int field_width = ceil(log(float(num_frames)) / log(10.0f)) + 1;
    */
    int field_width = 6;

    GLint uniModel = glGetUniformLocation(shaderProgram, "model");
    GLint uniObjectColor = glGetUniformLocation(shaderProgram, "objectColor");
    block.setGlVars(uniModel, uniObjectColor, 0);

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
        VectorXf x_t = block.getState();

        // Input at current time
        //auto t_now = chrono::high_resolution_clock::now();
        //float time = chrono::duration_cast< chrono::duration<float> >(t_now - t_start).count();
        Vector3f a_t(-r * rate * rate * cos(rate * frame * dt),
                     -r * rate * rate * sin(rate * frame * dt),
                     0.0f);
        Vector3f omega_t(0.0f, 0.0f, 0.0f);
        VectorXf u_t(a_t.size() + omega_t.size());
        u_t << a_t + a_g, omega_t;
        
        // Store data
        data[0].push_back(x_t);
        data[1].push_back(u_t);

        // Render scene
        block.draw();

        // Save the image
        string frame_str = to_string(frame);
        string fn = "../working/gl-render/" + frame_str.insert(0, field_width - frame_str.size(), '0') + ".png";
        saveImage(fn.c_str());

        // Take a step in state space
        block.updateState(u_t, dt);

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
