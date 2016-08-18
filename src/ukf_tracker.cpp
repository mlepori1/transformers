// Link statically with GLEW
//#define GLEW_STATIC

// Headers
#include "ukf.h"
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

// Shader macro
// FIXME: this should be an inline function
#define GLSL(src) "#version 150 core\n" #src


using namespace Eigen;
using namespace std;


// Define block colors and gravity vector
const Vector3f red(1.0f, 0.0f, 0.0f);
const Vector3f green(0.0f, 1.0f, 0.0f);
const Vector3f blue(0.0f, 0.0f, 1.0f);
const Vector3f yellow(1.0f, 1.0f, 0.0f);
const Vector3f a_g(0.0f, 0.0f, 9810.0f);   // millimeters / second^2
const float dt = 1.0 / 100.0;   // seconds
const float r = 100.0;          // millimeters
const float rate = glm::radians(180.0f);  // pi radians per second

// Define vertices
const GLfloat w = 31.8;
const GLfloat l = 31.8;
const GLfloat h = 19.2;
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

GLint makeShaderProgram(const GLuint vertexShader, const GLuint fragmentShader)
{
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

    return shaderProgram;
}

GLFWwindow* initializeGlfwWindow()
{
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

    return window;
}

vector<BlockModel> setInitialConditions(GLint uniModel, GLint uniObjectColor)
{
    // Initial conditions for the unscented kalman filter
    int num_blocks = 1;
    vector<BlockModel> blocks;
    for (int i = 0; i < num_blocks; ++i)
    {
        Vector3f center(0.0f, 0.0f, 0.0f);
        float angle = 0.0f;
        Vector3f s0 = center + Vector3f(r * cos(angle), r * sin(angle), 0.0f);
        Vector3f v0(-r * rate * sin(angle), r * rate * cos(angle), 0.0f);
        Vector3f theta0(0.0f, 0.0f, 0.0f);

        VectorXf x0(s0.size() + v0.size() + theta0.size());
        x0 << s0, v0, theta0;
        //MatrixXf K0 = MatrixXf::Zero(x0.size(), x0.size());

        // Initialize block model
        BlockModel block(x0, red, a_g);
        block.setGlVars(uniModel, uniObjectColor, 0);
        blocks.push_back(block);
    }

    return blocks;
}

void setTransformationMatrices(GLint shaderProgram)
{
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
}

void simulate(GLFWwindow* window, UnscentedKalmanFilter ukf, vector<VectorXf>& state, vector<VectorXf>& input, vector<string>& output_fns)
{
    int field_width = 1;   // No sequence should need more than 10 digits (famous last words)
    //int frame = 0;
    //while (!glfwWindowShouldClose(window))
    int num_samples = 5;
    for (int frame_index = 0; frame_index < num_samples; ++frame_index)
    {
        // State at current time
        VectorXf x_t = ukf.getState();

        // Input at current time
        Vector3f a_t(-r * rate * rate * cos(rate * frame_index * dt),
                     -r * rate * rate * sin(rate * frame_index * dt),
                     0.0f);
        Vector3f omega_t(0.0f, 0.0f, 0.0f);
        VectorXf u_t(a_t.size() + omega_t.size());
        u_t << a_t + a_g, omega_t;
        
        // Render scene and save resulting image
        string frame_str = to_string(frame_index);
        string fn = "../working/gl-render/" + frame_str.insert(0, field_width - frame_str.size(), '0') + ".png";
        //ukf.generateObservation(window, GL_DEPTH_COMPONENT, fn.c_str());
        ukf.generateObservation(window, GL_RGB, fn.c_str());

        // Store data
        state.push_back(x_t);
        input.push_back(u_t);
        output_fns.push_back(fn);

        // Take a step in state space
        ukf.updateState(u_t, dt);

        //++frame_index;
    }
}

void estimate(GLFWwindow* window, UnscentedKalmanFilter ukf, vector<VectorXf>& state)
{
    vector<VectorXf> u = readCsv("../working/gl-data/input-simulation.csv");
    vector<string> output_fns = readImagePaths("../working/gl-data/output-simulation.csv");

    // Every state should have a corresponding input and output
    assert(u.size() == output_fns.size());
    
    for (int frame_index = 0; frame_index < output_fns.size(); ++frame_index)
    //for (int frame_index = 0; frame_index < 1; ++frame_index)
    {
        cout << "Processing frame " << frame_index << endl;

        // Load image
        int width, height;
        png_byte* image_bytes = readPng(output_fns[frame_index].c_str(), width, height);
        int num_bytes = width * height * 3;
        VectorXf y = toVector(image_bytes, num_bytes);

        // Estimate the latent state
        ukf.inferState(u[frame_index], y, dt, window);
        VectorXf x_t = ukf.getStateEstimate();

        // Store data
        state.push_back(x_t);
    }
}

vector<VectorXf> calculateResidual(vector<VectorXf> x_estimated, vector<VectorXf> x_true)
{
    assert(x_estimated.size() == x_true.size());

    vector<VectorXf> residual;
    for (int i = 0; i < x_true.size(); ++i)
        residual.push_back(x_true[i] - x_estimated[i]);

    return residual;
}

int main(int argc, char* argv[])
{
    // Parse command line arguments
    assert(argc == 2);
    string arg_param(argv[1]);
    bool do_simulate;
    if (arg_param  == "simulate")
        do_simulate = true;
    else if (arg_param == "estimate")
        do_simulate = false;
    else
    {
        cerr << "Usage: " << argv[0] << " [simulate|estimate]" << endl;
        return 1;
    }

    // Initialize GLFW window and GLEW
    GLFWwindow* window = initializeGlfwWindow();
    glewExperimental = GL_TRUE;
    glewInit();

    // Compile shaders and create shader programs
    GLuint vertexShader = createShader(GL_VERTEX_SHADER, vertexShaderSrc);
    GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
    GLint shaderProgram = makeShaderProgram(vertexShader, fragmentShader);

    // Create Vertex Array Objects
    GLuint vao;
    glGenVertexArrays(1, &vao);
    GLuint vbo;
    glGenBuffers(1, &vbo);
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

    setTransformationMatrices(shaderProgram);

    GLint uniModel = glGetUniformLocation(shaderProgram, "model");
    GLint uniObjectColor = glGetUniformLocation(shaderProgram, "objectColor");
    vector<BlockModel> blocks = setInitialConditions(uniModel, uniObjectColor);

    // Generate vectors to hold simulation data
    vector<VectorXf> state;
    vector<string> state_colnames;
    state_colnames.push_back("position_x");
    state_colnames.push_back("position_y");
    state_colnames.push_back("position_z");
    state_colnames.push_back("velocity_x");
    state_colnames.push_back("velocity_y");
    state_colnames.push_back("velocity_z");
    state_colnames.push_back("orientation_x");
    state_colnames.push_back("orientation_y");
    state_colnames.push_back("orientation_z");
    
    vector<VectorXf> input;
    vector<string> input_colnames;
    input_colnames.push_back("acceleration_x");
    input_colnames.push_back("acceleration_y");
    input_colnames.push_back("acceleration_z");
    input_colnames.push_back("angular-velocity_x");
    input_colnames.push_back("angular-velocity_y");
    input_colnames.push_back("angular-velocity_z");

    // Initialize unscented kalman filter
    int dim = 0;
    for (int i = 0; i < blocks.size(); ++i)
        dim += blocks[i].statesize;
    VectorXf x(dim);
    for (int i = 0; i < blocks.size(); ++i)
        x << blocks[i].getState();
    MatrixXf cov = MatrixXf::Identity(dim, dim);
    UnscentedKalmanFilter ukf(x, cov, blocks);

    cout << "Status: ";
    if (do_simulate)
    {
        cout << "simulate" << endl;
        vector<string> output_fns;
        simulate(window, ukf, state, input, output_fns);
        string out_fn_state = "../working/gl-data/state-simulation.csv";
        string out_fn_input = "../working/gl-data/input-simulation.csv";
        string out_fn_output = "../working/gl-data/output-simulation.csv";
        writeCsv(out_fn_input.c_str(), input_colnames, input);
        writeImagePaths(out_fn_output.c_str(), output_fns);
        writeCsv(out_fn_state.c_str(), state_colnames, state);
    }
    else    // estimate
    {
        cout << "estimate" << endl;
        estimate(window, ukf, state);
        vector<VectorXf> true_state = readCsv("../working/gl-data/state-simulation.csv");
        vector<VectorXf> residual = calculateResidual(state, true_state);
        string out_fn_state = "../working/gl-data/state-estimation.csv";
        string out_fn_resid = "../working/gl-data/state-residual.csv";
        writeCsv(out_fn_resid.c_str(), state_colnames, residual);
        writeCsv(out_fn_state.c_str(), state_colnames, state);
    }


    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    glfwTerminate();

    return 0;
}
