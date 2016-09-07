// Link statically with GLEW
//#define GLEW_STATIC

// Headers
#include "ukf.h"

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
        fNormal = mat3(model) * normal;
        gl_Position = proj * view * vec4(fPos, 1.0);
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
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status)
    {
        char buffer[512];
        glGetShaderInfoLog(shader, 512, NULL, buffer);
        cerr << "Shader compile error: " << buffer << endl;
    }

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
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
    if (!status)
    {
        char buffer[512];
        glGetShaderInfoLog(shaderProgram, 512, NULL, buffer);
        cerr << "Shader program linking error: " << buffer << endl;
    }

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
    GLFWwindow* window = glfwCreateWindow(2 * 160, 2 * 120, "OpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);

    return window;
}

UnscentedKalmanFilter initializeUKF(const configParams params, GLint uniModel, GLint uniObjectColor)
{
    // Initial conditions for the unscented kalman filter
    int num_blocks = 1;
    vector<BlockModel> blocks;
    vector<VectorXf> means;
    vector<VectorXf> diag_covariances;
    for (int i = 0; i < num_blocks; ++i)
    {
        Vector3f center(0.0f, 0.0f, 0.0f);
        float angle = 0.0f;
        Vector3f s0 = center + Vector3f(r * cos(angle), r * sin(angle), 0.0f);
        Vector3f v0(-r * rate * sin(angle), r * rate * cos(angle), 0.0f);
        Vector3f theta0(0.0f, 0.0f, 0.0f);

        VectorXf mu(s0.size() + v0.size() + theta0.size());
        mu << s0, v0, theta0;

        /*
        VectorXf mu_noise = VectorXf::Zero(mu.size());
        VectorXf mu_augmented(mu.size() + mu_noise.size());
        mu_augmented << mu, mu_noise;
        */
        means.push_back(mu);
       
        // Process noise
        Vector3f sigma_s     = params.sigma_position * Vector3f::Ones();
        Vector3f sigma_v     = params.sigma_velocity * Vector3f::Ones();
        Vector3f sigma_theta = params.sigma_orientation * Vector3f::Ones();
        VectorXf sigma_noise(sigma_s.size() + sigma_v.size() + sigma_theta.size());
        sigma_noise << sigma_s, sigma_v, sigma_theta;

        VectorXf sigma(sigma_noise.size());
        float s_s = 1.0;
        float s_v = 1.0;
        float s_theta = 0.01;
        sigma << s_s * Vector3f::Ones(), s_v * Vector3f::Ones(), s_theta * Vector3f::Ones();
        diag_covariances.push_back(sigma);

        // Initialize block model
        BlockModel block(mu, sigma_noise, red, a_g);
        block.setGlVars(uniModel, uniObjectColor, 0);
        blocks.push_back(block);
    }


    // Initialize unscented kalman filter
    int dim = 0;
    for (int i = 0; i < blocks.size(); ++i)
        dim += blocks[i].statesize;

    int index = 0;
    VectorXf x(dim);
    MatrixXf cov = MatrixXf::Zero(dim, dim);
    for (int i = 0; i < blocks.size(); ++i)
    {
        int dim_i = blocks[i].statesize;
        cov.block(index, index, dim_i, dim_i) = diag_covariances[i].asDiagonal();
        x.segment(index, dim_i) = means[i];
        index += dim_i;
    }

    UnscentedKalmanFilter ukf(x, cov, blocks, params);

    return ukf;
}

void setTransformationMatrices(GLint shaderProgram)
{
    // Set up projection
    // Camera located 600mm above the origin, pointing along -z.
    glm::mat4 view = glm::lookAt(
        glm::vec3(0.0f, -600.0f, 600.0f),
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

void simulate(int num_samples,  UnscentedKalmanFilter ukf,
        configParams params, GLFWwindow* window, vector<VectorXf>& state,
        vector<VectorXf>& input, vector<VectorXf>& output,
        vector<string>& image_fns)
{
    // No sequence should need more than 10 digits (famous last words)
    int field_width = 10;
    for (int frame_index = 0; frame_index < num_samples; ++frame_index)
    {
        // Observed (noise-corrupted) state at current time
        VectorXf y_t = ukf.observeState();
        
        // True state at current time
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
        string fn;
        if (params.save_rgb)
        {
            fn = "../working/gl-render/rgb-" + frame_str.insert(0, field_width - frame_str.size(), '0') + ".png";
            ukf.generateObservation(window, GL_RGB, fn.c_str());
        }
        else if (params.save_depth)
        {
            fn = "../working/gl-render/depth-" + frame_str.insert(0, field_width - frame_str.size(), '0') + ".png";
            ukf.generateObservation(window, GL_DEPTH_COMPONENT, fn.c_str());
        }

        // Store data
        state.push_back(x_t);
        input.push_back(u_t);
        output.push_back(y_t);
        image_fns.push_back(fn);

        // Take a step in state space
        ukf.updateState(u_t, dt);
    }
}

void estimate(int num_samples, UnscentedKalmanFilter ukf,
        const configParams params, vector<VectorXf>& state,
        vector<VectorXf>& err_cov)
{
    // In this case the state vector is partially observed, so the input is x
    vector<VectorXf> input = readCsv(params.u_path.c_str());
    vector<VectorXf> output = readCsv(params.x_path.c_str());

    // Every state should have a corresponding input and output
    assert(input.size() == output.size());

    // It isn't possible to process more frames than exist
    assert(num_samples <= int(output.size()));

    // A negative value means: process all files in output.
    if (num_samples < 0)
        num_samples = output.size();

    for (int frame_index = 0; frame_index < num_samples; ++frame_index)
    {
        VectorXf u = input[frame_index];
        VectorXf x = output[frame_index];
        VectorXf y = ukf.observeState(x);

        /*
        if (params.observation == "position")
            y = output[frame_index].segment<3>(0);
        else if (params.observation == "orientation")   // orientation
            y = output[frame_index].segment<3>(6);
        */

        // Estimate and store the latent state
        ukf.inferState(u, y, params);
        VectorXf x_t = ukf.getStateEstimate();
        MatrixXf K_t = ukf.getErrorCovariance();
        VectorXf K_flat(Map<VectorXf>(K_t.data(), K_t.cols() * K_t.rows()));

        state.push_back(x_t);
        err_cov.push_back(K_flat);
    }
}

void estimate(int num_samples, UnscentedKalmanFilter ukf,
        const configParams params, GLFWwindow* window, vector<VectorXf> state)
{
    vector<VectorXf> input = readCsv(params.u_path.c_str());
    vector<string> output = readImagePaths(params.y_path.c_str());

    // Every state should have a corresponding input and output
    //assert(input.size() == output.size());

    // It isn't possible to process more frames than exist
    assert(num_samples <= int(output.size()));

    // A negative value means: process all files in output_fns.
    if (num_samples < 0)
        num_samples = output.size();

    stringstream ss;
    string fn_prefix;
    for (int frame_index = 0; frame_index < num_samples; ++frame_index)
    {
        ss << "../working/gl-render/sigma/frame" << frame_index << "-point";
        fn_prefix = ss.str();
        ss.str("");

        // Load image
        int width, height;
        png_byte* image_bytes = readPng(output[frame_index].c_str(), width, height);
        int num_bytes = width * height * 3;
        VectorXf y = toVector(image_bytes, num_bytes);

        VectorXf u = input[frame_index];

        // Estimate and store the latent state
        ukf.inferState(u, y, params, window, fn_prefix);
        VectorXf x_t = ukf.getStateEstimate();
        state.push_back(x_t);
    }
}

vector<VectorXf> calculateResidual(vector<VectorXf> x_estimated, vector<VectorXf> x_true)
{
    assert(x_estimated.size() <= x_true.size());

    vector<VectorXf> residual;
    for (int i = 0; i < x_estimated.size(); ++i)
        residual.push_back(x_true[i] - x_estimated[i]);

    return residual;
}

void printUsage(char* argv[])
{
    cerr << "Usage: " << argv[0] << " simulate <num samples>" << endl;
    cerr << "   OR: " << argv[0] << " [--debug] [--observe i|s|t]  estimate <num samples>" << endl;
}

struct args
{
    // Number of timesteps to simulate or estimate
    int num_samples;

    // Print debug output?
    bool debug;

    // Operation mode: generate simulation data or estimate state
    bool simulate;
};

int parseArgs(int argc, char* argv[], args& cl_args)
{
    cl_args.debug = false;

    string mode;
    if (argc == 4)
    {
        if (string(argv[1]) != "--debug")
        {
            printUsage(argv);
            return 1;
        }

        cl_args.debug = true;
        mode = string(argv[2]);
        cl_args.num_samples = atoi(argv[3]);
    }
    else if (argc == 3)
    {
        mode = string(argv[1]);
        cl_args.num_samples = atoi(argv[2]);
    }
    else
    {
        cout << argc << endl;
        printUsage(argv);
        return 1;
    }

    // Set operation mode
    if (mode == "simulate")
        cl_args.simulate = true;
    else if (mode == "estimate")
        cl_args.simulate = false;
    else
    {
        cout << mode << endl;
        printUsage(argv);
        return 1;
    }

    return 0;
}

int main(int argc, char* argv[])
{
    // Parse command line arguments
    args cl_args;
    if (parseArgs(argc, argv, cl_args))
        return 1;

    // Read config file
    configParams params;
    parseConfigFile("ukf.config", params);

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
    UnscentedKalmanFilter ukf = initializeUKF(params, uniModel, uniObjectColor);
    if (cl_args.debug)
        ukf.setDebugStatus(cl_args.debug);

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


    if (cl_args.simulate)
    {
        vector<string> image_fns;
        vector<VectorXf> output;
        simulate(cl_args.num_samples, ukf, params, window, state, input, output,
                 image_fns);
        writeCsv(params.u_path.c_str(), input_colnames, input);
        writeCsv(params.x_path.c_str(), state_colnames, state);
        writeCsv(params.y_path.c_str(), state_colnames, output);
        writeImagePaths(params.images_path.c_str(), image_fns);
    }
    else    // estimate
    {
        vector<VectorXf> err_cov;
        vector<string> cov_colnames;
        if (params.observe_image)
            estimate(cl_args.num_samples, ukf, params, window, state);
        else // state is partially observed
            estimate(cl_args.num_samples, ukf, params, state, err_cov);

        string out_fn_state = "../working/gl-data/state-estimation.csv";
        writeCsv(out_fn_state.c_str(), state_colnames, state);

        string out_fn_err_cov = "../working/gl-data/state-error-covariance.csv";
        writeCsv(out_fn_err_cov.c_str(), cov_colnames, err_cov);

        //vector<VectorXf> true_state = readCsv(params.x_path.c_str());
        //vector<VectorXf> residual = calculateResidual(state, true_state);
        //string out_fn_resid = "../working/gl-data/state-residual.csv";
        //writeCsv(out_fn_resid.c_str(), state_colnames, residual);
    }


    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    glfwTerminate();

    return 0;
}
