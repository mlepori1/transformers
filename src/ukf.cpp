#include "ukf.h"
#include "libIO.h"

using namespace Eigen;
using namespace std;


UnscentedKalmanFilter::UnscentedKalmanFilter(const VectorXf x0, const MatrixXf K0,
        const vector<BlockModel> blocks)
{
    // Set first and second moments of initial state space distribution
    mu_x = x0;
    K_x = K0;

    this->blocks = blocks;

    cout << "Number of objects: " << this->blocks.size() << endl;
    vector<BlockModel>::iterator it;
    for (it = this->blocks.begin(); it != this->blocks.end(); ++it)
    {
        it->printState();
    }

    // Set sigma points
    // Apparently w0 = 1/3 is good if you believe the initial distribution is
    // Gaussian
    float w0 = 1.0 / 3.0;
    initializeSigmaPoints(x0, K0, w0);

    // Set image dimensions
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    image_width = viewport[2];
    image_height = viewport[3];
}

void UnscentedKalmanFilter::initializeSigmaPoints(const VectorXf x0,
        const MatrixXf K0, const float w0)
{
    // Calculate matrix square root used in sigma point calculations
    float n = x0.size();
    MatrixXf K_scaled = n / (1 - w0) * K0;
    MatrixXf K_scaled_sqrt = K_scaled.llt().matrixL(); // Cholesky decomposition

    // Define matrix with sigma points as columns
    X = MatrixXf::Zero(n, 2 * n + 1);
    X.col(0) = x0;
    for (int i = 1; i <= n; ++i)
    {
        X.col(i) = x0 + K_scaled_sqrt.col(i - 1);
        X.col(i + n) = x0 - K_scaled_sqrt.col(i - 1);
    }

    // Define vector of sigma point weights
    w = VectorXf::Zero(2 * n + 1);
    float w_constant = (1 - w0) / (2 * n);
    w.fill(w_constant);
    w(0) = w0;
}

VectorXf UnscentedKalmanFilter::getState() const
{
    VectorXf x(mu_x.size());
    vector<BlockModel>::const_iterator it;
    for (it = blocks.begin(); it != blocks.end(); ++it)
    {
        x << it->getState();
    }

    return x;
}

VectorXf UnscentedKalmanFilter::updateState(const VectorXf x, const VectorXf u, const float dt)
{
    // Update and render each object in the scene
    VectorXf x_i(x.size());
    int cur_row_x = 0;
    int cur_row_u = 0;
    vector<BlockModel>::iterator it;
    for (it = blocks.begin(); it != blocks.end(); ++it)
    {
        int state_dim = it->statesize;
        int input_dim = 6;
        x_i << it->updateState(x.segment(cur_row_x, state_dim),
                               u.segment(cur_row_u, input_dim),
                               dt);
        cur_row_x += state_dim;
        cur_row_u += input_dim;
    }

    return x_i;
}

void UnscentedKalmanFilter::updateState(const VectorXf u, const float dt)
{
    // Update and render each object in the scene
    int cur_row_u = 0;
    vector<BlockModel>::iterator it;
    for (it = blocks.begin(); it != blocks.end(); ++it)
    {
        int input_dim = 6;
        it->updateState(u.segment(cur_row_u, input_dim), dt);
        cur_row_u += input_dim;
    }
}

Map<VectorXi> UnscentedKalmanFilter::generateObservation(GLFWwindow* window)
{
    // Enable depth rendering, clear the screen to black and clear depth buffer
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Update and render each object in the scene
    vector<BlockModel>::iterator it;
    int cur_row = 0;
    for (it = blocks.begin(); it != blocks.end(); ++it)
    {
        it->draw();
    }

    Map<VectorXi> image = sceneSnapshot();

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();

    return image;
}

void UnscentedKalmanFilter::generateObservation(GLFWwindow* window, const char* image_fn)
{
    // Enable depth rendering, clear the screen to black and clear depth buffer
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Update and render each object in the scene
    vector<BlockModel>::iterator it;
    int cur_row = 0;
    for (it = blocks.begin(); it != blocks.end(); ++it)
    {
        it->draw();
    }

    sceneSnapshot(image_fn);

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void UnscentedKalmanFilter::inferState(const VectorXf u, const VectorXf y, const float dt,
        GLFWwindow* window)
{
    int bytes_per_pixel = 3;
    int num_bytes = image_width * image_height * bytes_per_pixel;

    // Propagate sigma points through process and observation models
    MatrixXf Y = MatrixXf::Zero(num_bytes, X.cols());
    for (int i = 0; i < X.cols(); ++i)
    {
        X.col(i) = updateState(X.col(i), u, dt);
        Y.col(i) = generateObservation(window).cast<float>();
    }

    // Calculate predicted mean and covariance
    VectorXf mean_x = weightedMean(w, X);
    MatrixXf cov_x = weightedCovariance(w, X, mean_x);

    // Calculate predicted observation and innovation covariance
    VectorXf mean_y = weightedMean(w, Y);
    // This gives us everything we need to do computations with Ky
    // But it avoids storing Ky
    ArrayXf w_sqrt = w.array().sqrt();
    for (int i = 0; i < Y.cols(); ++i)
    {
        // Center and weight the observation samples
        Y.col(i) = w_sqrt(i) * (Y.col(i) - mean_y);
    }
    JacobiSVD<MatrixXf> svd(Y, ComputeThinU);

    // Calculate cross-covariance
    MatrixXf cov_xy = weightedCrossCovariance(w, X, Y, mean_x, mean_y);

    // Perform update using standard Kalman filter equations
    MatrixXf cov_y_inv_sqrt = svd.matrixU(); // We're not done with this yet
    ArrayXf S_inv = svd.singularValues().array().inverse();
    for (int i = 0; i < cov_y_inv_sqrt.cols(); ++i)
    {
        cov_y_inv_sqrt.col(i) = S_inv(i) * cov_y_inv_sqrt.col(i);
    }

    MatrixXf cov_x_sqrt = cov_xy * cov_y_inv_sqrt;
    
    // Update estimated moments
    mu_x = mu_x + cov_x_sqrt * cov_y_inv_sqrt.transpose() * (y - mean_y);
    K_x = K_x - cov_x_sqrt * cov_x_sqrt.transpose();
}

Map<VectorXi> UnscentedKalmanFilter::sceneSnapshot() const
{
    // Wait for OpenGL to finish rendering
    glFinish();

    // Read pixels on screen into a raw data buffer
    int bytes_per_pixel = 3;
    int num_bytes = image_height * image_width * bytes_per_pixel;
    int* image_as_ints = new int[num_bytes];
    glReadPixels(0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_INT,
            image_as_ints);
    
    // Convert to vector
    Map<VectorXi> image(image_as_ints, num_bytes);

    delete[] image_as_ints;

    return image;
}

void UnscentedKalmanFilter::sceneSnapshot(const char* image_fn) const
{

    // Wait for OpenGL to finish rendering
    glFinish();

    // Read pixels on screen into a raw data buffer
    int bytes_per_pixel = 3;
    int num_bytes = image_height * image_width * bytes_per_pixel;
    unsigned char* image_bytes = new unsigned char[num_bytes];
    glReadPixels(0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE,
            image_bytes);

    // Save image as PNG
    int bytes_per_row = image_width * bytes_per_pixel;
    unsigned char** row_pointers = new unsigned char*[image_height];
    for (int row = 0; row < image_height; row++)
        row_pointers[image_height - 1 - row] = image_bytes + row * bytes_per_row;
    writePng(image_fn, image_bytes, row_pointers, image_width, image_height);

    delete[] row_pointers;
    delete[] image_bytes;
}


// Helper functions for math operations
VectorXf weightedMean(const VectorXf weights, const MatrixXf samples)
{
    int n = samples.rows();
    VectorXf mean = VectorXf::Zero(n);

    // TODO: assert dimensions are correct

    for (int i = 0; i < samples.cols(); ++i)
    {
        mean += weights(i) * samples.col(i);
    }

    return mean;
}

MatrixXf weightedCovariance(const VectorXf weights, const MatrixXf samples,
        const VectorXf mean)
{
    int n = samples.rows();
    MatrixXf covariance = MatrixXf::Zero(n, n);

    // TODO: assert dimensions are correct

    for (int i = 0; i < samples.cols(); ++i)
    {
        VectorXf whitened_sample = samples.col(i) - mean;
        covariance += weights(i) * whitened_sample * whitened_sample.transpose();
    }

    return covariance;
}

MatrixXf weightedCrossCovariance(const VectorXf w, const MatrixXf X,
        const MatrixXf Y, const VectorXf mu_x, const VectorXf mu_y)
{
    int n = X.rows();
    int m = Y.rows();
    MatrixXf K_xy = MatrixXf::Zero(n, m);

    for (int i = 0; i < X.cols(); ++i)
    {
        VectorXf x = X.col(i) - mu_x;
        VectorXf y = Y.col(i) - mu_y;
        K_xy += w(i) * x * y.transpose();
    }

    return K_xy;
}
