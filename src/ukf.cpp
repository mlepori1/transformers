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

    cout << "Dimensionality of state vector: " << n << endl;
    cout << "Number of sigma points: " << 2 * n + 1 << endl;
    cout << "(scaled) Square root of covariance matrix: " << endl;
    cout << K_scaled_sqrt << endl;

    // Define matrix with sigma points as columns
    X = MatrixXf::Zero(n, 2 * n + 1);
    X.col(0) = x0;
    for (int i = 1; i <= n; ++i)
    {
        X.col(i) = x0 + n * K_scaled_sqrt.col(i - 1);
        X.col(i + n) = x0 - n * K_scaled_sqrt.col(i - 1);
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
        x << it->getState();

    return x;
}

VectorXf UnscentedKalmanFilter::updateState(const VectorXf x, const VectorXf u,
        const float dt)
{
    // Update the state of each object in the scene based on the observed input
    VectorXf x_i(x.size());
    int cur_row_x = 0;
    int cur_row_u = 0;
    vector<BlockModel>::iterator it;
    for (it = blocks.begin(); it != blocks.end(); ++it)
    {
        int state_dim = it->statesize;
        int input_dim = 6;  // FIXME
        x_i.segment(cur_row_x, state_dim) = it->updateState(
                x.segment(cur_row_x, state_dim),
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

VectorXf UnscentedKalmanFilter::generateObservation(GLFWwindow* window,
        const GLenum format) const
{
    // Enable depth rendering, clear the screen to white and clear depth buffer
    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Update and render each object in the scene
    vector<BlockModel>::const_iterator it;
    for (it = blocks.begin(); it != blocks.end(); ++it)
        it->draw();

    VectorXf image = sceneSnapshot(format);

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();

    return image;
}

void UnscentedKalmanFilter::generateObservation(GLFWwindow* window,
        const GLenum format, const char* image_fn) const
{
    // Enable depth rendering, clear the screen to white and clear depth buffer
    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Update and render each object in the scene
    vector<BlockModel>::const_iterator it;
    for (it = blocks.begin(); it != blocks.end(); ++it)
        it->draw();

    sceneSnapshot(format, image_fn);

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();
}

VectorXf UnscentedKalmanFilter::sceneSnapshot(const GLenum format) const
{
    // rgb format has 3 bytes per pixel; depth format has 1.
    assert(format == GL_RGB || format == GL_DEPTH_COMPONENT);
    int bytes_per_pixel;
    if (format == GL_RGB)
        bytes_per_pixel = 3;
    else
        bytes_per_pixel = 1;

    // Wait for OpenGL to finish rendering
    glFinish();

    // Read pixels on screen into a raw data buffer
    int num_bytes = image_height * image_width * bytes_per_pixel;
    unsigned char* image_bytes = new unsigned char[num_bytes];
    glReadPixels(0, 0, image_width, image_height, format, GL_UNSIGNED_BYTE,
            image_bytes);

    return toVector(image_bytes, num_bytes);
}

void UnscentedKalmanFilter::sceneSnapshot(const GLenum format,
        const char* image_fn) const
{
    // rgb format has 3 bytes per pixel; depth format has 1.
    assert(format == GL_RGB || format == GL_DEPTH_COMPONENT);
    int bytes_per_pixel, png_format;
    if (format == GL_RGB)
    {
        bytes_per_pixel = 3;
        png_format = PNG_COLOR_TYPE_RGB;
    }
    else
    {
        bytes_per_pixel = 1;
        png_format = PNG_COLOR_TYPE_GRAY;
    }

    // Wait for OpenGL to finish rendering
    glFinish();

    // Read pixels on screen into a raw data buffer
    int num_bytes = image_height * image_width * bytes_per_pixel;
    unsigned char* image_bytes = new unsigned char[num_bytes];
    glReadPixels(0, 0, image_width, image_height, format, GL_UNSIGNED_BYTE,
            image_bytes);

    // Save image as PNG
    writePng(image_fn, image_bytes, image_width, image_height, png_format);

    delete[] image_bytes;
}

void UnscentedKalmanFilter::inferState(const VectorXf u, const VectorXf y,
        const float dt, GLFWwindow* window)
{
    int bytes_per_pixel = 3;    // FIXME
    int num_bytes = image_width * image_height * bytes_per_pixel;
    int NUM_SINGULAR_DIMS = 2; // FIXME

    // Propagate sigma points through process and observation models
    MatrixXf Y = MatrixXf::Zero(y.size(), X.cols());
    for (int i = 0; i < X.cols(); ++i)
    {
        X.col(i) = updateState(X.col(i), u, dt);
        Y.col(i) = generateObservation(window, GL_RGB);
    }

    cout << "Sigma points:" << endl;
    cout << X << endl;

    // Calculate predicted mean and covariance
    VectorXf mean_x = weightedMean(w, X);
    MatrixXf cov_x = weightedCovariance(w, X, mean_x);

    cout << "Expected state:" << endl;
    cout << mean_x << endl;
    cout << "State covariance:" << endl;
    cout << cov_x << endl;

    // Calculate predicted observation and innovation covariance
    VectorXf mean_y = weightedMean(w, Y);
    // This gives us everything we need to do computations with Ky
    // But it avoids storing Ky
    ArrayXf w_sqrt = w.array().sqrt();
    for (int i = 0; i < Y.cols(); ++i)
        // Center and weight the observation samples
        Y.col(i) = w_sqrt(i) * (Y.col(i) - mean_y);
    JacobiSVD<MatrixXf> svd(Y, ComputeThinU);

    // Calculate cross-covariance
    MatrixXf cov_xy = weightedCrossCovariance(w, X, Y, mean_x, mean_y);

    // Perform update using standard Kalman filter equations
    MatrixXf cov_y_inv_sqrt = svd.matrixU(); // We're not done with this yet
    ArrayXf S_inv = svd.singularValues().array().inverse();
    for (int i = 0; i < NUM_SINGULAR_DIMS * 2; ++i)
        S_inv(S_inv.size() - 1 - i) = 0;
    for (int i = 0; i < cov_y_inv_sqrt.cols(); ++i)
        cov_y_inv_sqrt.col(i) = S_inv(i) * cov_y_inv_sqrt.col(i);

    cout << "Singular values:" << endl;
    cout << svd.singularValues() << endl;
    cout << "Inverse singular values:" << endl;
    cout << S_inv << endl;

    MatrixXf cov_x_sqrt = cov_xy * cov_y_inv_sqrt;
    
    // Update estimated moments
    mu_x = mu_x + cov_x_sqrt * cov_y_inv_sqrt.transpose() * (y - mean_y);
    K_x = K_x - cov_x_sqrt * cov_x_sqrt.transpose();

    cout << "New state estimate:" << endl;
    cout << mu_x << endl;

    cout << "Error covariance:" << endl;
    cout << K_x << endl;
}

// Helper functions for math operations
VectorXf weightedMean(const VectorXf weights, const MatrixXf samples)
{
    int n = samples.rows();
    VectorXf mean = VectorXf::Zero(n);

    // TODO: assert dimensions are correct

    for (int i = 0; i < samples.cols(); ++i)
        mean += weights(i) * samples.col(i);

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
