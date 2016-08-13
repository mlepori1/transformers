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

VectorXf UnscentedKalmanFilter::inferState(VectorXf u, VectorXf y, float dt)
{
    int bytes_per_pixel = 3;
    int num_bytes = image_width * image_height * bytes_per_pixel;

    // Propagate sigma points through process and observation models
    MatrixXf Y = MatrixXf::Zero(num_bytes, X.cols());
    for (int i = 0; i < X.cols(); ++i)
    {
        // Enable depth rendering, clear the screen to black and clear depth buffer
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Update and render each object in the scene
        VectorXf x_i(X.rows());
        vector<BlockModel>::iterator it;
        int cur_row = 0;
        for (it = blocks.begin(); it != blocks.end(); ++it)
        {
            int n_rows = it->statesize;
            x_i << it->updateState(X.col(i).segment(n_rows, cur_row), u, dt);
            it->draw();

            cur_row += it->statesize;
        }

        // Update sigma points
        X.col(i) = x_i;
        Y.col(i) = sceneSnapshot().cast<float>();
    }

    // Calculate predicted mean and covariance
    VectorXf mu_x = weightedMean(w, X);
    MatrixXf K_x = weightedCovariance(w, X, mu_x);

    // Calculate predicted observation and innovation covariance
    VectorXf mu_y = weightedMean(w, Y);
    // Don't calculate the actual covariance of Y here

    // Calculate cross-covariance
    MatrixXf K_xy = weightedCrossCovariance(w, X, Y, mu_x, mu_y);

    // Perform update using standard Kalman filter equations

    // FIXME
    return VectorXf::Zero(1);
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

Map<VectorXi> UnscentedKalmanFilter::sceneSnapshot(const char* image_fn) const
{
    // FIXME: This is just horribly stupid

    // Wait for OpenGL to finish rendering
    glFinish();

    // Read pixels on screen into a raw data buffer
    int bytes_per_pixel = 3;
    int num_bytes = image_height * image_width * bytes_per_pixel;
    unsigned char* image_bytes = new unsigned char[num_bytes];
    glReadPixels(0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE,
            image_bytes);

    // Convert to vector
    int* image_as_ints = new int[num_bytes];
    memcpy(image_as_ints, image_bytes, num_bytes);
    Map<VectorXi> image(image_as_ints, num_bytes);

    // Save image as PNG
    int bytes_per_row = image_width * bytes_per_pixel;
    unsigned char** row_pointers = new unsigned char*[image_height];
    for (int row = 0; row < image_height; row++)
        row_pointers[image_height - 1 - row] = image_bytes + row * bytes_per_row;
    writePng(image_fn, image_bytes, row_pointers, image_width, image_height);

    delete[] row_pointers;
    delete[] image_bytes;
    delete[] image_as_ints;

    return image;
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
