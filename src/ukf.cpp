#include "ukf.h"

using namespace Eigen;
using namespace std;


UnscentedKalmanFilter::UnscentedKalmanFilter(const VectorXf x0, const MatrixXf K0,
        const vector<BlockModel> blocks)
{
    debug = false;

    // Set first and second moments of initial state space distribution
    mu_x = x0;
    K_x = K0;

    this->blocks = blocks;

    // Set image dimensions
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    image_width = viewport[2];
    image_height = viewport[3];
}

int UnscentedKalmanFilter::stateSize() const
{
    vector<BlockModel>::const_iterator it;
    int statesize = 0;
    for (it = blocks.begin(); it != blocks.end(); ++it)
        statesize += it->statesize;

    return statesize;
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

    VectorXf image = sceneSnapshot(format, image_fn);

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();

    return image;
}

VectorXf UnscentedKalmanFilter::observePosition() const
{
    Vector3f position;

    vector<BlockModel>::const_iterator it;
    for (it = blocks.begin(); it != blocks.end(); ++it)
        position << it->observePosition();

    return position;
}

VectorXf UnscentedKalmanFilter::observeOrientation() const
{
    Vector3f orientation;

    vector<BlockModel>::const_iterator it;
    for (it = blocks.begin(); it != blocks.end(); ++it)
        orientation << it->observeOrientation();

    return orientation;
}

VectorXf UnscentedKalmanFilter::sceneSnapshot(const GLenum format,
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

    VectorXf image = toVector(image_bytes, num_bytes);

    // Save image as PNG
    if (image_fn)
        writePng(image_fn, image_bytes, image_width, image_height, png_format);

    delete[] image_bytes;

    return image;
}

void UnscentedKalmanFilter::setSigmaPoints(const float w0)
{
    // Calculate matrix square root used in sigma point calculations
    float n = mu_x.size();
    MatrixXf K_scaled = n / (1 - w0) * K_x;
    MatrixXf K_scaled_sqrt = K_scaled.llt().matrixL(); // Cholesky decomposition

    if (debug)
    {
        //cout << "Dimensionality of state vector: " << n << endl;
        //cout << "Number of sigma points: " << 2 * n + 1 << endl;
    }

    // Define matrix with sigma points as columns
    X = MatrixXf::Zero(n, 2 * n + 1);
    X.col(0) = mu_x;
    for (int i = 1; i <= n; ++i)
    {
        X.col(i) = mu_x + K_scaled_sqrt.col(i - 1);
        X.col(i + n) = mu_x - K_scaled_sqrt.col(i - 1);
    }

    // Define vector of sigma point weights
    w = VectorXf::Zero(2 * n + 1);
    float w_constant = (1 - w0) / (2 * n);
    w.fill(w_constant);
    w(0) = w0;
}

void UnscentedKalmanFilter::inferState(const VectorXf u, const VectorXf y,
        const configParams params)
{
    MatrixXf Q = params.q * MatrixXf::Identity(mu_x.size(), mu_x.size());
    MatrixXf R = params.r * MatrixXf::Identity(y.size(), y.size());

    setSigmaPoints(params.w0);

    // Propagate sigma points through process and observation models
    MatrixXf Y = MatrixXf::Zero(y.size(), X.cols());
    for (int i = 0; i < X.cols(); ++i)
    {
        X.col(i) = updateState(X.col(i), u, params.dt);

        if (params.observation == "position")
            Y.col(i) = observePosition();
        else if (params.observation == "orientation")
            Y.col(i) = observeOrientation();
    }

    // Calculate predicted means and covariances
    VectorXf mean_x = weightedMean(w, X);
    VectorXf mean_y = weightedMean(w, Y);
    MatrixXf cov_xx = weightedCovariance(w, X, mean_x) + Q;
    MatrixXf cov_yy = weightedCovariance(w, Y, mean_y) + R;
    MatrixXf cov_xy = weightedCrossCovariance(w, X, Y, mean_x, mean_y);

    // solve for the transpose of the Kalman gain matrix
    MatrixXf GT = cov_yy.transpose().ldlt().solve(cov_xy.transpose());
    mu_x = mean_x + GT.transpose() * (y - mean_y);
    K_x  = cov_xx - GT.transpose() * cov_yy * GT;

    if (debug)
    {
        cout << "INPUT (U)  : ";
        cout << u.transpose() << endl;
        cout << "INNOVATION : ";
        cout << (y - mean_y).transpose() << endl;
        cout << "X UPDATE   : ";
        cout << (GT.transpose() * (y - mean_y)).transpose() << endl;
        cout << "X EXPECTED : ";
        cout << mean_x.transpose() << endl;
        cout << "X ESTIMATED: ";
        cout << mu_x.transpose() << endl;
    }
}

void UnscentedKalmanFilter::inferState(const VectorXf u, const VectorXf y,
        const configParams params, GLFWwindow* window, const string fn_prefix)
{
    MatrixXf Q = params.q * MatrixXf::Identity(mu_x.size(), mu_x.size());

    setSigmaPoints(params.w0);

    // Propagate sigma points through process and observation models
    stringstream ss;
    string image_fn;
    MatrixXf Y = MatrixXf::Zero(y.size(), X.cols());
    for (int i = 0; i < X.cols(); ++i)
    {
        X.col(i) = updateState(X.col(i), u, params.dt);
        if (!fn_prefix.empty() && debug)
        {
            ss << fn_prefix << i << ".png";
            image_fn = ss.str();
            ss.str("");
            Y.col(i) = generateObservation(window, GL_RGB, image_fn.c_str());
        } else
            Y.col(i) = generateObservation(window, GL_RGB, NULL);
    }

    // Calculate predicted means and covariances for terms that permit direct
    // computation
    VectorXf mean_x = weightedMean(w, X);
    VectorXf mean_y = weightedMean(w, Y);
    MatrixXf cov_x  = weightedCovariance(w, X, mean_x) + Q;
    MatrixXf cov_xy = weightedCrossCovariance(w, X, Y, mean_x, mean_y);

    // This gives us everything we need to do computations with Ky
    // But it avoids storing Ky
    ArrayXf w_sqrt = w.array().sqrt();
    for (int i = 0; i < Y.cols(); ++i)
        // Center and weight the observation samples
        Y.col(i) = w_sqrt(i) * (Y.col(i) - mean_y);
    JacobiSVD<MatrixXf> svd(Y, ComputeThinU);

    // Perform update using standard Kalman filter equations
    // (modulo some algebraic manipulation to make the updates computable)
    float r_inv = 1.0 / params.r;
    ArrayXf S = svd.singularValues().array();
    MatrixXf S_tilde_inv = ((S.pow(2) + params.r * params.r).inverse() - r_inv)
        .matrix().asDiagonal();

    MatrixXf A = cov_xy * svd.matrixU();
    MatrixXf B = A * S_tilde_inv * svd.matrixU().transpose()
        + r_inv * cov_xy;

    VectorXf update = B * (y - mean_y);
    mu_x = mean_x + update;
    K_x = cov_x - A * S_tilde_inv * A.transpose()
        - r_inv * cov_xy * cov_xy.transpose();

    if (debug)
    {
        JacobiSVD<MatrixXf> svd_B(B, ComputeThinU | ComputeThinV);

        cout << "INPUT (U)  : ";
        cout << u.transpose() << endl;
        cout << "SING VALS Y: ";
        cout << S.transpose() << endl;
        cout << "SVALS Y INV: ";
        cout << S_tilde_inv.diagonal().transpose() << endl;
        cout << "I NORM     : ";
        cout << (y - mean_y).norm() << endl;
        cout << "I PROJECTED: ";
        cout << (y - mean_y).transpose() * svd_B.matrixV() << endl;
        cout << "SIGMA B    : ";
        cout << svd_B.singularValues().transpose() << endl;
        cout << "X UPDATE   : ";
        cout << (update).transpose() << endl;
        cout << "X EXPECTED : ";
        cout << mean_x.transpose() << endl;
        cout << "X ESTIMATED: ";
        cout << mu_x.transpose() << endl;
    }
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
