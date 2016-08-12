/*
 * blockmodel.cpp
 *   This class describes the dynamics and properties of a single DUPLO block.
 *
 * HISTORY
 * -------
 * 2016-08-10: Created by Jonathan D. Jones
 */

#include "blockmodel.h"
#include "libIO.h"

using namespace Eigen;

BlockModel::BlockModel(const VectorXf x0, const MatrixXf K0,
        const Vector3f c, const Vector3f a_g)
{
    // Set first and second moments of initial state space distribution
    x = x0;
    K = K0;

    // Set sigma points
    float w0 = 1.0 / 3.0;
    initializeSigmaPoints(x0, K0, w0);

    a_gravity = a_g;
    color = c;

    // Set image dimensions
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int image_width = viewport[2];
    int image_height = viewport[3];

}

void BlockModel::initializeSigmaPoints(const VectorXf mean,
        const MatrixXf covariance, const float w0)
{
    // Calculate matrix square root used in sigma point calculations
    float n = mean.size();
    MatrixXf M = n / (1 - w0) * covariance;
    MatrixXf L = M.llt().matrixL(); // Cholesky decomposition

    // Define matrix with sigma points as columns
    X = MatrixXf::Zero(n, 2 * n + 1);
    X.col(0) = mean;
    for (int i = 1; i <= n; ++i)
    {
        X.col(i) = mean + L.col(i - 1);
        X.col(i + n) = mean - L.col(i - 1);
    }

    // Define vector of sigma point weights
    w = VectorXf::Zero(2 * n + 1);
    float w_constant = (1 - w0) / (2 * n);
    w.fill(w_constant);
    w(0) = w0;
}

void BlockModel::updateState(const VectorXf u, const float dt)
{
    // Split state vector
    Vector3f s     = x.segment<3>(0);
    Vector3f v     = x.segment<3>(3);
    Vector3f theta = x.segment<3>(6);

    // Split input vector
    Vector3f a     = u.segment<3>(0);
    Vector3f omega = u.segment<3>(3);

    Vector3f theta_next = theta + omega * dt;

    // FIXME: maybe using previous theta here accidentally
    // FIXME: Make sure orientation and angular velocity are in radians
    AngleAxis<float> Cx(theta[0], Vector3f(1.0f, 0.0f, 0.0f));
    AngleAxis<float> Cy(theta[1], Vector3f(0.0f, 1.0f, 0.0f));
    AngleAxis<float> Cz(theta[2], Vector3f(0.0f, 0.0f, 1.0f));
    Vector3f a_global = Cx * Cy * Cz * a - a_gravity;

    Vector3f v_next = v + a_global * dt;
    Vector3f s_next = s + v * dt;

    // Update state vector
    x << s_next, v_next, theta_next;
}

VectorXf BlockModel::updateState(const VectorXf x_i, const VectorXf u, const float dt)
    const
{
    // Split state vector
    Vector3f s     = x_i.segment<3>(0);
    Vector3f v     = x_i.segment<3>(3);
    Vector3f theta = x_i.segment<3>(6);

    // Split input vector
    Vector3f a     = u.segment<3>(0);
    Vector3f omega = u.segment<3>(3);

    Vector3f theta_next = theta + omega * dt;

    // FIXME: maybe using previous theta here accidentally
    // FIXME: Make sure orientation and angular velocity are in radians
    AngleAxis<float> Cx(theta[0], Vector3f(1.0f, 0.0f, 0.0f));
    AngleAxis<float> Cy(theta[1], Vector3f(0.0f, 1.0f, 0.0f));
    AngleAxis<float> Cz(theta[2], Vector3f(0.0f, 0.0f, 1.0f));
    Vector3f a_global = Cx * Cy * Cz * a - a_gravity;

    Vector3f v_next = v + a_global * dt;
    Vector3f s_next = s + v * dt;

    // Update state vector
    VectorXf x_new(s_next.size() + v_next.size() + theta_next.size());
    x_new << s_next, v_next, theta_next;

    return x_new;
}

void BlockModel::setGlVars(const GLint model_loc, const GLint color_loc,
        const int offset)
{
    this->model_loc = model_loc;
    this->color_loc = color_loc;
    this->offset = offset;
}

void BlockModel::draw() const
{
    // Split state vector
    Vector3f s     = x.segment<3>(0);
    Vector3f theta = x.segment<3>(6);

    AngleAxis<float> Mx(theta[0], Vector3f(1.0f, 0.0f, 0.0f));
    AngleAxis<float> My(theta[1], Vector3f(0.0f, 1.0f, 0.0f));
    AngleAxis<float> Mz(theta[2], Vector3f(0.0f, 0.0f, 1.0f));
    Transform<float,3,Affine> t = Translation<float,3>(s) * Mx * My * Mz;
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, t.data());

    glUniform3f(color_loc, color[0], color[1], color[2]);

    int num_vertices = 36;  // number of vertices in block model
    glDrawArrays(GL_TRIANGLES, offset, num_vertices);
}

void BlockModel::draw(const VectorXf x_i) const
{
    // Enable depth rendering, clear the screen to black and clear depth buffer
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Split state vector
    Vector3f s     = x_i.segment<3>(0);
    Vector3f theta = x_i.segment<3>(6);

    AngleAxis<float> Mx(theta[0], Vector3f(1.0f, 0.0f, 0.0f));
    AngleAxis<float> My(theta[1], Vector3f(0.0f, 1.0f, 0.0f));
    AngleAxis<float> Mz(theta[2], Vector3f(0.0f, 0.0f, 1.0f));
    Transform<float,3,Affine> t = Translation<float,3>(s) * Mx * My * Mz;
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, t.data());

    glUniform3f(color_loc, color[0], color[1], color[2]);

    int num_vertices = 36;  // number of vertices in block model
    glDrawArrays(GL_TRIANGLES, offset, num_vertices);
}

Map<VectorXi> BlockModel::sceneSnapshot() const
{
    // Wait for OpenGL to finish rendering
    glFinish();

    // Read pixels on screen into a raw data buffer
    int num_bytes = image_height * image_width * bytes_per_pixel;
    int* image_as_ints = new int[num_bytes];
    glReadPixels(0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_INT, image_as_ints);
    
    // Convert to vector
    Map<VectorXi> image(image_as_ints, num_bytes);

    delete[] image_as_ints;

    return image;
}

Map<VectorXi> BlockModel::sceneSnapshot(const char* image_fn) const
{
    // FIXME: This is just horribly stupid

    // Wait for OpenGL to finish rendering
    glFinish();

    // Read pixels on screen into a raw data buffer
    int num_bytes = image_height * image_width * bytes_per_pixel;
    unsigned char* image_bytes = new unsigned char[num_bytes];
    glReadPixels(0, 0, image_width, image_height, GL_RGB, GL_UNSIGNED_BYTE, image_bytes);

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

VectorXf BlockModel::inferState(VectorXf u, VectorXf y, float dt)
{
    // TODO

    // Propagate sigma points through process and observation models
    MatrixXf X_new = MatrixXf::Zero(X.rows(), X.cols());
    MatrixXi Y_new = MatrixXi::Zero(image_width * image_height * bytes_per_pixel, X.cols());
    for (int i = 0; i < X.cols(); ++i)
    {
        X_new.col(i) = updateState(X.col(i), u, dt);
        draw(X_new.col(i));
        Y_new.col(i) = sceneSnapshot();
    }

    // Calculate predicted mean and covariance
    VectorXf X_mean = weightedMean(w, X);
    MatrixXf X_covariance = weightedCovariance(w, X, X_mean);

    // Calculate predicted observation and innovation covariance
    // Calculate cross-covariance
    // Perform update using standard Kalman filter equations

    // FIXME
    return VectorXf::Zero(1);
}

VectorXf BlockModel::weightedMean(const VectorXf weights, const MatrixXf samples)
    const
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

MatrixXf BlockModel::weightedCovariance(const VectorXf weights,
        const MatrixXf samples, const VectorXf mean) const
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
