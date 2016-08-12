/*
 * blockmodel.cpp
 *   This class describes the dynamics and properties of a single DUPLO block.
 *
 * HISTORY
 * -------
 * 2016-08-10: Created by Jonathan D. Jones
 */

#include "blockmodel.h"

using namespace Eigen;

BlockModel::BlockModel(const VectorXf x0, const MatrixXf K0,
        const Vector3f c, const Vector3f a_g)
{
    x = x0;
    K = K0;

    a_gravity = a_g;
    color = c;
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

void BlockModel::draw(const GLint model_loc, const GLint color_loc,
        const int offset) const
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

Map<MatrixXf> BlockModel::sceneSnapshot() const
{
    // Save image
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int w = viewport[2];
    int h = viewport[3];

    glFinish();

    int bytes_per_pixel = 3;
    float* out_channel = new float[h * w];
    glReadPixels(0, 0, w, h, GL_RED, GL_FLOAT, out_channel);

    // Convert to matrix
    Map<MatrixXf> map(out_channel, h, w);

    delete[] out_channel;

    return map;
}

VectorXf BlockModel::inferState(VectorXf x0, MatrixXf k0, VectorXf u, float dt)
{
    // TODO

    // 1. Initialize sigma points and weights
    //    (should be done already)

    // 2. Propagate sigma points through process model
    MatrixXf X_new = MatrixXf::Zero(X.rows(), X.cols());
    for (int i = 0; i < X.cols(); ++i)
    {
        X_new.col(i) = updateState(X.col(i), u, dt);
    }

    // 3. Calculate predicted mean
    VectorXf X_mean = weightedMean(w, X);

    // 4. Calculate predicted covariance
    MatrixXf X_covariance = weightedCovariance(w, X, X_mean);

    // 5. Propagate sigma points through observation model
    // 6. Calculate predicted observation
    // 7. Calculate innovation covariance
    // 8. Calculate cross-covariance matrix
    // 9. Perform update using standard Kalman filter equations

    // FIXME
    return VectorXf::Zero(1);
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
        X.col(i) = mean + L.col(i);
        X.col(i + n) = mean - L.col(i + n);
    }

    // Define vector of sigma point weights
    w = VectorXf::Zero(2 * n + 1);
    float w_constant = (1 - w0) / (2 * n);
    w.fill(w_constant);
    w(0) = w0;
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
