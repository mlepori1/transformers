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

BlockModel::BlockModel(const Vector3f s0, const Vector3f v0,
        const Vector3f theta0, const Vector3f c, const Vector3f a_g)
{
    a_gravity = a_g;
    color = c;
    initializeState(s0, v0, theta0);
}

void BlockModel::initializeState(const Vector3f s0, const Vector3f v0,
        const Vector3f theta0)
{
    s = s0;
    v = v0;
    theta = theta0;
}

void BlockModel::updateState(const Vector3f a, const Vector3f w, const float dt)
{
    Vector3f theta_next = theta + w * dt;

    // FIXME: maybe using previous theta here accidentally
    // FIXME: Make sure orientation and angular velocity are in radians
    AngleAxis<float> Cx(theta[0], Vector3f(1.0f, 0.0f, 0.0f));
    AngleAxis<float> Cy(theta[1], Vector3f(0.0f, 1.0f, 0.0f));
    AngleAxis<float> Cz(theta[2], Vector3f(0.0f, 0.0f, 1.0f));
    Vector3f a_global = Cx * Cy * Cz * a - a_gravity;

    Vector3f v_next = v + a_global * dt;
    Vector3f s_next = s + v * dt;

    theta = theta_next;
    v = v_next;
    s = s_next;
}

void BlockModel::draw(const GLint model_loc, const GLint color_loc,
        const int offset) const
{
    AngleAxis<float> Mx(theta[0], Vector3f(1.0f, 0.0f, 0.0f));
    AngleAxis<float> My(theta[1], Vector3f(0.0f, 1.0f, 0.0f));
    AngleAxis<float> Mz(theta[2], Vector3f(0.0f, 0.0f, 1.0f));
    Transform<float,3,Affine> t = Translation<float,3>(s) * Mx * My * Mz;
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, t.data());

    glUniform3f(color_loc, color[0], color[1], color[2]);

    glDrawArrays(GL_TRIANGLES, offset, 6 * 6);
}

void BlockModel::initializeSigmaPoints(const VectorXf mean,
        const MatrixXf covariance, const float w0)
{
    // Calculate matrix square root used in sigma point calculations
    float Nx = mean.size();
    LLT<MatrixXf> CholeskyDecomp(Nx / (1 - w0) * covariance);
    MatrixXf L = CholeskyDecomp.matrixL();

    W = VectorXf::Zero(2 * Nx + 1);
    X = MatrixXf::Zero(2 * Nx + 1, Nx);

    // Set sigma points and weights...
    W(0) = w0;
    X.row(0) = mean;
    for (int i = 1; i <= Nx; ++i)
    {
        X.row(i) = mean + L.col(i);
        X.row(i + Nx) = mean - L.col(i + Nx);

        W(i) = 1 - w0 / (2 * Nx);
        W(i + Nx) = 1 - w0 / (2 * Nx);
    }
}
