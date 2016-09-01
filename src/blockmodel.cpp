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
using namespace std;


BlockModel::BlockModel(const VectorXf x, const VectorXf sigma,
        const Vector3f c, const Vector3f a_g)
{
    assert(x.size() == sigma.size());

    initializeNoise(sigma);

    setState(x);

    color = c;
    a_gravity = a_g;
}

void BlockModel::printState() const
{
    cout << " s: " << s.transpose()
         << " v: " << v.transpose()
         << " t: " << theta.transpose()
         << endl;
}

// FIXME: Can only initialize an independent Gaussian random vector
void BlockModel::initializeNoise(const VectorXf sigma)
{
    // Process noise
    VectorXf sigma_s     = sigma.segment<3>(0);
    VectorXf sigma_v     = sigma.segment<3>(3);
    VectorXf sigma_theta = sigma.segment<3>(6);

    // FIXME: EXTREMELY BAD -- I ignore all but the first entry of each sigma
    N_s     = normal_distribution<float>(0.0, sigma_s(0));
    N_v     = normal_distribution<float>(0.0, sigma_v(0));
    N_theta = normal_distribution<float>(0.0, sigma_theta(0));
}

void BlockModel::setGlVars(const GLint model_loc, const GLint color_loc,
        const int offset)
{
    this->model_loc = model_loc;
    this->color_loc = color_loc;
    this->offset = offset;
}

void BlockModel::setState(const VectorXf x_a)
{
    // Augmented state concatenates state and process noise
    statesize = x_a.size();

    // State
    s     = x_a.segment<3>(0);
    v     = x_a.segment<3>(3);
    theta = x_a.segment<3>(6);
}

VectorXf BlockModel::getState() const
{
    // State vector
    VectorXf x(s.size() + v.size() + theta.size());
    x << s, v, theta;

    return x;
}

VectorXf BlockModel::updateState(const VectorXf x, const VectorXf u, const float dt)
{
    setState(x);
    updateState(u, dt);

    return getState();
}

void BlockModel::updateState(const VectorXf u, const float dt)
{
    // Process noise is drawn IID from zero-mean multivariate Gaussian
    // FIXME: Assuming additive noise is IID doesn't make sense when the state
    //   vector represents a reference frame (maybe?)
    Vector3f n_s (N_s(generator), N_s(generator), N_s(generator));
    Vector3f n_v (N_v(generator), N_v(generator), N_v(generator));
    Vector3f n_theta (N_theta(generator), N_theta(generator), N_theta(generator));

    // Split input vector
    Vector3f a     = u.segment<3>(0) - a_gravity;
    Vector3f omega = u.segment<3>(3);

    Vector3f theta_next = theta + omega * dt; // + n_theta;

    // FIXME: Make sure orientation and angular velocity are in radians
    AngleAxis<float> Cx(theta(0), Vector3f::UnitX());
    AngleAxis<float> Cy(theta(1), Vector3f::UnitY());
    AngleAxis<float> Cz(theta(2), Vector3f::UnitZ());
    //Vector3f a_global = Cz * Cy * Cx * a - a_gravity;
    //cout << a_global.transpose() << endl;
    Vector3f a_global = Cz * Cy * Cz * a;

    Vector3f v_next = v + a_global * dt; // + n_v;
    Vector3f s_next = s + v * dt; // + n_s;

    // Update state vector
    s     = s_next;
    v     = v_next;
    theta = theta_next;
}

void BlockModel::draw() const
{
    AngleAxis<float> Mx(theta(0), Vector3f(1.0f, 0.0f, 0.0f));
    AngleAxis<float> My(theta(1), Vector3f(0.0f, 1.0f, 0.0f));
    AngleAxis<float> Mz(theta(2), Vector3f(0.0f, 0.0f, 1.0f));
    Transform<float,3,Affine> t = Translation<float,3>(s) * Mx * My * Mz;
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, t.data());

    glUniform3f(color_loc, color(0), color(1), color(2));

    int num_vertices = 36;  // number of vertices in block model
    glDrawArrays(GL_TRIANGLES, offset, num_vertices);
}
