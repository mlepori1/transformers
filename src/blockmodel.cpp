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


BlockModel::BlockModel(const VectorXf x, const Vector3f c, const Vector3f a_g)
{
    setState(x);

    color = c;
    a_gravity = a_g;

}

void BlockModel::printState()
{
    cout << " s: " << s.transpose()
         << " v: " << v.transpose()
         << " t: " << theta.transpose()
         << endl;
}

void BlockModel::setGlVars(const GLint model_loc, const GLint color_loc,
        const int offset)
{
    this->model_loc = model_loc;
    this->color_loc = color_loc;
    this->offset = offset;
}

void BlockModel::setState(const VectorXf x)
{
    statesize = x.size();

    s     = x.segment<3>(0);
    v     = x.segment<3>(3);
    theta = x.segment<3>(6);
}

VectorXf BlockModel::getState() const
{
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
    // Split input vector
    Vector3f a     = u.segment<3>(0);
    Vector3f omega = u.segment<3>(3);

    Vector3f theta_next = theta + omega * dt;

    // FIXME: maybe using previous theta here accidentally
    // FIXME: Make sure orientation and angular velocity are in radians
    AngleAxis<float> Cx(theta(0), Vector3f(1.0f, 0.0f, 0.0f));
    AngleAxis<float> Cy(theta(1), Vector3f(0.0f, 1.0f, 0.0f));
    AngleAxis<float> Cz(theta(2), Vector3f(0.0f, 0.0f, 1.0f));
    Vector3f a_global = Cx * Cy * Cz * a - a_gravity;

    Vector3f v_next = v + a_global * dt;
    Vector3f s_next = s + v * dt;

    // Update state vector
    s = s_next;
    v = v_next;
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
