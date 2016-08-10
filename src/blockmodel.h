/*
 * blockmodel.h
 *   This class describes the dynamics and properties of a single DUPLO block.
 *
 * HISTORY
 * -------
 * 2016-08-10: Created by Jonathan D. Jones
 */

#include <GL/glew.h>
#include "Eigen/Dense"

using namespace Eigen;

class BlockModel
{
    public:
        BlockModel(const Vector3f, const Vector3f, const Vector3f,
                   const Vector3f, const Vector3f);

        void initializeState(const Vector3f, const Vector3f, const Vector3f);
        void updateState(const Vector3f, const Vector3f, const float);

        Vector3f getPosition() const {return s;};
        Vector3f getVelocity() const {return v;};
        Vector3f getOrientation() const {return theta;};

        void draw(const GLint, const GLint, const int) const;

        void initializeSigmaPoints(const VectorXf, const MatrixXf, const float);

    private:
        Vector3f v;
        Vector3f s;
        Vector3f theta;

        Vector3f a_gravity;

        Vector3f color;

        MatrixXf X;     // Sigma points
        VectorXf W;     // Sigma point weights
};

