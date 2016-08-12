/**
 * Representation of a DUPLO block
 *
 * This class describes the dynamics and properties of a single DUPLO block.
 *
 * HISTORY
 * -------
 * 2016-08-10: Created by Jonathan D. Jones
 */

#include <GL/glew.h>
#include "Eigen/Dense"

using namespace Eigen;

/**
 * Representation of a DUPLO block
 *
 * This class includes methods for animating a block using OpenGL, and for
 * inferring a block's position and orientation in (in 3D metric space) given a
 * sequence of images and inertial measurements as input.
 *
 * The block's state (x) is a vertical concatenation of vectors in R3:
 *          position (in global reference frame)
 *      x = velocity (in global reference frame)
 *          orientation (in global reference frame)
 *  
 * The block's input (u) is a vertical concatenation of vectors in R3:
 *      u = acceleration (in body reference frame)
 *          angular velocity (in body reference frame)
 */
class BlockModel
{
    public:
        /**
         * Constructor.
         * \note The block's initial position in state space has some (not
         * necessarily Gaussian) distribution with mean x0 and covariance K0.
         *
         * \param x0 Mean of initial state distribution
         * \param K0 Covariance of initial state distribution
         * \param c Block color
         * \param a_g Gravitational acceleration vector (in global reference frame)
         */
        BlockModel(const VectorXf x0, const MatrixXf K0, const Vector3f c,
                const Vector3f a_g);

        /**
         * Process model.
         * Take a step in state space by driving the dynamical system with
         * input u.
         *
         * \param u Input vector at time t
         * \param dt Amount of time that has passed since previous update
         */
        void updateState(const VectorXf u, const float dt);

        /**
         * Process model.
         * Take a step in state space by driving the dynamical system with
         * input u.
         *
         * \param x_i Location of the i-th sigma point at time t
         * \param u Input vector at time t
         * \param dt Amount of time that has passed since previous update
         *
         * \return x_new New location of the i-th sigma point
         */
        VectorXf updateState(const VectorXf x_i, const VectorXf u, const float dt)
            const;

        /**
         * \return x The block's current state configuration.
         */
        VectorXf getState() const {return x;}

        /**
         * Observation model.
         * Draw block to current OpenGL context.
         *
         * \param model_loc Location of the OpenGL uniform matrix representing
         *   the model transformation.
         * \param color_loc Location of the OpenGL uniform vector representing
         *   the block's color.
         * \param offset Index of the first vertex in the block's 3D model.
         */
        void draw(const GLint model_loc, const GLint color_loc, const int offset)
            const;

        /**
         * \return out_image Current OpenGL context represented as a w-by-h-by-3
         *    array, where w is the image width and h is the image height, in
         *    pixels.
         */
        Map<MatrixXf> sceneSnapshot() const;

        VectorXf inferState(VectorXf x0, MatrixXf k0, VectorXf u, float dt);

    private:

        // First and second moments of state space distribution
        VectorXf x;
        MatrixXf K;

        // UKF sigma points
        MatrixXf X;
        VectorXf w;

        Vector3f color;

        Vector3f a_gravity;

        // UKF helper functions
        VectorXf weightedMean(const VectorXf, const MatrixXf) const;
        MatrixXf weightedCovariance(const VectorXf, const MatrixXf,
                const VectorXf) const;

        void initializeSigmaPoints(const VectorXf mean, const MatrixXf covariance,
                const float w0);
};

