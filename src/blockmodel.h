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
#include <random>

using namespace Eigen;
using namespace std;

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

        int statesize;

        /**
         * Constructor.
         * \note The block's initial position in state space has some (not
         * necessarily Gaussian) distribution with mean x0 and covariance K0.
         *
         * \param x Initial location in state space
         * \param c Block color
         * \param a_g Gravitational acceleration vector (in global reference frame)
         */
        BlockModel(const VectorXf x, const VectorXf sigma, const Vector3f c,
                const Vector3f a_g);

        /**
         * Initialize locations for some GLSL uniform variables.
         *
         * \param model_loc Location of the OpenGL uniform matrix representing
         *   the model transformation.
         * \param color_loc Location of the OpenGL uniform vector representing
         *   the block's color.
         * \param offset Index of the first vertex in the block's 3D model.
         */
        void setGlVars(const GLint model_loc, const GLint color_loc,
            const int offset);

        /**
         * Set state parameters.
         *
         * \param s Position in global frame
         * \param v Velocity in global frame
         * \param theta Orientation in global frame
         */
        void setState(const VectorXf x);

        VectorXf getState() const;

        Vector3f observePosition() const { return s; };
        Vector3f observeVelocity() const { return v; };
        Vector3f observeOrientation() const { return theta; };

        /**
         * Process model.
         * Take a step in state space by driving the dynamical system with
         * input u.
         *
         * \param x Location in state space at time t
         * \param u Input vector at time t
         * \param dt Amount of time to calculate update over
         *
         * \return x_new Location in state space at time t + dt
         *
         * \note If x is omitted, the update is performed on the current
         *   state parameters.
         */
        VectorXf updateState(const VectorXf x, const VectorXf u, const float dt);
        void updateState(const VectorXf u, const float dt);

        /**
         * Observation model.
         * Draw block to current OpenGL context.
         */
        void draw() const;

        void printState() const;

    private:

        // Block position and orientation (used when rendering)
        Vector3f s;
        Vector3f v;
        Vector3f theta;

        default_random_engine generator;
        normal_distribution<float> N_s;
        normal_distribution<float> N_v;
        normal_distribution<float> N_theta;

        Vector3f color;

        Vector3f a_gravity;

        // Output image dimensions
        int image_width;
        int image_height;
        int bytes_per_pixel = 3;    // RGB

        // OpenGL variable locations
        GLint model_loc;
        GLint color_loc;
        int offset;

        void initializeNoise(const VectorXf sigma);
};
