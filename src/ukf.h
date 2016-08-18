/**
 * ----
 *
 * HISTORY
 * -------
 * 2016-08-13: Created by Jonathan D. Jones
 */

#include "blockmodel.h"
#include <vector>
#include <GLFW/glfw3.h>
#include <GL/glew.h>

using namespace Eigen;
using namespace std;


/**
 * Use the unscented transform to estimate the state of a dynamical system.
 *
 * This implementation is optimized for situations where certain subspaces in
 * state space have independent dynamics, and the dimension of the observation
 * space is much greater than that of the state space.
 */
class UnscentedKalmanFilter
{
    public:

        UnscentedKalmanFilter(const VectorXf x0, const MatrixXf K0,
                const vector<BlockModel> blocks);

        /**
         * Process model.
         * Take a step in state space by driving the dynamical system with
         * input u.
         *
         * \param u Input vector at time t
         * \param dt Amount of time that has passed since previous update
         */
        void updateState(const VectorXf u, const float dt);
        VectorXf updateState(const VectorXf x, const VectorXf u, const float dt);

        VectorXf generateObservation(GLFWwindow* window, const GLenum format)
            const;
        void generateObservation(GLFWwindow* window, const GLenum format,
                const char* image_fn) const;

        void inferState(const VectorXf u, const VectorXf y,  const float dt,
                GLFWwindow* window);

        VectorXf getState() const;
        VectorXf getStateEstimate() { return mu_x; } const;
        MatrixXf getErrorCovariance() { return K_x; } const;


    private:

        vector<BlockModel> blocks;

        // First and second moments of state space distribution
        VectorXf mu_x;
        MatrixXf K_x;

        // UKF sigma points and weights
        MatrixXf X;
        VectorXf w;

        // For reading pixels from the OpenGL context
        int image_width;
        int image_height;


        // (UKF helper functions below)
        /**
         * Save the current OpenGL scene.
         *
         * \param image_fn Location where the scene should be saved (as PNG).
         *   This argument is optional, and if omitted, no PNG is created.
         * \return out_image Current OpenGL context represented as a vector
         *    with w*h*3 entries, where w is the image width and h is the image
         *    height (in pixels).
         */
        VectorXf sceneSnapshot(const GLenum format) const;
        void sceneSnapshot(const GLenum format, const char* image_fn) const;

        void initializeSigmaPoints(const VectorXf mean, const MatrixXf covariance,
                const float w0);
};


VectorXf weightedMean(const VectorXf weights, const MatrixXf samples);

MatrixXf weightedCovariance(const VectorXf weights, const MatrixXf samples,
        const VectorXf mean);

MatrixXf weightedCrossCovariance(const VectorXf w, const MatrixXf X,
        const MatrixXf Y, const VectorXf mu_x, const VectorXf mu_y);
