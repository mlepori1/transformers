/**
 * ----
 *
 * HISTORY
 * -------
 * 2016-08-13: Created by Jonathan D. Jones
 */

#include "blockmodel.h"
#include <vector>

using namespace Eigen;
using namespace std;

/**
 * Use the unscented transform to estimate the state of a linear dynamical system.
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
         * Save the current OpenGL scene.
         *
         * \param image_fn Location where the scene should be saved (as PNG).
         *   This argument is optional, and if omitted, no PNG is created.
         * \return out_image Current OpenGL context represented as a vector
         *    with w*h*3 entries, where w is the image width and h is the image
         *    height (in pixels).
         */
        Map<VectorXi> sceneSnapshot() const;
        Map<VectorXi> sceneSnapshot(const char* image_fn) const;

        VectorXf inferState(VectorXf u, VectorXf y, float dt);


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

        void initializeSigmaPoints(const VectorXf mean, const MatrixXf covariance,
                const float w0);
        /**
         * Process model.
         * Take a step in state space by driving the dynamical system with
         * input u.
         *
         * \param u Input vector at time t
         * \param dt Amount of time that has passed since previous update
         */
        void updateState(const VectorXf u, const float dt) const;
};


VectorXf weightedMean(const VectorXf weights, const MatrixXf samples);

MatrixXf weightedCovariance(const VectorXf weights, const MatrixXf samples,
        const VectorXf mean);

MatrixXf weightedCrossCovariance(const VectorXf w, const MatrixXf X,
        const MatrixXf Y, const VectorXf mu_x, const VectorXf mu_y);
