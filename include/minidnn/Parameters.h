#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <Eigen/Core>
#include <vector>
#include "Config.h"
#include "RNG.h"
#include "Optimizer.h"

namespace MiniDNN {

    class Parameters {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        int layer_n;

        ///
        /// Virtual destructor
        ///
        virtual ~Parameters() = default;

        virtual void update_cw(Matrix m_df_data, Vector m_db, Optimizer *opt) = 0;

        virtual Parameters *clone() = 0;

        //virtual Parameters* clone() {return nullptr;}
    };


} // namespace MiniDNN


#endif /* PARAMETERS_H_ */
