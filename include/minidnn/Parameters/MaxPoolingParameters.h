#ifndef MAXPOOLINGPARAMETERS_H_
#define MAXPOOLINGPARAMETERS_H_

#include <Eigen/Core>
#include <vector>
#include "Config.h"
#include "RNG.h"
#include "Optimizer.h"
#include "Parameters.h"

namespace MiniDNN {

    class MaxPoolingParameters : public Parameters {
    public:

        ///
        /// Virtual destructor
        ///
        virtual ~MaxPoolingParameters() = default;

        ///
        /// Constructor
        ///
        MaxPoolingParameters() = default;

        void update_cw(Matrix m_df_data, Vector m_db, Optimizer *opt) {}

        MaxPoolingParameters(MaxPoolingParameters &) = default;

        MaxPoolingParameters *clone() override {
            return new MaxPoolingParameters(*this);
        }

    };


} // namespace MiniDNN


#endif /* MAXPOOLINGPARAMETERS_H_ */
