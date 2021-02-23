#ifndef PARAMETERCONTAINER_H_
#define PARAMETERCONTAINER_H_

#include <Eigen/Core>
#include <vector>
#include "Config.h"
#include "RNG.h"
#include "Optimizer.h"
#include "Parameters.h"

namespace MiniDNN {

    class ParameterContainer {
    public:

        long timestamp;
        std::vector<Parameters *> param_list;
        std::atomic<int> active_readers;
        std::atomic<bool> stale_flag;
        bool deleted;

        ///
        /// Constructor
        ///
        ParameterContainer() = default;

        void print() {
            std::cout << std::endl << "ParameterContainer object at: " << this << std::endl;
            for (int i = 0; i < param_list.size(); i++) {
                std::cout << param_list[i] << std::endl;
            }
            std::cout << std::endl;
        }

        void delete_theta() {
            for (int i = 0; i < param_list.size(); ++i) {
                delete param_list[i];
            }
        }

        ///
        /// Virtual destructor
        ///
        virtual ~ParameterContainer() {
            for (int i = 0; i < param_list.size(); ++i) {
                delete param_list[i];
            }
        }

        ParameterContainer(const ParameterContainer &other) {
            for (int i = 0; i < other.param_list.size(); ++i) {
                param_list.push_back(other.param_list[i]->clone());
            }
            timestamp = other.timestamp;
            active_readers.store(0);
            stale_flag = false;
            deleted = false;
        }

    };


} // namespace MiniDNN


#endif /* PARAMETERCONTAINER_H_ */
