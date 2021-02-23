#ifndef THREADPOOL_H_
#define THREADPOOL_H_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>


#include <utils.h>

namespace MiniDNN {
    class ThreadPool {
    public:
        ThreadPool(int n_threads, std::vector<std::function<void(int id)>> &jobs) :
                num_threads(n_threads),
                tasks(jobs) {
            this->threads.resize(num_threads);
            this->flags.resize(num_threads);
            for (int i = 0; i < num_threads; ++i) {
                this->flags[i] = std::make_shared<std::atomic<bool>>(false);
                this->set_thread(i);
            }
        }

        void stop(bool isWait = false) {
            this->isDone = true;
            for (int i = 0, n = this->flags.size(); i < n; ++i) {
                *this->flags[i] = true;  // command the threads to stop
            }

            {
                std::unique_lock<std::mutex> lock(this->m_mutex);
                this->tasks_are_available.notify_all();  // stop all waiting threads
            }

            for (int i = 0; i < static_cast<int>(this->threads.size()); ++i) {
                // wait for the computing threads to finish
                if (this->threads[i]->joinable())
                    this->threads[i]->join();
            }
            this->threads.clear();
            this->flags.clear();
        }

        void wait_for_all() {
            std::unique_lock<std::mutex> lock(this->m_mutex);
            workers_are_availble.wait(lock, [this] { return this->num_waiting == this->num_threads; });
            this->num_waiting = 0;
        }

        void wait() {
            std::unique_lock<std::mutex> lock(this->m_mutex);
            workers_are_availble.wait(lock, [this] { return this->num_waiting > 0; });
            --this->num_waiting;
        }

        void start_all() {
            for (int i = 0, n = this->flags.size(); i < n; ++i) {
                *this->flags[i] = true;  // command the threads to stop
            }
            std::unique_lock<std::mutex> lock(this->m_mutex);
            this->tasks_are_available.notify_all();  // wake all waiting threads
        }


    private:

        void set_thread(int i) {
            std::shared_ptr<std::atomic<bool>> flag(this->flags[i]);

            auto f = [this, i, flag/* a copy of the shared ptr to the flag */]() {
                std::atomic<bool> &_flag = *flag;
                // set_cpu(i);
                while (true) {
                    {
                        std::unique_lock<std::mutex> lock(this->m_mutex);
                        ++this->num_waiting;
                        this->workers_are_availble.notify_one();
                        this->tasks_are_available.wait(lock, [this, i, &_flag]() { return _flag.load(); });
                    }

                    if (this->isDone) {
                        return;
                    }

                    {
                        // For DEBUGGING uncomment below
                        // std::unique_lock<std::mutex> lock1(this->m_mutex);
                        this->tasks[i](i);
                    }
                    _flag = false;
                }
            };

            this->threads[i].reset(new std::thread(f));
        }

        std::atomic<bool> isDone{false};
        std::atomic<int> num_waiting{0};  // how many threads are waiting
        size_t num_threads{0};
        std::vector<std::unique_ptr<std::thread>> threads;
        std::vector<std::shared_ptr<std::atomic<bool>>> flags;

        std::vector<std::function<void(int id)>> &tasks;

        std::mutex m_mutex;
        std::condition_variable tasks_are_available;
        std::condition_variable workers_are_availble;
    };
} // namespace MiniDNN


#endif /* THREADPOOL_H_ */
