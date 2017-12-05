#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <iostream>
#include <vector>
#include <stack>
#include <stdexcept>

namespace DG
{
    /** @brief A simple timer
     *
     *  The timer will record a timing event with the start time when tic() is
     *  called. A subsequent call to toc() will record the end time and an
     *  (optional) message to print with the timing. Nesting tic-toc pairs
     *  within one another is permitted as long as each toc() has a
     *  corresponding tic(). Nested timings will be printed with corresponding
     *  indentation.
     *
     *  This should probably be a singleton instead of a class with only static
     *  methods and data.
     */
    class Timer
    {
        public:
            /** @brief A timing event */
            struct Event
            {
                std::chrono::high_resolution_clock::time_point start, end;
                std::string message;
                int depth, id;
            };

            /** @brief Start the timer */
            static void tic()
            {
                Event e;
                e.start = std::chrono::high_resolution_clock::now();
                e.depth = depth++;
                e.id = next_id++;
                events.push(e);
            }

            /** @brief Stop the timer for the matching tic().
             *
             *  @param[in] message : The message to print
             *
             *  @pre A call to tic() must precede toc().
             */
            static void toc(const std::string& message = "")
            {
                auto end = std::chrono::high_resolution_clock::now();

                if (events.empty()) throw std::logic_error("toc() has no matching tic().");
                Event e = events.top();
                e.end = end;
                e.message = message;
                if (spool.empty()) {
                    spool.push_back(e);
                } else {
                    // Order the timing events correctly
                    auto it = spool.begin();
                    while (it != spool.end() && it->id < e.id) ++it;
                    spool.insert(it, e);
                }
                events.pop();
                depth--;

                // Print the spool if possible
                if (active && depth == 0) print();
            }

            static void on()
            {
                active = true;
            }

            static void off()
            {
                active = false;
            }

        private:
            /** @brief Print the spool of timings */
            static void print()
            {
                for (const Event& e : spool) {
                    int ni = e.depth == 0 ? 0 : 4*e.depth-2;
                    std::string indent(ni, ' ');
                    std::string sep;
                    if (!e.message.empty()) sep = ": ";
                    if (!indent.empty()) indent.append("- ");
                    std::chrono::duration<double> duration = e.end - e.start;
                    std::cout << indent << e.message << sep << duration.count() << "s" << std::endl;
                }
                spool.clear();
            }

            static int depth, next_id;
            static bool active;
            static std::stack<Event> events;
            static std::vector<Event> spool;
    };

    int Timer::depth = 0;
    int Timer::next_id = 0;
    bool Timer::active = true;
    std::stack<Timer::Event> Timer::events;
    std::vector<Timer::Event> Timer::spool;
}

#endif
