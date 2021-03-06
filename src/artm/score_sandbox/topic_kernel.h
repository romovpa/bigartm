// Copyright 2014, Additive Regularization of Topic Models.

#ifndef SRC_ARTM_SCORE_SANDBOX_TOPIC_KERNEL_H_
#define SRC_ARTM_SCORE_SANDBOX_TOPIC_KERNEL_H_

#include <string>
#include <vector>

#include "artm/messages.pb.h"
#include "artm/score_calculator_interface.h"

namespace artm {
namespace score_sandbox {

class TopicKernel : public ScoreCalculatorInterface {
 public:
  explicit TopicKernel(const TopicKernelScoreConfig& config)
    : config_(config) {}

  std::shared_ptr<Score> CalculateScore(const artm::core::TopicModel& topic_model);

  virtual bool is_cumulative() const { return false; }

  virtual ScoreData_Type score_type() const { return ::artm::ScoreData_Type_TopicKernel; }

 private:
  TopicKernelScoreConfig config_;
};

}  // namespace score_sandbox
}  // namespace artm

#endif  // SRC_ARTM_SCORE_SANDBOX_TOPIC_KERNEL_H_
