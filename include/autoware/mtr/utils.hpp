// Copyright 2024 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef AUTOWARE__MTR__UTILS_HPP_
#define AUTOWARE__MTR__UTILS_HPP_

#include "autoware/mtr/agent.hpp"
#include "autoware/mtr/node.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>

#include <geometry_msgs/msg/detail/pose__struct.hpp>
#include <geometry_msgs/msg/detail/twist__struct.hpp>
#include <geometry_msgs/msg/detail/twist_with_covariance__struct.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <tf2/utils.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace autoware::mtr::utils
{

inline void print_data(
  const float * data, const std::string & obj_name, const size_t num_timestamps = 1)
{
  const std::vector<std::string> state_data_names{"x_",     "y_",      "z_",   "length_",
                                                  "width_", "height_", "yaw_", "vx_",
                                                  "vy_",    "ax_",     "ay_",  "is_valid_"};

  std::cerr << "-----------------\n";
  std::cerr << "Object " << obj_name << "\n";
  for (size_t t = 0; t < num_timestamps; ++t) {
    size_t i{0};
    for (const auto & name : state_data_names) {
      std::cerr << name << " " << data[t * state_data_names.size() + i] << ",";
      ++i;
    }
    std::cerr << "\n-----------------\n";
  }
};

inline void print_agent_data(const AgentData & agent_data)
{
  std::cerr << "num agents " << agent_data.num_agent() << "\n";
  std::cerr << "num targets " << agent_data.num_target() << "\n";
  std::cerr << "time_length_ " << agent_data.time_length() << "\n";
  std::cerr << "ego_index_ " << agent_data.ego_index() << "\n";
  std::cerr << "target index(" << agent_data.target_indices().size() << "): ";
  for (const auto target : agent_data.target_indices()) {
    std::cerr << target << ",";
  }
  std::cerr << "\n";

  std::cerr << "label index(" << agent_data.label_ids().size() << "): ";
  for (const auto label : agent_data.label_ids()) {
    std::cerr << label << ",";
  }
  std::cerr << "\n";
  std::cerr << "timestamps(" << agent_data.timestamps().size() << "): ";
  for (const auto timestamps : agent_data.timestamps()) {
    std::cerr << timestamps << ",";
  }
  std::cerr << "\n";

  const auto & d = agent_data.state_dim();
  const auto & t = agent_data.time_length();
  const auto target_data_ptr = agent_data.target_data_ptr();
  const auto ego_data_ptr = agent_data.ego_data_ptr();
  print_data(ego_data_ptr, "EGO", t);
  for (const auto & idx : agent_data.target_indices()) {
    // Compute the starting address of the n-th object's data
    auto object_data = new float[d * t];
    const float * object_start = target_data_ptr + (idx * d * t);
    std::memcpy(object_data, object_start, d * t * sizeof(float));
    print_data(object_data, "object# " + std::to_string(idx), t);
  }
}

}  // namespace autoware::mtr::utils

#endif  // AUTOWARE__MTR__UTILS_HPP_
