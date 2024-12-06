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
#include <iostream>
#include <string>
#include <vector>
namespace autoware::mtr::utils
{

inline void print_data(const float * data, const std::string & obj_name)
{
  const std::vector<std::string> state_data_names{"x_",     "y_",      "z_",   "length_",
                                                  "width_", "height_", "yaw_", "vx_",
                                                  "vy_",    "ax_",     "ay_",  "is_valid_"};

  std::cerr << "-----------------\n";
  std::cerr << "Object " << obj_name << "\n";
  size_t i{0};
  for (const auto & name : state_data_names) {
    std::cerr << name << " " << data[i++] << "\n";
  }
  std::cerr << "-----------------\n";
};

}  // namespace autoware::mtr::utils

#endif  // AUTOWARE__MTR__UTILS_HPP_
