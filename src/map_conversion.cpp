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

#include "autoware/mtr/map_conversion.hpp"

#include "autoware/mtr/polyline.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace autoware::mtr
{

std::vector<LanePoint> from_linestring(const lanelet::ConstLineString3d & linestring)
{
  if (linestring.size() == 0) {
    return {};
  }

  const auto & start = linestring.begin();
  std::vector<LanePoint> points{
    {static_cast<float>(start->x()), static_cast<float>(start->y()), static_cast<float>(start->z()),
     0.0f, 0.0f, 0.0f}};
  points.reserve(linestring.size());
  for (auto itr = start + 1; itr != linestring.end(); ++itr) {
    const auto dx = (itr)->x() - (itr - 1)->x();
    const auto dy = (itr)->y() - (itr - 1)->y();
    const auto dz = (itr)->z() - (itr - 1)->z();
    const auto norm = std::hypot(dx, dy, dz);
    points.emplace_back(
      static_cast<float>(itr->x()), static_cast<float>(itr->y()), static_cast<float>(itr->z()),
      static_cast<float>(dx / norm), static_cast<float>(dy / norm), static_cast<float>(dz / norm));
  }
  return points;
}

std::vector<LanePoint> from_polygon(const lanelet::CompoundPolygon3d & polygon)
{
  if (polygon.size() == 0) {
    return {};
  }

  const auto & start = polygon.begin();
  std::vector<LanePoint> points{
    {static_cast<float>(start->x()), static_cast<float>(start->y()), static_cast<float>(start->z()),
     0.0f, 0.0f, 0.0f}};
  points.reserve(polygon.size());
  for (auto itr = start + 1; itr != polygon.end(); ++itr) {
    const auto dx = (itr)->x() - (itr - 1)->x();
    const auto dy = (itr)->y() - (itr - 1)->y();
    const auto dz = (itr)->z() - (itr - 1)->z();
    const auto norm = std::hypot(dx, dy, dz);
    points.emplace_back(
      static_cast<float>(itr->x()), static_cast<float>(itr->y()), static_cast<float>(itr->z()),
      static_cast<float>(dx / norm), static_cast<float>(dy / norm), static_cast<float>(dz / norm));
  }
  return points;
}

std::shared_ptr<PolylineData> lanelet_to_polyline(
  const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr, size_t max_num_polyline,
  size_t max_num_point, float point_break_distance)
{
  if (!lanelet_map_ptr) {
    return nullptr;
  }

  std::vector<LanePoint> container;
  for (const auto & lanelet : lanelet_map_ptr->laneletLayer) {
    const auto lanelet_subtype = to_subtype_name(lanelet);
    if (is_lane_like(lanelet_subtype)) {  // convert lane
      // Convert centerlines
      if (is_roadway_like(lanelet_subtype)) {
        auto points = from_linestring(lanelet.centerline3d());
        insert_lane_points(points, container);
      }
      // Convert boundaries except of virtual lines
      if (!is_turnable_intersection(lanelet)) {
        const auto left_boundary = lanelet.leftBound3d();
        if (is_boundary_like(left_boundary)) {
          auto points = from_linestring(left_boundary);
          insert_lane_points(points, container);
        }
        const auto right_boundary = lanelet.rightBound3d();
        if (is_boundary_like(right_boundary)) {
          auto points = from_linestring(right_boundary);
          insert_lane_points(points, container);
        }
      }
    } else if (is_crosswalk_like(lanelet_subtype)) {  // convert crosswalk
      auto points = from_polygon(lanelet.polygon3d());
      insert_lane_points(points, container);
    }
  }

  for (const auto & linestring : lanelet_map_ptr->lineStringLayer) {
    if (is_boundary_like(linestring)) {
      auto points = from_linestring(linestring);
      insert_lane_points(points, container);
    }
  }

  return container.size() == 0
           ? nullptr
           : std::make_shared<PolylineData>(
               container, max_num_polyline, max_num_point, point_break_distance);
}
}  // namespace autoware::mtr
