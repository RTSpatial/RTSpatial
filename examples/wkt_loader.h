#ifndef SPATIALQUERYBENCHMARK_LOADER_H
#define SPATIALQUERYBENCHMARK_LOADER_H
#include <dirent.h>
#include <sys/stat.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/register/box.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <vector>

#include <boost/geometry.hpp>
using coord_t = float;
using point_t = boost::geometry::model::d2::point_xy<coord_t>;
using box_t = boost::geometry::model::box<point_t>;
using polygon_t = boost::geometry::model::polygon<point_t>;

std::vector<box_t> LoadBoxes(const std::string &path,
                             int limit = std::numeric_limits<int>::max()) {
  std::ifstream ifs(path);
  std::string line;
  std::vector<box_t> boxes;

  while (std::getline(ifs, line)) {
    if (!line.empty()) {
      std::vector<point_t> points;

      if (line.rfind("MULTIPOLYGON", 0) == 0) {
        boost::geometry::model::multi_polygon<polygon_t> multi_poly;
        boost::geometry::read_wkt(line, multi_poly);

        for (auto &poly : multi_poly) {
          for (auto &p : poly.outer()) {
            points.push_back(p);
          }
        }
      } else if (line.rfind("POLYGON", 0) == 0) {
        polygon_t poly;
        boost::geometry::read_wkt(line, poly);

        for (auto &p : poly.outer()) {
          points.push_back(p);
        }
      } else {
        std::cerr << "Bad Geometry " << line << "\n";
        abort();
      }

      coord_t lows[2] = {std::numeric_limits<coord_t>::max(),
                         std::numeric_limits<coord_t>::max()};
      coord_t highs[2] = {std::numeric_limits<coord_t>::lowest(),
                          std::numeric_limits<coord_t>::lowest()};

      for (auto &p : points) {
        lows[0] = std::min(lows[0], p.x());
        highs[0] = std::max(highs[0], p.x());
        lows[1] = std::min(lows[1], p.y());
        highs[1] = std::max(highs[1], p.y());
      }

      box_t box(point_t(lows[0], lows[1]), point_t(highs[0], highs[1]));

      boxes.push_back(box);
      //      if (boxes.size() % 1000 == 0) {
      //        std::cout << "Loaded geometries " << boxes.size() / 1000 << " K"
      //                  << std::endl;
      //      }
      if (boxes.size() >= limit) {
        break;
      }
    }
  }
  ifs.close();
  return boxes;
}

std::vector<point_t> LoadPoints(const std::string &path,
                                int limit = std::numeric_limits<int>::max()) {
  std::ifstream ifs(path);
  std::string line;
  std::vector<point_t> points;

  while (std::getline(ifs, line)) {
    if (!line.empty()) {
      if (line.rfind("MULTIPOLYGON", 0) == 0) {
        boost::geometry::model::multi_polygon<polygon_t> multi_poly;
        boost::geometry::read_wkt(line, multi_poly);

        for (auto &poly : multi_poly) {
          for (auto &p : poly.outer()) {
            points.push_back(p);
          }
        }
      } else if (line.rfind("POLYGON", 0) == 0) {
        polygon_t poly;
        boost::geometry::read_wkt(line, poly);

        for (auto &p : poly.outer()) {
          points.push_back(p);
        }
      } else if (line.rfind("POINT", 0) == 0) {
        point_t p;
        boost::geometry::read_wkt(line, p);

        points.push_back(p);
      } else {
        std::cerr << "Bad Geometry " << line << "\n";
        abort();
      }
      //      if (points.size() % 1000 == 0) {
      //        std::cout << "Loaded geometries " << points.size() / 1000 <<
      //        std::endl;
      //      }
      if (points.size() >= limit) {
        break;
      }
    }
  }
  ifs.close();
  return points;
}

void SerializeBoxes(const char *file, const std::vector<box_t> &boxes) {
  std::ofstream ofs(file, std::ios::binary);
  boost::archive::binary_oarchive oa(ofs);
  oa << boost::serialization::make_nvp("boxes", boxes);
  ofs.close();
}

std::vector<box_t> DeserializeBoxes(const char *file) {
  std::vector<box_t> deserialized_boxes;
  std::ifstream ifs(file, std::ios::binary);
  boost::archive::binary_iarchive ia(ifs);
  ia >> boost::serialization::make_nvp("boxes", deserialized_boxes);
  ifs.close();
  return deserialized_boxes;
}

std::vector<box_t> LoadBoxes(const std::string &path,
                             const std::string &serialize_prefix,
                             int limit = std::numeric_limits<int>::max()) {
  std::string escaped_path;
  std::replace_copy(path.begin(), path.end(), std::back_inserter(escaped_path),
                    '/', '-');

  if (!serialize_prefix.empty()) {
    DIR *dir = opendir(serialize_prefix.c_str());
    if (dir) {
      closedir(dir);
    } else if (ENOENT == errno) {
      if (mkdir(serialize_prefix.c_str(), 0755)) {
        std::cerr << "Cannot create dir " << path;
        abort();
      }
    } else {
      std::cerr << "Cannot open dir " << path;
      abort();
    }
  }
  auto ser_path = serialize_prefix + '/' + escaped_path + "_limit_" +
                  std::to_string(limit) + ".bin";

  std::vector<box_t> boxes;

  if (access(ser_path.c_str(), R_OK) == 0) {
    boxes = DeserializeBoxes(ser_path.c_str());
  } else {
    boxes = LoadBoxes(path, limit);
    if (!serialize_prefix.empty()) {
      SerializeBoxes(ser_path.c_str(), boxes);
    }
  }
  return boxes;
}


inline void CopyBoxes(
    const std::vector<box_t> &boxes,
    thrust::device_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>>
        &d_boxes) {
  pinned_vector<rtspatial::Envelope<rtspatial::Point<coord_t, 2>>> h_boxes;

  h_boxes.resize(boxes.size());

  for (size_t i = 0; i < boxes.size(); i++) {
    rtspatial::Point<coord_t, 2> p_min(boxes[i].min_corner().x(),
                                       boxes[i].min_corner().y());
    rtspatial::Point<coord_t, 2> p_max(boxes[i].max_corner().x(),
                                       boxes[i].max_corner().y());

    h_boxes[i] =
        rtspatial::Envelope<rtspatial::Point<coord_t, 2>>(p_min, p_max);
  }

  d_boxes = h_boxes;
}

inline void
CopyPoints(const std::vector<point_t> &points,
           thrust::device_vector<rtspatial::Point<coord_t, 2>> &d_points) {
  pinned_vector<rtspatial::Point<coord_t, 2>> h_points;

  h_points.resize(points.size());

  for (size_t i = 0; i < points.size(); i++) {
    h_points[i] = rtspatial::Point<coord_t, 2>(points[i].x(), points[i].y());
  }

  d_points = h_points;
}

namespace boost {
namespace serialization {
template <class Archive>
void serialize(Archive &ar, box_t &box, const unsigned int version) {
  ar &boost::serialization::make_nvp("min_corner", box.min_corner());
  ar &boost::serialization::make_nvp("max_corner", box.max_corner());
}

template <class Archive>
void serialize(Archive &ar, point_t &p, const unsigned int version) {
  ar &const_cast<coord_t &>(p.x());
  ar &const_cast<coord_t &>(p.y());
}
} // namespace serialization
} // namespace boost

#endif // SPATIALQUERYBENCHMARK_LOADER_H
