// This program converts detection bounding box labels to Dataum proto buffers
// and save them in LMDB.
// Usage:
//   convert_annotation_data [FLAGS] id_list_file annotation_list_file img_list_file type_list_file DB_NAME
//

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>
#include <cstdio>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#undef NDEBUG
#include <cassert>
#include <jsoncons/json.hpp>

using jsoncons::json;

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;
using google::protobuf::Message;

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    if (s.size()>0 && s[s.size()-1] == delim) elems.push_back("");
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

int find_type(std::vector<string> &types, string tp) {
    int id=-1;
    for (int i=0; i<types.size(); i++)
        if (tp == types[i]) {
            id = i;
            break;
        }
    if (id == -1) {
        LOG(INFO) << "new type " << tp << " " << types.size();
        id = types.size();
        types.push_back(tp);
    }
    return id;
}

DEFINE_bool(test_run, false, "If set to true, only generate 100 images.");
DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, true,
    "Randomly shuffle the order of images and their labels");
DEFINE_bool(use_rgb, false, "use RGB channels");
DEFINE_int32(resize_width, 640 + 32, "Width images are resized to");
DEFINE_int32(resize_height, 480 + 32, "Height images are resized to");

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_annotation_data [FLAGS] annotation_list_file id_list_file DB_NAME\n");
  /*
   * type_list_file: map type to int
   * img_list_file: a set of image path
   * annotation_list_file: annotation_json_file
   * id_list_file: ids in dataset
   */
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_annotation_data2");
    return 1;
  }

  bool is_color = !FLAGS_gray;
  string root_path = argv[1];
  int pos = root_path.find_last_of("/\\");
  std::cout << "Root path : " << root_path <<  pos << std::endl;
  root_path = root_path.substr(0, pos+1);
  std::cout << "Root path : " << root_path <<  pos << std::endl;

  std::ifstream fin1(argv[1]);
  string s;
  std::getline(fin1,s);
  json annotation = json::parse(s);
  std::cout << annotation.to_string().size() << std::endl;
  std::cout << s.size() << std::endl;
  fin1.close();

// load img list
  std::map<string,string> id2file;
  std::cout << annotation.to_string().substr(0,100) << std::endl;
  std::cout << annotation.has_member("imgs") << std::endl;
  for (const auto &member : annotation.members()) {
      std::cout << member.name() << std::endl;
  }
  for (const auto &member : annotation["imgs"].members()) {
      string path = root_path + member.value()["path"].as_string();

      id2file[member.name()] = path;
  }
  std::cout << id2file.size() << std::endl;
// load types
  std::vector<string> types;
  for (const auto &e : annotation["types"].elements()) {
      types.push_back(e.as<string>());
  }
  std::cout << types.size() << std::endl;

  std::ifstream idfile(argv[2]);
  std::vector<string> ids;
  string filename;
  while (idfile >> filename) {
      ids.push_back(filename);
  }
  idfile.close();

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(ids.begin(), ids.end());
  }
  LOG(INFO) << "A total of " << ids.size() << " images.";

  std::map< string, caffe::DrivingData > id2data;
  for (const auto &member : annotation["imgs"].members()) {
      string id = member.name();
      caffe::DrivingData data;
      data.Clear();
      for (const auto &b : member.value()["objects"].elements()) {
          caffe::CarBoundingBox box;
          caffe::FixedPoint *fp;
          box.Clear();
          box.set_type(find_type(types, b["category"].as_string()));
          box.set_xmin(b["bbox"]["xmin"].as<float>());
          box.set_xmax(b["bbox"]["xmax"].as<float>());
          box.set_ymin(b["bbox"]["ymin"].as<float>());
          box.set_ymax(b["bbox"]["ymax"].as<float>());
          if (b.has_member("ellipse_org")) {
              for (const auto &p : b["ellipse_org"].elements()) {
                  fp = box.add_ellipse_mask();
                  fp->set_x(p[0].as<float>());
                  fp->set_y(p[1].as<float>());
              }
          }
          if (b.has_member("polygon")) {
              for (const auto &p : b["polygon"].elements()) {
                  fp = box.add_poly_mask();
                  fp->set_x(p[0].as<float>());
                  fp->set_y(p[1].as<float>());
              }
          }
          data.add_car_boxes()->CopyFrom(box);
      }
      id2data[id] = data;
  }

  const char* db_path = argv[3];

  bool generate_img = true;
  std::string db_str(db_path);
  if (db_str == "none") {
    generate_img = false;
  }

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Open new db
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  // Open db
  LOG(INFO) << "Opening lmdb " << db_path;
  CHECK_EQ(mkdir(db_path, 0744), 0)
      << "mkdir " << db_path << "failed";
  CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
      << "mdb_env_set_mapsize failed";
  CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
      << "mdb_env_open failed";
  CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
      << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
      << "mdb_open failed. Does the lmdb already exist? ";

  const int kMaxKeyLength = 256;
  char key_cstr[kMaxKeyLength];
  int count = 0;
  LOG(ERROR) << "Total to be processed: " << ids.size() << ".\n";

  for (int i = 0; i < ids.size(); i++) {
    DrivingData data;
    string id = ids[i];
    assert(id2file.count(id));
    assert(id2data.count(id));
    data.CopyFrom(id2data[id]);
    if (data.car_boxes_size()==0) {
        LOG(ERROR) << "error, no box inside " << id;
        continue;
    }
  }
  for (int i = 0; i < ids.size(); i++) {
    DrivingData data;
    string id = ids[i];
    assert(id2file.count(id));
    assert(id2data.count(id));
    data.CopyFrom(id2data[id]);
    if (data.car_boxes_size()==0) {
        continue;
    }
    const string image_path = id2file[id];
    data.set_car_img_source(image_path);

    if (!ReadImageToDatum(image_path, 0,
        resize_height, resize_width, is_color, data.mutable_car_image_datum())) {
      LOG(INFO) << "read failed " << image_path;
      continue;
    }

    // sequential
    snprintf(key_cstr, kMaxKeyLength, "%08d_%s", i,
        id.c_str());
    string value;
    data.SerializeToString(&value);
    string keystr(key_cstr);

    // Put in db
    mdb_data.mv_size = value.size();
    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
    mdb_key.mv_size = keystr.size();
    mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
        << "mdb_put failed";

    if (++count % 1000 == 0) {
      // Commit txn
      CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
          << "mdb_txn_commit failed";
      CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
          << "mdb_txn_begin failed";
      LOG(ERROR) << "Processed " << count << " files.";
    } else
    if (count % 10 == 0) {
        LOG(ERROR) << "Processed " << count << " files.";
    }

    if (FLAGS_test_run && count == 10) {
      break;
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
    mdb_close(mdb_env, mdb_dbi);
    mdb_env_close(mdb_env);
    LOG(ERROR) << "Processed " << count << " files.";
  }
  return 0;
}
