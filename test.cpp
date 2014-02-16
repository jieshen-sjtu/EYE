/*
 * test.cpp
 *
 *  Created on: Feb 10, 2014
 *      Author: jieshen
 */

#include "EYE.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <opencv2/opencv.hpp>
using namespace std;
using boost::shared_ptr;

namespace EYE
{
  void test_codebook(int argc, char* argv[])
  {
    VlRand rand;

    uint32_t numData = 5000;
    uint32_t dimension = 128;
    uint32_t numCenters = 2000;

    uint32_t dataIdx, d;

    vl_rand_init(&rand);
    vl_rand_seed(&rand, 1000);

    cerr << "Start Generating data" << endl;

    float* _data = (float*) malloc(sizeof(float) * dimension * numData);
    shared_ptr<float> data(_data);

    for (dataIdx = 0; dataIdx < numData; dataIdx++)
    {
      for (d = 0; d < dimension; d++)
      {
        float randomNum = (float) vl_rand_real3(&rand) + 1;
        _data[dataIdx * dimension + d] = randomNum;
      }
    }

    EYE::CodeBook codebook;

    cerr << "Start clustering" << endl;

    codebook.GenKMeans(data, numData, dimension, numCenters);
    cerr << "Done" << endl;

    FILE* output = fopen("data/eye_codebook.txt", "w");
    float* _cluster = (float*) malloc(sizeof(float) * dimension * numCenters);
    memcpy(_cluster, codebook.get_clusters(),
           sizeof(float) * dimension * numCenters);
    shared_ptr<float> cluster(_cluster);
    codebook.save(output, cluster, dimension, numCenters);
    cerr << "save to eye_codebook.txt" << endl;

  }

  void test_dsift(int argc, char* argv[])
  {
    if (argc < 2)
    {
      cerr << "ERROR: must pass an image" << endl;
      exit(-1);
    }
    cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    img.convertTo(img, CV_32FC1);

    /*
    float* data_ = (float*) malloc(sizeof(float) * img.rows * img.cols);
    memcpy(data_, img.data, sizeof(float) * img.rows * img.cols);
    shared_ptr<float> data(data_);
    */

    cerr << "data copy done" << endl;

    cerr << img.cols << " " << img.rows << endl;

    DSift dsift_model;
    vector<VlDsiftKeypoint> frames;
    vector<float> descrs;
    uint32_t dim(0);

    const uint32_t width = img.cols;
    const uint32_t height = img.rows;

    ofstream output("data/eye_imgdata.txt");
    for (int i = 0; i < width * height; ++i)
    {
      output << img.data[i] << " ";
      if ((i + 1) % width == 0)
        output << endl;
    }

    dsift_model.Extract((float*)img.data, width, height, &frames, &descrs, &dim);
    cerr << frames.size() << endl;

    output.close();
    output.open("data/eye_dsiftfeature.txt");
    for (int i = 0; i < frames.size() * dim; ++i)
    {
      output << descrs[i] << " ";
      if ((i + 1) % dim == 0)
        output << endl;
    }
    output.close();

    output.open("data/eye_dsiftframe.txt");
    for (int i = 0; i < frames.size(); ++i)
      output << frames[i].x << " " << frames[i].y << endl;
    output.close();
  }

  void test_llc(int argc, char* argv[])
  {
    // load the codebook
    const uint32_t num_center = 4000;
    const uint32_t dim = 128;
    const uint32_t len_cb = num_center * dim;
    float* codebook = (float*) malloc(sizeof(float) * len_cb);
    memset(codebook, 0, sizeof(float) * len_cb);

    const string bookfile("data/eye_llccodebook.txt");
    const string ver_book("data/ver_llccodebook.txt");
    ifstream input(bookfile.c_str());
    ofstream output(ver_book.c_str());
    for (uint32_t i = 0; i < len_cb; ++i)
    {
      input >> codebook[i];
      output << codebook[i] << " ";
      if ((i + 1) % dim == 0)
        output << endl;
    }
    input.close();
    output.close();

    cerr << "loaded codebook" << endl;

    // load the testing feature
    const uint32_t num_samples = 16116;
    const uint32_t len_sp = num_samples * dim;
    float* features = (float*) malloc(sizeof(float) * len_sp);
    memset(features, 0, sizeof(float) * len_sp);

    const string featfile("data/eye_llcfeature.txt");
    const string ver_feat("data/ver_llcfeature.txt");
    input.open(featfile.c_str());
    output.open(ver_feat.c_str());
    for (uint32_t i = 0; i < len_sp; ++i)
    {
      input >> features[i];
      output << features[i] << " ";
      if ((i + 1) % dim == 0)
        output << endl;
    }
    input.close();
    output.close();
    cerr << "loaded samples" << endl;

    // coding
    shared_ptr<float> base(codebook);
    //shared_ptr<float> X(features);
    shared_ptr<float> code;

    EYE::LLC llc_model;
    llc_model.set_base(base, dim, num_center);
    llc_model.SetUp();

    cerr << "init model done" << endl;

    llc_model.Encode_with_max_pooling(features, dim, num_samples, code);

    cerr << "encode done" << endl;

    const float* pcode = code.get();

    output.open("data/eye_llccode.txt");
    for (uint32_t i = 0; i < num_center; ++i)
      output << pcode[i] << "\n";
    output.close();
  }
}
