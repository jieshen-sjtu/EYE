/*
 * main.cpp
 *
 *  Created on: Feb 10, 2014
 *      Author: jieshen
 */

#include "test.hpp"
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{

  cerr << "Start Testing" << endl;
  cerr << "1. codebook" << endl << "2. dsift" << endl << "3. LLC" << endl;

  int sel(0);
  cin >> sel;

  switch (sel)
  {
    case 1:
      EYE::test_codebook(argc, argv);
      break;
    case 2:
      EYE::test_dsift(argc, argv);
      break;
    case 3:
      EYE::test_llc(argc, argv);
      break;
    default:
      break;
  }

  return 0;
}
