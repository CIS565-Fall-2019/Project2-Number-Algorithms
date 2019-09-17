/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"

int main(int argc, char* argv[]) {
    float *xorInput1 = new float[3];
    xorInput1[0] = 0;
    xorInput1[1] = 0;
    xorInput1[2] = 1; //bias

    float *xorTarget1 = new float[1];
    xorTarget1[0] = 0;

    float *xorInput2 = new float[3];
    xorInput2[0] = 0;
    xorInput2[1] = 1;
    xorInput2[2] = 1;

    float *xorTarget2 = new float[1];
    xorTarget2[0] = 1;

    float *xorInput3 = new float[3];
    xorInput3[0] = 1;
    xorInput3[1] = 0;
    xorInput3[2] = 1;

    float *xorTarget3 = new float[1];
    xorTarget3[0] = 1;

    float *xorInput4 = new float[3];
    xorInput4[0] = 1;
    xorInput4[1] = 1;
    xorInput4[2] = 1;

    float *xorTarget4 = new float[1];
    xorTarget4[0] = 0;

    float *wkj = new float[9];
    float *wji = new float[3];
    CharacterRecognition::makeWeightMat(9, wkj);
    CharacterRecognition::makeWeightMat(3, wji);
    // testing values from spreadsheet, 
    // make sure to change j and k to 2 to get rid of bias
    /*wkj[0] = 10.1;
    wkj[1] = 0.9;
    wkj[2] = 20;
    wkj[3] = 0.87;
    wji[0] = 41;
    wji[1] = -54;*/


    float *xorOutput = new float[1];

    int i = 1;
    int j = 3;
    int k = 3;
    
    //training
    float tgtError = 0.01f;
    float currError = 100000.0f;
    int count = 0;
    while (currError > tgtError && count < 15000)
    {
        currError = 0;
        currError += CharacterRecognition::mlpTrain(i, j, k, xorOutput, xorInput1, wkj, wji, xorTarget1);
        currError += CharacterRecognition::mlpTrain(i, j, k, xorOutput, xorInput2, wkj, wji, xorTarget2);
        currError += CharacterRecognition::mlpTrain(i, j, k, xorOutput, xorInput3, wkj, wji, xorTarget3);
        currError += CharacterRecognition::mlpTrain(i, j, k, xorOutput, xorInput4, wkj, wji, xorTarget4);
        count++;
    }

    //test
    printf("Ran %d iterations of training\n", count);
    CharacterRecognition::mlpRun(i, j, k, xorOutput, xorInput1, wkj, wji);
    printf("    (0, 0) expected: %f, result %f\n", xorTarget1[0], xorOutput[0]);
    CharacterRecognition::mlpRun(i, j, k, xorOutput, xorInput2, wkj, wji);
    printf("    (0, 1) expected: %f, result %f\n", xorTarget2[0], xorOutput[0]);
    CharacterRecognition::mlpRun(i, j, k, xorOutput, xorInput3, wkj, wji);
    printf("    (1, 0) expected: %f, result %f\n", xorTarget3[0], xorOutput[0]);
    CharacterRecognition::mlpRun(i, j, k, xorOutput, xorInput4, wkj, wji);
    printf("    (1, 1) expected: %f, result %f\n", xorTarget4[0], xorOutput[0]);

    system("pause"); // stop Win32 console from closing on exit
    
}
