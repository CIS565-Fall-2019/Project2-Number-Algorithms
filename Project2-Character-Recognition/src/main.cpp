/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <fstream>
#include <sstream>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"

void readFromFile(int idx, float* inputArr)
{
    std::string fileName = std::to_string(idx) + "info.txt";
    if (idx < 10) { fileName = std::to_string(0) + fileName; }
    fileName = "../data-set/" + fileName;
    
    std::ifstream infile(fileName);

    int n1, n2, count;
    float x;
    count = 0;

    if (!(infile >> n1 >> n2)) { printf("Error reading first two lines of file %s\n", fileName); }

    while (infile >> x && count < 10201)
    {
        inputArr[count] = x;
        count++;
    }
}

int main(int argc, char* argv[]) {
    // XOR TESTING
    printf("XOR TESTING\n");

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


    // CHAR RECOG TESTING
    printf("CHAR RECOG TESTING\n");

    i = 52; 
    j = 10202; 
    k = 10202; // +1 for bias

    float *CRwkj = new float[k*j];
    float *CRwji = new float[j*i];
    CharacterRecognition::makeWeightMat(k*j, wkj);
    CharacterRecognition::makeWeightMat(j*i, wji);

    float *CRoutput = new float[i];

    tgtError = 0.01f;
    currError = 100000.0f;
    count = 0;
    while (currError > tgtError && count < 10)
    {
        currError = 0;
        for (int f = 0; f < 52; f++)
        {
            float* tgt = new float[i];
            zeroArray(i, tgt);
            tgt[f] = 1;

            float* input = new float[k];
            readFromFile(f + 1, input);
            input[k-1] = 1;

            currError += CharacterRecognition::mlpTrain(i, j, k, CRoutput, input, CRwkj, CRwji, tgt);

            delete[] input;
            delete[] tgt;
        }
        printf("After %d iterations, error = %f\n", count, currError);
        count++;
    }

    system("pause"); // stop Win32 console from closing on exit
    delete[] xorInput1;
    delete[] xorTarget1;
    delete[] xorInput2;
    delete[] xorTarget2;
    delete[] xorInput3;
    delete[] xorTarget3;
    delete[] xorInput4;
    delete[] xorTarget4;
    delete[] xorOutput;
    delete[] wkj;
    delete[] wji;

    delete[] CRoutput;
    delete[] CRwkj;
    delete[] CRwji;
}
