/*
 * 3 Inputs 2 Outputs Perceptron Learning Algorithm
 * Author: Marcos Moura
 * Contact: gabbyru2@gmail.com
 */

#include <iostream>

#include <vector>
#include <tuple>
#include <string>

using namespace std;

#define LEARNING_RATE 0.001
#define ITERATION_LIMIT 100000
#define ERROR_TOLERENCE 0.0001
#define EPOCH 1000

//#define COLLECT_STAT

#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

//typedef std::vector<std::tuple<double, double, int> > DataSet;

typedef std::vector<std::tuple<double, double, double, std::vector<int>> > DataSet;

class Perceptron {

  private:
    DataSet allTds; // all training instances
    DataSet tds;
    DataSet testSet; // instances used for testing the Perceptron
    std::vector<std::vector<double>> weights; // weights associated with the inputs and the bias

    Perceptron(std::string& trainFileName);
    void operator=(Perceptron&);
    Perceptron(const Perceptron&);

  public:
    static Perceptron& getPerceptron(std::string& trainFileName) {
      static Perceptron pt(trainFileName);
      return pt;
    }

    bool isCorrectlyPredicted(std::tuple<double, double, double, std::vector<int>>& input, bool printPredictedOutput=false);

    int getSizeOfTrainingDataSet();

    void populateTrainingDataSet(std::string& trainingFileName);
    void populateTrainingDataSet();
    void trainPerceptron();
    void initializeWeights();
    void updateWeights(
        int weightIndex, 
        double input, 
        std::vector<int> actualOutput, 
        std::vector<double> predictedOutput);
    void perform10FoldXValidation();
    void crossSplitDataSet(int counter);
    void resetDB();

    std::vector<double> activationFunc(std::vector<double> input);
    double calculateAvgSquaredError(std::vector<std::vector<double>>& predictedOutput);
    double reportAccuracy(int counter);

    double get_wall_time();
    double get_cpu_time();
};

#endif // PERCEPTRON_HPP
