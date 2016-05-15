package ru.ifmo.ctddev.mazin.AFA;

import weka.classifiers.trees.J48;

import java.util.List;

interface AFAMethod {
    List<Pair<List<Pair<Integer, Integer>>, J48>> perform(int k) throws Exception;
    J48 makeClassifier() throws Exception;
    int getAllQueriesNum();
    int getRealPossibleQueries();
}
