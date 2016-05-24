package ru.ifmo.ctddev.mazin.AFA;

import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.util.List;
import java.util.Map;
import java.util.Set;

abstract class AFAMethod {
    protected Instances instances;
    protected int n; // number of instances
    protected int m; // number of attributes (features)
    protected int b; // batch size to query at one step. If there are fewer than 'b'
                   // possible queries, acquire all remaining possible queries.
    protected Map<Integer, Set<Integer>> possibleQueries; // list of attributes of instance with missing values
    protected QueryManager queryManager;

    public J48 makeClassifier() throws Exception {
        J48 classifier = new J48();
        classifier.setUseLaplace(true);
        classifier.buildClassifier(instances);
        return classifier;
    }

    public int getAllQueriesNum() {
        return n * m;
    }

    abstract List<Pair<List<Pair<Integer, Integer>>, J48>> perform(int k) throws Exception;
    abstract int getRealPossibleQueries();
}
