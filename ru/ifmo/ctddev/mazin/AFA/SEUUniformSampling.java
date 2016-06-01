package ru.ifmo.ctddev.mazin.AFA;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
*
* Assumed that all features are nominal (i.e. they can take on values from a
* finite set of values).
*
*/
public class SEUUniformSampling extends AFAMethod {
    private int realPossibleQueries;

    private final int alpha; // param which controls the complexity of the search, random sub-sample of
                             // alpha * b queries is selected from the available pool

    private double[][] C; // todo cost matrix for all instance-feature pairs

    private Pair<J48, Integer>[] attrsClassifiers; // for getProb(..) computing
    private Instances[] attrsInstances; // for getProb(..) computing

    Discretize discretizer;
    Set<Integer> numericAttrsIndexes;

    Instances discInstances;

    public SEUUniformSampling(Instances instances,
                              QueryManager queryManager,
                              int alpha,
                              int b,
                              Discretize discretizer,
                              Set<Integer> numericAttrsIndexes) throws Exception {
        this.instances = new Instances(instances);
        this.queryManager = queryManager;
        this.alpha = alpha;
        this.b = b;
        this.discretizer = discretizer; // todo;
        this.numericAttrsIndexes = numericAttrsIndexes; // todo

        init(alpha);
    }

    private void init(int alpha) throws Exception {
        n = instances.numInstances();
        m = instances.numAttributes() - 1;

        ArrayList<Pair<Integer, Integer>> possibleQueriesAsList = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (instances.instance(i).isMissing(j)) {
                    possibleQueriesAsList.add(new Pair<>(i, j));
                }
            }
        }

        // init possible queries
        realPossibleQueries = possibleQueriesAsList.size();

        if (alpha < 1) {
            throw new IllegalArgumentException("'alpha' should be > 1");
        }
        alpha = Math.min(alpha, possibleQueriesAsList.size() / b);

        Collections.shuffle(possibleQueriesAsList);
        List<Pair<Integer, Integer>> uniformSample = possibleQueriesAsList.subList(0, alpha * b);

        possibleQueries = new HashMap<>(n);
        for (Pair<Integer, Integer> query : uniformSample) {
            if (!possibleQueries.containsKey(query.first)) {
                possibleQueries.put(query.first, new HashSet<>());
            }
            possibleQueries.get(query.first).add(query.second);
        }
    }

    public List<Pair<List<Pair<Integer, Integer>>, J48>> perform(int k) throws Exception {
//      todo  curMissingCells = chooseSubsetOfMissingCells(curMissingCells);
        J48 classifier = makeClassifier();
        List<Pair<List<Pair<Integer, Integer>>, J48>> res = new LinkedList<>();
        while (k-- > 0) {
            if (possibleQueries.size() == 0) {
                break; // nothing to query
            }
            int batchSize = b;
            int possibleQueriesNum = getPossibleQueriesNum();
            if (possibleQueriesNum < b) { // todo
                batchSize = possibleQueriesNum;
            }

            // set discInstances
            discInstances = Filter.useFilter(instances, discretizer);

            List<Pair<Integer, Integer>> bestQueries = concurrentPerformStep(batchSize, classifier);
            classifier = makeClassifier();
            res.add(new Pair<>(bestQueries, classifier));

            updatePossibleQueries();
        }

        return res;
    }

//    private List<Pair<Integer, Integer>> performStep(int batchSize, J48 classifier) throws Exception {
//        attrsClassifiers = new J48[m]; // todo mb optimize updating - update not all
//        attrsInstances = new Instances[m]; // todo mb optimize updating - update not all
//        List<Pair<Double, Pair<Integer, Integer>>> scores = new ArrayList<>();
//        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
//            int i = entry.getKey();
//            for (Integer j : entry.getValue()) {
//                double score = getScore(i, j, classifier);
//                scores.add(new Pair<>(score, new Pair<>(i, j)));
//            }
//        }
//        // Choose best 'b' queries to acquire:
//        Collections.sort(scores, (o1, o2) -> o1.first < o2.first ? 1 : o1.first.equals(o2.first) ? 0 : -1);
//        ArrayList<Pair<Integer, Integer>> bestQueries = scores.subList(0, batchSize).stream()
//                .map(pair -> pair.second)
//                .collect(Collectors.toCollection(ArrayList::new));
//
//        for (Pair<Integer, Integer> query : bestQueries) {
//            acquireQuery(query.first, query.second);
//
//            possibleQueries.get(query.first).remove(query.second);
//        }
//
//        // update discInstances
//        discInstances = Filter.useFilter(instances, discretizer);
//
//        return bestQueries;
//    }

    /**
     * Perform step with calculation score for each possible query in parallel
     *
     * @param batchSize
     * @param cls
     * @return
     * @throws Exception
     */
    private List<Pair<Integer, Integer>> concurrentPerformStep(int batchSize, J48 cls) throws Exception {
        attrsClassifiers = new Pair[m];
        attrsInstances = new Instances[m];
        for (int i = 0; i < m; ++i) {
            initClassifierForAttr(i);
        }

        double oldAcc = DatasetFactory.calculateAccuracy(cls, instances);

        List<Pair<Double, Pair<Integer, Integer>>> scores = Collections.synchronizedList(new ArrayList<>());
        ExecutorService execSvc = Executors.newCachedThreadPool();
        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
            int i = entry.getKey();
            for (Integer j : entry.getValue()) {
                CodeRunner runner = new CodeRunner(new Instances(discInstances), i, j, scores, oldAcc);
                execSvc.execute(runner);
            }
        }
        execSvc.shutdown();
        execSvc.awaitTermination(Long.MAX_VALUE, TimeUnit.MINUTES); // todo

        // Choose best 'b' queries to acquire:
        Collections.sort(scores, (o1, o2) -> o1.first < o2.first ? 1 : o1.first.equals(o2.first) ? 0 : -1);
        ArrayList<Pair<Integer, Integer>> bestQueries = scores.subList(0, batchSize).stream()
                .map(pair -> pair.second)
                .collect(Collectors.toCollection(ArrayList::new));

        for (Pair<Integer, Integer> query : bestQueries) {
            acquireQuery(query.first, query.second);

            possibleQueries.get(query.first).remove(query.second);
        }

        // update discInstances
        discInstances = Filter.useFilter(instances, discretizer);

        return bestQueries;
    }

    /**
     * Remove complete instances from 'possibleQueries'
     */
    private void updatePossibleQueries() {
        Set<Integer> completeInstances = possibleQueries.entrySet().stream()
                .filter(entry -> entry.getValue().isEmpty())
                .map(Map.Entry::getKey).collect(Collectors.toSet());

        completeInstances.forEach(possibleQueries::remove);
    }

    private void acquireQuery(int instIndex, int attrIndex) throws Exception {
        Instance inst = instances.instance(instIndex);
        double value = queryManager.getValue(instIndex, attrIndex);
        inst.setValue(attrIndex, value);
    }

//    public static void acquireQuery(QueryManager queryManager,
//                                    Instances instances,
//                                    int instIndex,
//                                    int attrIndex) throws Exception {
//        Instance inst = instances.instance(instIndex);
//        double value = queryManager.getValue(instIndex, attrIndex);
//
//        inst.setValue(attrIndex, value);
//        // todo
////        if (!numericAttrsIndexes.contains(attrIndex)) {
////            inst.setValue(attrIndex, value);
////        } else {
////            double[] cutPoints = discretizer.getCutPoints(attrIndex);
////            int pos = Arrays.binarySearch(cutPoints, value);
////            pos = pos >= 0 ? pos : -(pos + 1);
////            inst.setValue(attrIndex, pos);
////        }
//    }

//    /**
//     * Get expected utility of possible query.
//     *
//     * @return
//     */
//    private double getScore(int instIndex, int attrIndex, J48 classifier) throws Exception {
//        double score = 0.0;
//        double[] estimatedProbs = getProbs(instIndex, attrIndex);
//
//        int numValues = instances.attribute(attrIndex).numValues();
//        for (int valueIndex = 0; valueIndex < numValues; ++valueIndex) {
//            score += estimatedProbs[valueIndex] *
//                     getUtility(instIndex, attrIndex, valueIndex, classifier);
//        }
//        return score;
//    }

//    /**
//     * The probability that 'query' has the value 'value'. todo
//     *
//     * @param instIndex
//     * @param attrIndex
//     * @return
//     * @throws Exception
//     */
//    private double[] getProbs(int instIndex, int attrIndex) throws Exception {
//        if (attrsClassifiers[attrIndex] == null) {
//            initClassifierForAttr(attrIndex);
//        }
//        Instance inst = new Instance(instances.instance(instIndex));
//        inst.setDataset(attrsInstances[attrIndex]);
//        return attrsClassifiers[attrIndex].distributionForInstance(inst);
//    }


//    /**
//     * The utility of knowing that the 'query' has the value 'value'.
//     * Formula:
//     * U(query = value) = A(Q, query = value) − A(Q) /  C[query],
//     * where A(..) - accuracy.
//     *
//     * @param instIndex
//     * @param attrIndex
//     * @param value
//     * @param classifier
//     * @return
//     * @throws Exception
//     */
//    private double getUtility(int instIndex, int attrIndex, double value, J48 classifier) throws Exception {
//        instances.instance(instIndex).setValue(attrIndex, value);
//        J48 newClassifier = new J48();
//
//
//
//        double newAcc = DatasetFactory.calculateAccuracy(newClassifier, instances);
//        instances.instance(instIndex).setMissing(attrIndex);
//
//        double oldAcc = DatasetFactory.calculateAccuracy(classifier, instances);
//
////        return (newAcc - oldAcc) / C[instIndex][attrIndex];
//        return newAcc - oldAcc;
//    }

    /**
     * todo
     *
     * @return
     * @throws Exception
     */
    public J48 makeClassifier() throws Exception {
        return DatasetFactory.staticMakeClassifier(instances);
    }

    private void initClassifierForAttr(int attrIndex) throws Exception {
        int capacity = n;
        for (int i = 0; i < n; ++i) {
            if (Instance.isMissingValue(discInstances.instance(i).value(attrIndex))) {
                --capacity;
            }
        }

        Instances instancesForAttr = new Instances(discInstances, capacity);
        boolean isUnaryClassifier = true;
        int classValue = -1;
        for (int i = 0; i < n; ++i) {
            // todo !
            if (!Instance.isMissingValue(discInstances.instance(i).value(attrIndex))) {
                int value = (int) discInstances.instance(i).value(attrIndex);
                if (classValue != -1 && classValue != value) {
                    isUnaryClassifier = false;
                } else {
                    classValue = value;
                }
                instancesForAttr.add(discInstances.instance(i));
            }
        }
        instancesForAttr.setClassIndex(attrIndex);
        attrsInstances[attrIndex] = instancesForAttr;
        if (isUnaryClassifier) {
            attrsClassifiers[attrIndex] = new Pair<>(null, classValue);
        } else {
            J48 attrCls = DatasetFactory.staticMakeClassifier(instancesForAttr);
            attrsClassifiers[attrIndex] = new Pair<>(attrCls, -1);
        }
    }

    public int getPossibleQueriesNum() {
        int res = 0;
        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
            res += entry.getValue().size();
        }

        return res;
    }

    class CodeRunner implements Runnable {

        public Instances instances;
        public int instIndex;
        public int attrIndex;
        public List<Pair<Double, Pair<Integer, Integer>>> scores;
        public final double oldAcc;

        CodeRunner(Instances instances, int instIndex, int attrIndex, List<Pair<Double,
                Pair<Integer, Integer>>> scores, double oldAcc) {
            this.instances = instances;
            this.instIndex = instIndex;
            this.attrIndex = attrIndex;
            this.scores = scores;
            this.oldAcc = oldAcc;
        }

        @Override
        public void run() {
            try {
                double score = concurrentGetScore(instances, instIndex, attrIndex, oldAcc, attrsClassifiers, attrsInstances);
                scores.add(new Pair<>(score, new Pair<>(instIndex, attrIndex)));

                // todo what
//                cls.setUseLaplace(true);
//                cls.buildClassifier(instances);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private double concurrentGetScore(Instances instances,
                                          int instIndex,
                                          int attrIndex,
                                          double oldAcc,
                                          Pair<J48, Integer>[] attrsClassifiers,
                                          Instances[] attrsInstances) throws Exception {
            double score = 0.0;
            Pair<J48, Integer> attrClassifier = attrsClassifiers[attrIndex];
            if (attrClassifier.first == null) {
                if (attrClassifier.second == -1) {
                    int x = 0;
                }
                score = concurrentGetUtility(instances, instIndex, attrIndex, (double) attrClassifier.second, oldAcc);
            } else {
                double[] estimatedProbs = concurrentGetProbs(instances, instIndex, attrIndex, attrClassifier.first, attrsInstances);
                int numValues = instances.attribute(attrIndex).numValues();
                for (int valueIndex = 0; valueIndex < numValues; ++valueIndex) {
                    score += estimatedProbs[valueIndex] *
                            concurrentGetUtility(instances, instIndex, attrIndex, valueIndex, oldAcc);
                }
            }
            return score;
        }

        private double[] concurrentGetProbs(Instances instances,
                                            int instIndex,
                                            int attrIndex,
                                            J48 attrClassifier,
                                            Instances[] attrsInstances) throws Exception {
            Instance inst = new Instance(instances.instance(instIndex));
            inst.setDataset(attrsInstances[attrIndex]);
            return attrClassifier.distributionForInstance(inst);
        }

        private double concurrentGetUtility(Instances instances, int instIndex, int attrIndex, double valueIndex, double oldAcc) throws Exception {
            instances.instance(instIndex).setValue(attrIndex, valueIndex);
            J48 newClassifier = DatasetFactory.staticMakeClassifier(instances);
            double newAcc = DatasetFactory.calculateAccuracy(newClassifier, instances);
            instances.instance(instIndex).setMissing(attrIndex);

            return newAcc - oldAcc;
        }

    }

    public int getRealPossibleQueries() {
        return realPossibleQueries;
    }

}























