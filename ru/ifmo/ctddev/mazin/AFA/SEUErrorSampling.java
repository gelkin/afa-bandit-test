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
public class SEUErrorSampling extends AFAMethod {
    private int realPossibleQueries;

    private int esParam; // param which controls the complexity of the search:
                         // at each step we choose 'esParam' instances, and among them
                         // choose 'b' missing queries to acquire

    private Pair<J48, Integer>[] attrsClassifiers; // for getProb(..) computing
    private Instances[] attrsInstances; // for getProb(..) computing

    Discretize discretizer;
    Set<Integer> numericAttrsIndexes;

    Instances discInstances;

    public SEUErrorSampling(Instances instances,
                              QueryManager queryManager,
                              int esParam,
                              int b,
                              Discretize discretizer,
                              Set<Integer> numericAttrsIndexes) throws Exception {
        this.instances = new Instances(instances);
        this.queryManager = queryManager;
        this.esParam = esParam;
        this.b = b;
        this.discretizer = discretizer; // todo;
        this.numericAttrsIndexes = numericAttrsIndexes; // todo

        init();
    }

    private void init() throws Exception {
        if (esParam < 1 || esParam > instances.numInstances()) {
            throw new IllegalArgumentException("'esParam' should be > 0");
        }

        esParam = Math.min(esParam, instances.numInstances());

        n = instances.numInstances();
        m = instances.numAttributes() - 1;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (instances.instance(i).isMissing(j)) {
                    ++realPossibleQueries;
                }
            }
        }

//        todo discInstances = Filter.useFilter(instances, discretizer);
    }

    public List<Pair<List<Pair<Integer, Integer>>, J48>> perform(int k) throws Exception {
        J48 classifier = makeClassifier();
        List<Pair<List<Pair<Integer, Integer>>, J48>> res = new LinkedList<>();
        while (k-- > 0) {
            possibleQueries = getPossibleQueries(classifier);

            if (possibleQueries.size() == 0) {
                break; // nothing to query
            }

            // set discInstances
            discInstances = Filter.useFilter(instances, discretizer);

            int batchSize = b;
            int possibleQueriesNum = getPossibleQueriesNum();
            if (possibleQueriesNum < b) { // todo
                batchSize = possibleQueriesNum;
            }

            List<Pair<Integer, Integer>> bestQueries = concurrentPerformStep(batchSize, classifier);
            classifier = makeClassifier();
            res.add(new Pair<>(bestQueries, classifier));
        }

        return res;
    }

// todo
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
//        }
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
        for (int j = 0; j < m; ++j) {
            initClassifierForAttr(j);
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
        boolean finshed = execSvc.awaitTermination(Long.MAX_VALUE, TimeUnit.MINUTES); // todo

        // Choose best 'b' queries to acquire:
        Collections.sort(scores, (o1, o2) -> o1.first < o2.first ? 1 : o1.first.equals(o2.first) ? 0 : -1);
        ArrayList<Pair<Integer, Integer>> bestQueries = scores.subList(0, batchSize).stream()
                .map(pair -> pair.second)
                .collect(Collectors.toCollection(ArrayList::new));

        for (Pair<Integer, Integer> query : bestQueries) {
            acquireQuery(query.first, query.second);
        }

        // update discInstances
//        discInstances = Filter.useFilter(instances, discretizer);

        return bestQueries;
    }

    // ok
    public Map<Integer, Set<Integer>> getPossibleQueries(J48 classifier) throws Exception {
        List<Integer> matchedInstances = getInterestingInstances(classifier);

        Map<Integer, Set<Integer>> possibleQueries = new HashMap<>();
        for (int instIndex : matchedInstances) {
            Set<Integer> possibleAttrsToQuery = new HashSet<>();
            for (int attrIndex = 0; attrIndex < m; ++attrIndex) {
                if (Instance.isMissingValue(instances.instance(instIndex).value(attrIndex))) {
                    possibleAttrsToQuery.add(attrIndex);
                }
            }
            if (!possibleAttrsToQuery.isEmpty()) {
                possibleQueries.put(instIndex, possibleAttrsToQuery);
            }
        }

        return possibleQueries;
    }

    // ok
    private List<Integer> getInterestingInstances(J48 cls) throws Exception {
        ArrayList<Integer> misclassified = new ArrayList<>(n);
        for (int i = 0; i < n; ++i) {
            Instance inst = instances.instance(i);
            if (inst.hasMissingValue() && DatasetFactory.isInstMisclassified(inst, cls)) { // todo
                misclassified.add(i);
            }
        }

        List<Integer> matchedInstances = new ArrayList<>(esParam);
        if (misclassified.size() >= esParam) {
            // choose random 'esParam' misclassified instances
            Collections.shuffle(misclassified);
            matchedInstances.addAll(misclassified.subList(0, esParam));
        } else {
            matchedInstances.addAll(misclassified);

            ArrayList<Pair<Double, Integer>> scores = new ArrayList<>(n);
            for (int i = 0; i < n; ++i) {
                Instance inst = instances.instance(i);
                if (inst.hasMissingValue() && !DatasetFactory.isInstMisclassified(inst, cls)) { // todo
                    double score = getUncertaintyScore(inst, cls);
                    scores.add(new Pair<>(score, i));
                }
            }
            // choose min scores
            Collections.sort(scores, (o1, o2) -> o1.first > o2.first ? 1 : o1.first.equals(o2.first) ? 0 : -1);
            int subListSize = Math.min(scores.size(), esParam - misclassified.size());
            ArrayList<Integer> bestInstancesByScore
                    = scores.subList(0, subListSize).stream()
                    .map(pair -> pair.second)
                    .collect(Collectors.toCollection(ArrayList::new));
            matchedInstances.addAll(bestInstancesByScore);
        }

        return matchedInstances;
    }

    // ok
    private double getUncertaintyScore(Instance inst, J48 classifier) throws Exception {
        double[] probs = classifier.distributionForInstance(inst);

        // find class with maximum probability
        int maxIndex = -1;
        double maxProb = Double.MIN_VALUE;
        for (int j = 0; j < probs.length; ++j) {
            if (maxProb < probs[j]) {
                maxIndex = j;
                maxProb = probs[j];
            }
        }

        double secondMaxProb = Double.MIN_VALUE;
        for (int j = 0; j < probs.length; ++j) {
            if (j != maxIndex && secondMaxProb < probs[j]) {
                secondMaxProb = probs[j];
            }
        }

        return maxProb - secondMaxProb;
    }

    private void acquireQuery(int instIndex, int attrIndex) throws Exception {
        Instance inst = instances.instance(instIndex);
        double value = queryManager.getValue(instIndex, attrIndex);
        inst.setValue(attrIndex, value);
    }

    public static void acquireQuery(QueryManager queryManager,
                                    Instances instances,
                                    int instIndex,
                                    int attrIndex) throws Exception {
        Instance inst = instances.instance(instIndex);
        double value = queryManager.getValue(instIndex, attrIndex);

        inst.setValue(attrIndex, value);
        // todo
//        if (!numericAttrsIndexes.contains(attrIndex)) {
//            inst.setValue(attrIndex, value);
//        } else {
//            double[] cutPoints = discretizer.getCutPoints(attrIndex);
//            int pos = Arrays.binarySearch(cutPoints, value);
//            pos = pos >= 0 ? pos : -(pos + 1);
//            inst.setValue(attrIndex, pos);
//        }
    }

    /**
     * Get expected utility of possible query.
     *
     * @return
     */
//    private double getScore(int instIndex, int attrIndex, J48 classifier) throws Exception {
//        double score = 0.0;
//        double[] estimatedProbs = getProbs(instIndex, attrIndex);
//
//        int numValues = discInstances.attribute(attrIndex).numValues();
//        for (int valueIndex = 0; valueIndex < numValues; ++valueIndex) {
//            score += estimatedProbs[valueIndex] *
//                    getUtility(instIndex, attrIndex, valueIndex, classifier);
//        }
//        return score;
//    }

    /**
     * The probability that 'query' has the value 'value'. todo
     *
     * @param instIndex
     * @param attrIndex
     * @return
     * @throws Exception
     */
//    private double[] getProbs(int instIndex, int attrIndex) throws Exception {
//        if (attrsClassifiers[attrIndex] == null) {
//            initClassifierForAttr(attrIndex);
//        }
//        Instance inst = new Instance(discInstances.instance(instIndex));
//        inst.setDataset(attrsInstances[attrIndex]);
//        return attrsClassifiers[attrIndex].distributionForInstance(inst);
//    }


    /**
     * The utility of knowing that the 'query' has the value 'value'.
     * Formula:
     * U(query = value) = A(Q, query = value) − A(Q) /  C[query],
     * where A(..) - accuracy.
     *
     * @param instIndex
     * @param attrIndex
     * @param value
     * @param classifier
     * @return
     * @throws Exception
     */
//    private double getUtility(int instIndex, int attrIndex, double value, J48 classifier) throws Exception {
//        discInstances.instance(instIndex).setValue(attrIndex, value);
//        J48 newClassifier = makeClassifier();
//        double newAcc = DatasetFactory.calculateAccuracy(newClassifier, discInstances);
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

    /**
     * Build a classifier based on instances with class-attribute and attribute
     * at 'attrIndex' position swapped.
     * It is needed to predict the value at 'attrIndex' position for an instance
     * with missing value for this attribute.
     *
     * @param attrIndex
     * @throws Exception
     */
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

    public static int getPossibleQueriesNum(Map<Integer, Set<Integer>> possibleQueries) {
        int res = 0;
        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
            res += entry.getValue().size();
        }

        return res;
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

        CodeRunner(Instances instances, int instIndex, int attrIndex, List<Pair<Double, Pair<Integer, Integer>>> scores, double oldAcc) {
            this.instances = instances;
            this.instIndex = instIndex;
            this.attrIndex = attrIndex;
            this.scores = scores;
            this.oldAcc = oldAcc;
        }

        @Override
        public void run() {
            try {
                double score;
                score = concurrentGetScore(instances, instIndex, attrIndex, oldAcc, attrsClassifiers, attrsInstances);
                scores.add(new Pair<>(score, new Pair<>(instIndex, attrIndex)));

                // todo what
//                cls.setUseLaplace(true);
//                cls.buildClassifier(instances);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

    private static double concurrentGetScore(Instances instances,
                                             int instIndex,
                                             int attrIndex,
                                             double oldAcc,
                                             Pair<J48, Integer>[] attrsClassifiers,
                                             Instances[] attrsInstances) throws Exception {
        double score = 0.0;

        Pair<J48, Integer> attrClassifier = attrsClassifiers[attrIndex];
        if (attrClassifier.first == null) {
            score = concurrentGetUtility(instances, instIndex, attrIndex, attrClassifier.second, oldAcc);
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

    private static double[] concurrentGetProbs(Instances instances,
                                               int instIndex,
                                               int attrIndex,
                                               J48 attrClassifier,
                                               Instances[] attrsInstances) throws Exception {
        Instance inst = new Instance(instances.instance(instIndex));
        inst.setDataset(attrsInstances[attrIndex]);
        return attrClassifier.distributionForInstance(inst);
    }

    private static double concurrentGetUtility(Instances instances, int instIndex, int attrIndex, double valueIndex, double oldAcc) throws Exception {
        instances.instance(instIndex).setValue(attrIndex, valueIndex);
        J48 newClassifier = DatasetFactory.staticMakeClassifier(instances);
        double newAcc = DatasetFactory.calculateAccuracy(newClassifier, instances);
        instances.instance(instIndex).setMissing(attrIndex);

        return newAcc - oldAcc;
    }

    public int getRealPossibleQueries() {
        return realPossibleQueries;
    }
}
























