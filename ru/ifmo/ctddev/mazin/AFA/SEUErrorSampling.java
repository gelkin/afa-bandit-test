package ru.ifmo.ctddev.mazin.AFA;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
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
public class SEUErrorSampling implements AFAMethod {
    private Instances instances;
    private QueryManager queryManager;
    private Map<Integer, Set<Integer>> possibleQueries; // list of attributes of instance with missing values

    private int realPossibleQueries;

    private final int b; // size of query batch

    private int esParam; // param which controls the complexity of the search:
                         // at each step we choose 'esParam' instances, and among them
                         // choose 'b' missing queries to acquire

    private J48[] attrsClassifiers; // for getProb(..) computing
    private Instances[] attrsInstances; // for getProb(..) computing

    private int n;
    private int m;

    Discretize discretizer;
    Set<Integer> numericAttrsIndexes;

    public SEUErrorSampling(Instances instances,
                              QueryManager queryManager,
                              int esParam,
                              int b,
                              Discretize discretizer,
                              Set<Integer> numericAttrsIndexes) {
        this.instances = new Instances(instances);
        this.queryManager = queryManager;
        this.esParam = esParam;
        this.b = b;
        this.discretizer = discretizer; // todo;
        this.numericAttrsIndexes = numericAttrsIndexes; // todo

        init();
    }

    private void init() throws IllegalArgumentException {
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
    }

    public List<Pair<List<Pair<Integer, Integer>>, J48>> perform(int k) throws Exception {
        J48 classifier = makeClassifier();
        List<Pair<List<Pair<Integer, Integer>>, J48>> res = new LinkedList<>();
        while (k-- > 0) {
            possibleQueries = getPossibleQueries(classifier);

            if (possibleQueries.size() == 0) {
                break; // nothing to query
            }

            int batchSize = b;
            if (possibleQueries.size() < b) {
                batchSize = possibleQueries.size();
            }

            List<Pair<Integer, Integer>> bestQueries = concurrentPerformStep(batchSize, classifier);
            classifier = makeClassifier();
            res.add(new Pair<>(bestQueries, classifier));
        }

        return res;
    }

    private List<Pair<Integer, Integer>> performStep(int batchSize, J48 classifier) throws Exception {
        attrsClassifiers = new J48[m]; // todo mb optimize updating - update not all
        attrsInstances = new Instances[m]; // todo mb optimize updating - update not all
        List<Pair<Double, Pair<Integer, Integer>>> scores = new ArrayList<>();
        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
            int i = entry.getKey();
            for (Integer j : entry.getValue()) {
                double score = getScore(i, j, classifier);
                scores.add(new Pair<>(score, new Pair<>(i, j)));
            }
        }
        // Choose best 'b' queries to acquire:
        Collections.sort(scores, (o1, o2) -> o1.first < o2.first ? 1 : o1.first.equals(o2.first) ? 0 : -1);
        ArrayList<Pair<Integer, Integer>> bestQueries = scores.subList(0, batchSize).stream()
                .map(pair -> pair.second)
                .collect(Collectors.toCollection(ArrayList::new));

        for (Pair<Integer, Integer> query : bestQueries) {
            acquireQuery(query.first, query.second);
        }

        return bestQueries;
    }

    /**
     * Perform step with calculation score for each possible query in parallel
     *
     * @param batchSize
     * @param classifier
     * @return
     * @throws Exception
     */
    private List<Pair<Integer, Integer>> concurrentPerformStep(int batchSize, J48 classifier) throws Exception {
        attrsClassifiers = new J48[m];
        attrsInstances = new Instances[m];
        for (int i = 0; i < m; ++i) {
            initClassifierForAttr(i);
        }

        List<Pair<Double, Pair<Integer, Integer>>> scores = Collections.synchronizedList(new ArrayList<>());
        ExecutorService execSvc = Executors.newCachedThreadPool();
        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
            int i = entry.getKey();
            for (Integer j : entry.getValue()) {
                CodeRunner runner = new CodeRunner(new Instances(instances), (J48) Classifier.makeCopy(classifier), i, j, scores);
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

        return bestQueries;
    }

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

    private List<Integer> getInterestingInstances(J48 classifier) throws Exception {
        ArrayList<Integer> misclassified = new ArrayList<>(n);
        for (int i = 0; i < n; ++i) {
            Instance inst = instances.instance(i);
            if (inst.hasMissingValue()) {
                double value = classifier.classifyInstance(inst);
                if (inst.classValue() == value) {
                    misclassified.add(i);
                }
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
                if (inst.hasMissingValue()) {
                    double value = classifier.classifyInstance(inst);
                    if (inst.classValue() != value) {
                        double score = getUncertaintyScore(inst, classifier);
                        scores.add(new Pair<>(score, i));
                    }
                }
            }
            Collections.sort(scores, (o1, o2) -> o1.first > o2.first ? 1 : o1.first.equals(o2.first) ? 0 : -1);
            int subListSize = Math.min(scores.size(), esParam - misclassified.size());
            ArrayList<Integer> bestInstances
                    = scores.subList(0, subListSize).stream()
                    .map(pair -> pair.second)
                    .collect(Collectors.toCollection(ArrayList::new));
            matchedInstances.addAll(bestInstances);
        }

        return matchedInstances;
    }

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
    private double getScore(int instIndex, int attrIndex, J48 classifier) throws Exception {
        double score = 0.0;
        double[] estimatedProbs = getProbs(instIndex, attrIndex);

        int numValues = instances.attribute(attrIndex).numValues();
        for (int valueIndex = 0; valueIndex < numValues; ++valueIndex) {
            score += estimatedProbs[valueIndex] *
                    getUtility(instIndex, attrIndex, valueIndex, classifier);
        }
        return score;
    }

    /**
     * The probability that 'query' has the value 'value'. todo
     *
     * @param instIndex
     * @param attrIndex
     * @return
     * @throws Exception
     */
    private double[] getProbs(int instIndex, int attrIndex) throws Exception {
        if (attrsClassifiers[attrIndex] == null) {
            initClassifierForAttr(attrIndex);
        }
        Instance inst = new Instance(instances.instance(instIndex));
        inst.setDataset(attrsInstances[attrIndex]);
        return attrsClassifiers[attrIndex].distributionForInstance(inst);
    }


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
    private double getUtility(int instIndex, int attrIndex, double value, J48 classifier) throws Exception {
        instances.instance(instIndex).setValue(attrIndex, value);
        J48 newClassifier = makeClassifier();
        double newAcc = DatasetFactory.calculateAccuracy(newClassifier, instances);
        instances.instance(instIndex).setMissing(attrIndex);

        double oldAcc = DatasetFactory.calculateAccuracy(classifier, instances);

//        return (newAcc - oldAcc) / C[instIndex][attrIndex];
        return newAcc - oldAcc;
    }

    /**
     * todo
     *
     * @return
     * @throws Exception
     */
    public J48 makeClassifier() throws Exception {
        J48 classifier = new J48();
        classifier.setUseLaplace(true); // todo as in paper
        classifier.buildClassifier(instances);
        return classifier;
    }

    private void initClassifierForAttr(int attrIndex) throws Exception {
        // todo get rid of copying
        int cap = n;
        for (int i = 0; i < n; ++i) {
            if (Instance.isMissingValue(instances.instance(i).value(attrIndex))) {
                --cap;
            }
        }
        Instances instancesForAttr = new Instances(instances, cap);
        for (int i = 0; i < n; ++i) {
            if (Instance.isMissingValue(instances.instance(i).value(attrIndex))) {
                instancesForAttr.add(instances.instance(i));
            }
        }
        instancesForAttr.setClassIndex(attrIndex);

        J48 classifier = new J48();
        classifier.setUseLaplace(true); // todo as in paper
        classifier.buildClassifier(instancesForAttr);

        attrsInstances[attrIndex] = instancesForAttr;
        attrsClassifiers[attrIndex] = classifier;
    }

    public int getPossibleQueriesNum(Map<Integer, Set<Integer>> possibleQueries) {
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

    public int getAllQueriesNum() {
        return n * m;
    }

    class CodeRunner implements Runnable {

        public Instances instances;
        public J48 cls;
        public int instIndex;
        public int attrIndex;
        public List<Pair<Double, Pair<Integer, Integer>>> scores;

        CodeRunner(Instances instances, J48 cls, int instIndex, int attrIndex, List<Pair<Double, Pair<Integer, Integer>>> scores) {
            this.instances = instances;
            this.cls = cls;
            this.instIndex = instIndex;
            this.attrIndex = attrIndex;
            this.scores = scores;
        }

        @Override
        public void run() {
            try {
                double score;
                score = concurrentGetScore(instances, instIndex, attrIndex, cls, attrsClassifiers, attrsInstances);
                scores.add(new Pair<>(score, new Pair<>(instIndex, attrIndex)));

                cls.setUseLaplace(true);
                cls.buildClassifier(instances);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

    private static double concurrentGetScore(Instances instances,
                                             int instIndex,
                                             int attrIndex,
                                             J48 classifier,
                                             J48[] attrsClassifiers,
                                             Instances[] attrsInstances) throws Exception {
        double score = 0.0;
        double[] estimatedProbs = concurrentGetProbs(instances, instIndex, attrIndex, attrsClassifiers, attrsInstances);

        int numValues = instances.attribute(attrIndex).numValues();
        for (int valueIndex = 0; valueIndex < numValues; ++valueIndex) {
            score += estimatedProbs[valueIndex] *
                    concurrentGetUtility(instances, instIndex, attrIndex, valueIndex, classifier);
        }
        return score;
    }

    private static double[] concurrentGetProbs(Instances instances,
                                               int instIndex,
                                               int attrIndex,
                                               J48[] attrsClassifiers,
                                               Instances[] attrsInstances) throws Exception {
        Instance inst = new Instance(instances.instance(instIndex));
        inst.setDataset(attrsInstances[attrIndex]);
        return attrsClassifiers[attrIndex].distributionForInstance(inst);
    }

    private static double concurrentGetUtility(Instances instances, int instIndex, int attrIndex, double value, J48 classifier) throws Exception {
        instances.instance(instIndex).setValue(attrIndex, value);
        J48 newClassifier = staticMakeClassifier(instances);
        double newAcc = DatasetFactory.calculateAccuracy(newClassifier, instances);
        instances.instance(instIndex).setMissing(attrIndex);

        double oldAcc = DatasetFactory.calculateAccuracy(classifier, instances);

//        return (newAcc - oldAcc) / C[instIndex][attrIndex];
        return newAcc - oldAcc;
    }

    private static J48 staticMakeClassifier(Instances instances) throws Exception {
        J48 classifier = new J48();
        classifier.setUseLaplace(true); // todo as in paper
        classifier.buildClassifier(instances);
        return classifier;
    }

    public int getRealPossibleQueries() {
        return realPossibleQueries;
    }
}
























