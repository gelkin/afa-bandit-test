package ru.ifmo.ctddev.mazin.AFA;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.text.DecimalFormat;
import java.util.*;
import java.util.stream.Collectors;


public class AFABandit extends AFAMethod {
    private int totalNumQueries; // total number of queries

    private List<Integer> instNumQueries; // number of queries for instance (by index)
    private List<Integer> attrNumQueries; // number of queries for attribute (by index)

    private List<Double> instancesReward;
    private List<Double> attributesReward;

    private double alpha; // todo obj coef
    private double beta; // todo feat coef

    public AFABandit(Instances instances, QueryManager queryManager, int b) {
        this.instances = new Instances(instances);
        this.queryManager = queryManager;
        this.b = b;
        alpha = 1.0;
        beta = 1.0;

        init();
    }

    public AFABandit(Instances instances, QueryManager queryManager, int b, double alpha) {
        this.instances = new Instances(instances);
        this.queryManager = queryManager;
        this.b = b;
        this.alpha = alpha;
        beta = 1.0 - alpha;

        init();
    }

    public AFABandit(Instances instances, QueryManager queryManager, int b, double alpha, double beta) {
        this.instances = new Instances(instances);
        this.queryManager = queryManager;
        this.b = b;
        this.alpha = alpha;
        this.beta = beta;

        init();
    }

    private void init() throws IllegalArgumentException {
        String msg;
        if ((msg = areInstancesCorrect()) != null) {
            throw new IllegalArgumentException(msg);
        }

        n = instances.numInstances();
        m = instances.numAttributes() - 1;

        instNumQueries = new ArrayList<>(n);
        for (int i = 0; i < n; ++i) {
            instNumQueries.add(0);
        }

        attrNumQueries = new ArrayList<>(m);
        for (int i = 0; i < m; ++i) {
            attrNumQueries.add(0);
        }

        possibleQueries = new LinkedHashMap<>(n);
        for (int i = 0; i < n; ++i) {
            Set<Integer> possibleAttrsToQuery = new HashSet<>();
            for (int j = 0; j < m; ++j) {
                if (!instances.instance(i).isMissing(j)) {
                    instNumQueries.set(i, instNumQueries.get(i) + 1);
                    attrNumQueries.set(j, attrNumQueries.get(j) + 1);
                    ++totalNumQueries;
                } else {
                    possibleAttrsToQuery.add(j);
                }
            }
            if (!possibleAttrsToQuery.isEmpty()) {
                possibleQueries.put(i, possibleAttrsToQuery);
            }
        }

        instancesReward = new ArrayList<>(n);
        for (int i = 0; i < n; ++i) {
            instancesReward.add(0.0);
        }

        attributesReward = new ArrayList<>(m);
        for (int i = 0; i < m; ++i) {
            attributesReward.add(0.0);
        }
    }

    private String areInstancesCorrect() {
        String msg = null;
        if (instances == null) {
            msg = "The instances are null";
        } else if (instances.classIndex() < 0) {
            msg = "The class index for the instances isn't specified";
        } else if (instances.classIndex() != (instances.numAttributes() - 1)) {
            msg = "The class index for the instances is not the last attribute";
        } else {
            for (int i = 0; i < instances.numInstances(); ++i) {
                if (instances.instance(i).classIsMissing()) {
                    msg = "The class labels should be specified for all the " +
                          "instances, but it isn't specified for instance #" + i;
                    break;
                }
            }
        }

        return msg;
    }

    /**
     * Perform AFABandit method for active feature-value acquiring.
     *
     * @param k number of performStep running
     * @return list of pairs (acquired queries, build classifier) for each step
     * @throws Exception
     */
    public List<Pair<List<Pair<Integer, Integer>>, J48>> perform(int k) throws Exception {
        J48 classifier = makeClassifier();
        List<Pair<List<Pair<Integer, Integer>>, J48>> res = new LinkedList<>();
        for (int i = 0; i < k; ++i) {
            if (possibleQueries.size() == 0) {
                break; // nothing to query
            }

            setRewardsForInstances(classifier);
            setRewardsForAttributes();

            // todo
//            for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
//                int instInd = entry.getKey();
//                if (DatasetFactory.isInstMisclassified(classifier, instances.instance(instInd))) {
//                    System.out.println(String.format("Object %s has margin = %s and is misclassified", instInd, instancesReward.get(instInd)));
//                }
//            }
//            System.out.println("end.. another one");

            int batchSize = b;
            int possibleQueriesNum = getRealPossibleQueries();
            if (possibleQueriesNum < b) { // todo
                batchSize = possibleQueriesNum;
            }

            List<Pair<Integer, Integer>> bestQueries = performStep(batchSize);
            classifier = makeClassifier();
            res.add(new Pair<>(bestQueries, classifier));

            updatePossibleQueries();
        }

        return res;
    }

// todo
//    boolean firstTime = true;

    private List<Pair<Integer, Integer>> performStep(int b) {
        List<Pair<Double, Pair<Integer, Integer>>> scores = new ArrayList<>();

//        todo
//        DecimalFormat df = new DecimalFormat("#.####");
//        System.out.println("Attribute's reward");
//        for (int i = 0; i < attributesReward.size(); ++i) {
//            System.out.print(i + ": " + Double.valueOf(df.format(attributesReward.get(i))) + ", ");
//        }
//        System.out.println("");
//        System.out.println("Instance's reward");
//        for (int i = 0; i < instancesReward.size(); ++i) {
//            System.out.print(i + ": " + Double.valueOf(df.format(instancesReward.get(i))) + ", ");
//        }
//        System.out.println("");

//        System.out.println(String.format("%s\t%s\t\t\tREWARD: INST\t\tATTR\t\tCOMPL INST\t\tATTR\t\tSCORE", "i", "j"));
//        todo

        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
            int i = entry.getKey();
            for (Integer j : entry.getValue()) {
                double score = getScoreAttr(i, j);
                scores.add(new Pair<>(score, new Pair<>(i, j)));
//                if (score < Double.MAX_VALUE && score > Double.MIN_VALUE) {
//                    System.out.print(String.format("(%s, %s, %s), ", i, j, Double.valueOf(df.format(score))));
//                }
            }
//            todo
//            System.out.println("");
        }

//        todo
//        firstTime = false;
//        System.out.println("-----------------");


        // Choose best 'b' queries to acquire:
        Collections.sort(scores, (o1, o2) -> o1.first < o2.first ? 1 : o1.first.equals(o2.first) ? 0 : -1);

        ArrayList<Pair<Integer, Integer>> bestQueries = scores.subList(0, b).stream()
                .map(pair -> pair.second)
                .collect(Collectors.toCollection(ArrayList::new));

        for (Pair<Integer, Integer> query : bestQueries) {
            acquireQuery(query.first, query.second);

            instNumQueries.set(query.first, instNumQueries.get(query.first) + 1);
            attrNumQueries.set(query.second, attrNumQueries.get(query.second) + 1);

            ++totalNumQueries;

            possibleQueries.get(query.first).remove(query.second);
        }

        return bestQueries;
    }

    private double getScoreInst(int instIndex, int attrIndex) {
        double instReward = instancesReward.get(instIndex);
        double instExploration = Math.sqrt(2 * Math.log(totalNumQueries) /
                instNumQueries.get(instIndex));
        return instReward + instExploration;
    }

    private double getScoreAttr(int instIndex, int attrIndex) {
        double attrReward = attributesReward.get(attrIndex);
        double attrExploration = Math.sqrt(2 * Math.log(totalNumQueries) /
                                           attrNumQueries.get(attrIndex));
        return attrReward + attrExploration;
    }

    private double getScoreMixed(int instIndex, int attrIndex) {
        double instReward = instancesReward.get(instIndex); // todo CHANGE REWARD
        double attrReward = attributesReward.get(attrIndex);

        double commonExploration = Math.sqrt(2 * Math.log(totalNumQueries) /
                (instNumQueries.get(instIndex) + attrNumQueries.get(attrIndex)));

        double score = alpha * instReward + beta * attrReward + commonExploration;


//        boolean showLog = false;
//        if (showLog && firstTime) {
//            DecimalFormat df = new DecimalFormat("#.####");
//
//            // todo
//            int percInstReward = (int) (100 * instReward / score);
//            int percAttrReward = (int) (100 * attrReward / score);
//            int percInstExploration = (int) (100 * instExploration / score);
//            int percAttrExploration = (int) (100 * attrExploration / score);
//
//            System.out.println(String.format("%s\t%s\t\t\t%s%%(%s)\t\t%s%%(%s)\t\t%s%%(%s)\t\t%s%%(%s)\t\t%s", instIndex, attrIndex
//                    , percInstReward, Double.valueOf(df.format(instReward))
//                    , percAttrReward, Double.valueOf(df.format(attrReward))
//                    , percInstExploration, Double.valueOf(df.format(instExploration))
//                    , percAttrExploration, Double.valueOf(df.format(attrExploration))
//                    , Double.valueOf(df.format(score))));
//        }
        // todo

        return score;
    }

    private void acquireQuery(int instIndex, int attrIndex) {
        Instance inst = instances.instance(instIndex);
        inst.setValue(attrIndex, queryManager.getValue(instIndex, attrIndex));
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

    private void setRewardsForInstances(J48 classifier) throws Exception {
        for (int i = 0; i < n; ++i) {
            double[] probs = classifier.distributionForInstance(instances.instance(i));

            // find class with maximum probability
            int maxIndex = -1;
            double maxProb = Double.MIN_VALUE;
            for (int j = 0; j < probs.length; ++j) {
                if (maxProb < probs[j]) {
                    maxIndex = j;
                    maxProb = probs[j];
                }
            }

            // if instance is classified correctly
            if (maxIndex == (int) instances.instance(i).classValue()) {
                // then maxProb - trueClassProb
                double secondMaxProb = Double.MIN_VALUE;
                for (int j = 0; j < probs.length; ++j) {
                    if (j != maxIndex && secondMaxProb < probs[j]) {
                        secondMaxProb = probs[j];
                    }
                }
                // then trueClassProb == maxProb:
                instancesReward.set(i, getInstReward(maxProb, secondMaxProb));
            } else { // if instance is misclassified
                double trueClassProb = probs[(int) instances.instance(i).classValue()];
                instancesReward.set(i, getInstReward(trueClassProb, maxProb));
            }
        }
    }

    /**
     * Returns substraction of probabilities.
     * If instance is correctly classified, then reward is negative and it is higher
     * when probabilities are closer (so classifier is worse on this instance)
     * If instance is misclassified, then reward is positive and it is higher
     * when probabilities are farther (so classifier is worse on this instance)
     *
     * @param trueClassProb probability of instance to belong to its true class
     * @param maxProb max probability for instance exclusive trueClassProb
     * @return
     */
    private double getInstReward(double trueClassProb, double maxProb) {
        return (1.0 + (maxProb - trueClassProb)) / 2.0;
    }

    private void setRewardsForAttributes() throws Exception {
        InfoGainAttributeEval ig = new InfoGainAttributeEval();
        ig.buildEvaluator(instances);

        for (int i = 0; i < m; ++i) {
            attributesReward.set(i, ig.evaluateAttribute(i));
        }
    }

    public J48 makeClassifier() throws Exception {
        J48 classifier = new J48();
        classifier.setUseLaplace(true);
        classifier.buildClassifier(instances);

        return classifier;
    }

    public int getInstanceNum() {
        return n;
    }

    public int getAttributesNum() {
        return m;
    }

    public int getRealPossibleQueries() {
        int res = 0;
        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
            res += entry.getValue().size();
        }

        return res;
    }



    public int getAllQueriesNum() {
        return n * m;
    }
}
