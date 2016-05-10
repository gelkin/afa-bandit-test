package ru.ifmo.ctddev.mazin.AFA;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;


public class AFABandit {
    private Instances instances;
    private QueryManager queryManager;

    private int n; // number of instances
    private int m; // number of attributes (features)

    private int totalNumQueries; // total number of queries

    private List<Integer> instNumQueries; // number of queries for instance (by index)
    private List<Integer> attrNumQueries; // number of queries for attribute (by index)

    private List<Double> instancesReward;
    private List<Double> attributesReward;

    private LinkedHashMap<Integer, Set<Integer>> possibleQueries; // list of attributes of instance with missing values

    public AFABandit(Instances instances, QueryManager queryManager) {
        this.instances = new Instances(instances);
        this.queryManager = queryManager;

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
     * @param b batch size to query at one step. If there are fewer than 'b'
     *          possible queries, acquire all remaining possible queries.
     * @return list of pairs (acquired queries, build classifier) for each step
     * @throws Exception
     */
    public List<Pair<List<Pair<Integer, Integer>>, J48>> perform(int k, int b) throws Exception {
        J48 classifier = makeClassifier();
        List<Pair<List<Pair<Integer, Integer>>, J48>> res = new LinkedList<>();
        for (int i = 0; i < k; ++i) {
            if (possibleQueries.size() == 0) {
                break; // nothing to query
            }

            setRewardsForInstances(classifier);
            setRewardsForAttributes();

            if (possibleQueries.size() < b) {
                b = possibleQueries.size();
            }
            List<Pair<Integer, Integer>> bestQueries = performStep(b);
            classifier = makeClassifier();
            res.add(new Pair<>(bestQueries, classifier));

            updatePossibleQueries();
        }

        return res;
    }

    private List<Pair<Integer, Integer>> performStep(int b) {
        // todo write your own normal pair
        List<Pair<Double, Pair<Integer, Integer>>> scores = new ArrayList<>();

        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
            int i = entry.getKey();
            for (Integer j : entry.getValue()) {
                double score = getScore(i, j);
                scores.add(new Pair<>(score, new Pair<>(i, j)));
            }
        }

        // Choose best 'b' queries to acquire:
        Collections.sort(scores, (o1, o2) -> o1.first < o2.first ? 1 : o1.first.equals(o2.first) ? 0 : -1);

        ArrayList<Pair<Integer, Integer>> bestQueries = scores.subList(0, b).stream()
                .map(pair -> pair.second)
                .collect(Collectors.toCollection(ArrayList::new));

        for (Pair<Integer, Integer> query : bestQueries) {
            acquireQuery(query.first, query.second);

            instNumQueries.set(query.first, instNumQueries.get(query.first) + 1);
            attrNumQueries.set(query.second, attrNumQueries.get(query.second) + 1);

            possibleQueries.get(query.first).remove(query.second);
        }

        return bestQueries;
    }

    private double getScore(int instIndex, int attrIndex) {
        double instanceScore = instancesReward.get(instIndex) +
                               Math.sqrt(2 * Math.log(totalNumQueries) /
                                         instNumQueries.get(instIndex));
        double attributeScore = attributesReward.get(attrIndex) +
                               Math.sqrt(2 * Math.log(totalNumQueries) /
                                         attrNumQueries.get(attrIndex));

        return instanceScore + attributeScore;
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
            if (maxIndex == instances.instance(i).classValue()) {
                // then maxProb - trueClassProb
                double secondMaxProb = Double.MIN_VALUE;
                for (int j = 0; j < probs.length; ++j) {
                    if (j != maxIndex && secondMaxProb < probs[j]) {
                        secondMaxProb = probs[j];
                    }
                }
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
        return maxProb - trueClassProb;
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
}
