package ru.ifmo.ctddev.mazin.AFA;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;

public class RandomAFA extends AFAMethod {
    public RandomAFA(Instances instances, QueryManager queryManager, int b) {
        this.instances = new Instances(instances);
        this.queryManager = queryManager;
        this.b = b;

        init();
    }

    private void init() throws IllegalArgumentException {
        String msg;
        if ((msg = areInstancesCorrect()) != null) {
            throw new IllegalArgumentException(msg);
        }

        n = instances.numInstances();
        m = instances.numAttributes() - 1;

        possibleQueries = new LinkedHashMap<>(n);
        for (int i = 0; i < n; ++i) {
            Set<Integer> possibleAttrsToQuery = new HashSet<>();
            for (int j = 0; j < m; ++j) {
                if (instances.instance(i).isMissing(j)) {
                    possibleAttrsToQuery.add(j);
                }
            }
            if (!possibleAttrsToQuery.isEmpty()) {
                possibleQueries.put(i, possibleAttrsToQuery);
            }
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
        J48 classifier;
        List<Pair<List<Pair<Integer, Integer>>, J48>> res = new LinkedList<>();
        for (int i = 0; i < k; ++i) {
            if (possibleQueries.size() == 0) {
                break; // nothing to query
            }

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
        List<Pair<Integer, Integer>> queries = new ArrayList<>();

        for (Map.Entry<Integer, Set<Integer>> entry : possibleQueries.entrySet()) {
            int i = entry.getKey();
            for (Integer j : entry.getValue()) {
                queries.add(new Pair<>(i, j));
            }
        }

        // Choose best 'b' queries to acquire:
        Collections.shuffle(queries);

        List<Pair<Integer, Integer>> bestQueries = queries.subList(0, b);

        for (Pair<Integer, Integer> query : bestQueries) {
            acquireQuery(query.first, query.second);

            possibleQueries.get(query.first).remove(query.second);
        }

        return bestQueries;
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