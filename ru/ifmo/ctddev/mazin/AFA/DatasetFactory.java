package ru.ifmo.ctddev.mazin.AFA;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DatasetFactory {

    /**
     * Get deep copy of 'original' with random (percent * original.numAttributes())
     * attributes for each instance missing.
     *
     * @param original original instances set
     * @param percent percent of all attributes that will be assumed missing.
     *                'percent' value has to belong to interval (0, 1) (exclusively).
     * @return
     */
    public static Instances getInstancesWithMissingAttrs(Instances original, double percent) {
        if (percent >= 1.0 || percent <= 0.0) {
            throw new IllegalArgumentException("'percent' value has to belong to interval (0, 1) (exclusively)");
        }

        Instances resInstances = new Instances(original);

        int classIndex = resInstances.classIndex();
        List<Integer> indexesList;
        if (classIndex < 0) {
            // no class attribute
            indexesList = IntStream.range(0, resInstances.numAttributes())
                          .boxed().collect(Collectors.toList());
        } else {
            // exclude class attribute
            indexesList = IntStream.concat(IntStream.range(0, classIndex),
                                           IntStream.range(classIndex + 1, resInstances.numAttributes()))
                          .boxed().collect(Collectors.toList());
        }

        int numOfAttrsToSetMissing = (int) (indexesList.size() * percent);
        for (int i = 0; i < resInstances.numInstances(); ++i) {
            Collections.shuffle(indexesList);
            Instance inst = resInstances.instance(i);
            indexesList.subList(0, numOfAttrsToSetMissing).forEach(inst::setMissing);
        }

        return resInstances;
    }

    /**
     * todo
     *
     * @param cls
     * @param test
     * @return
     * @throws Exception
     */
    public static double calculateAvgAccuracy(J48 cls, Instances test) throws Exception {
        if (test.classIndex() < 0) {
            throw new IllegalArgumentException("The class index for the instances isn't specified");
        }

        double res = 0.0;
        int classesNumber = test.attribute(test.classIndex()).numValues();
        for (int curClass = 0; curClass < classesNumber; ++curClass) {
            int acc = 0;
            for (int instIndex = 0; instIndex < test.numInstances(); ++instIndex) {
                double classIndex = cls.classifyInstance(test.instance(instIndex));
                if (test.instance(instIndex).classValue() == curClass) {
                    if (classIndex == curClass) {
                        ++acc; // true positive
                    }
                } else if (classIndex != curClass) {
                    ++acc; // true negative
                }
            }
            res += ((double) acc) / test.numInstances();
        }

        return res / ((double) classesNumber);
    }

    /**
     * acc = Nc / Nt
     * , where Nt - number of test instances
     *         Nc - number of correctly classified instances
     *
     * @param cls
     * @param test
     * @return
     * @throws Exception
     */
    public static double calculateAccuracy(J48 cls, Instances test) throws Exception {
        if (test.classIndex() < 0) {
            throw new IllegalArgumentException("The class index for the instances isn't specified");
        }

        int acc = 0;
        for (int i = 0; i < test.numInstances(); ++i) {
            Instance inst = test.instance(i);
            if (inst.classValue() == cls.classifyInstance(test.instance(i))) {
                ++acc;
            }
        }

        return (double) acc / ((double) test.numInstances());
    }

    /**
     * todo
     *
     * @param instances
     * @param runsNum
     * @param seed
     * @param folds
     * @param percents
     * @param batchSize
     * @return
     * @throws Exception
     */
    public static Map<Integer, Double> afaBanditGetLerningCurve(Instances instances,
                                                                int runsNum,
                                                                int seed,
                                                                int folds,
                                                                double percents,
                                                                int batchSize) throws Exception {
        // todo
        int iterationsNumber = 1;

        ArrayList<Pair<Integer, List<Double>>> numToAcc = new ArrayList<>();
        for (int i = 0; i < runsNum; ++i) {
            // Randomize data
            Random rand = new Random(seed + i);
            Instances randData = new Instances(instances);   // create copy of original data
            randData.randomize(rand);
            randData.stratify(folds);

            for (int j = 0; j < folds; ++j) {
                Instances train = randData.trainCV(folds, j); // not always same size
                Instances test = randData.testCV(folds, j);

                int cnt = 0; // counter for numToAcc

                // Add missing
                Instances trainMissing = DatasetFactory.getInstancesWithMissingAttrs(train, percents);
                System.out.println(train.numInstances());

                QueryManager queryManager = new SimpleQueryManager(train);
                AFABandit afaMethod = new AFABandit(trainMissing, queryManager);

                // Zero step
                int num = afaMethod.getAllQueriesNum() - afaMethod.getPossibleQueriesNum();
                J48 initialCls = afaMethod.makeClassifier();
                double acc = DatasetFactory.calculateAccuracy(initialCls, test);
                if (numToAcc.size() <= cnt) { // todo
                    List<Double> accs = new ArrayList<>();
                    accs.add(acc);
                    numToAcc.add(new Pair<>(num, accs));
                } else {
                    numToAcc.get(cnt).second.add(acc);
                }
                ++cnt;


                List<Pair<List<Pair<Integer, Integer>>, J48>> res = afaMethod.perform(iterationsNumber, batchSize);
                J48 lastCls = new J48();
                while (res.size() > 0) {
                    num = num + res.get(0).first.size();
                    acc = DatasetFactory.calculateAccuracy(res.get(0).second, test);
                    if (numToAcc.size() <= cnt) { // todo
                        List<Double> accs = new ArrayList<>();
                        accs.add(acc);
                        numToAcc.add(new Pair<>(num, accs));
                    } else {
                        numToAcc.get(cnt).second.add(acc);
                    }
                    ++cnt;

                    lastCls = res.get(0).second; // todo

                    res = afaMethod.perform(iterationsNumber, batchSize);
                }

                if (j == folds - 1) { // todo
                    J48 cls = lastCls;
                    System.out.println("Super test of last classifier:");
                    System.out.println("Acccuracy = " + DatasetFactory.calculateAccuracy(cls, test));
                    System.out.println("_____");
                }
            }
        }

        int allQueriesNum = (int) ((double) (instances.numInstances() * (instances.numAttributes() - 1)) * ((double) (folds - 1) / folds));
        Map<Integer, Double> numToAccMap = new LinkedHashMap<>();
        for (Pair<Integer, List<Double>> pair : numToAcc) {
            double avgAcc = pair.second.stream().mapToDouble(val -> val).average().getAsDouble();
            numToAccMap.put((int) (Math.round(100.0 * ((double) pair.first / (double) allQueriesNum))), avgAcc);
//            numToAccMap.put(pair.first, avgAcc);
        }

        return numToAccMap;
    }

    public static Map<Integer, Double> seuIniformSamplingGetLerningCurve(Instances instances,
                                                                int runsNum,
                                                                int seed,
                                                                int folds,
                                                                double percents,
                                                                int batchSize,
                                                                int alpha) throws Exception {
        // todo
        int iterationsNumber = 1;

        Set<Integer> numericAttrsIndexes = getNumericAttrsIndexes(instances);

        Discretize discretizer = new Discretize();
        discretizer.setInputFormat(instances);
        Instances discInstances = Filter.useFilter(instances, discretizer);

        ArrayList<Pair<Integer, List<Double>>> numToAcc = new ArrayList<>();
        for (int i = 0; i < runsNum; ++i) {
            // Randomize data
            Random rand = new Random(seed + i);
            Instances randData = new Instances(discInstances);   // create copy of original data
            randData.randomize(rand);
            randData.stratify(folds);

            for (int j = 0; j < folds; ++j) {
                Instances train = randData.trainCV(folds, j); // todo: not always same size
                Instances test = randData.testCV(folds, j);

                int cnt = 0; // counter for numToAcc

                // Add missing
                Instances trainMissing = DatasetFactory.getInstancesWithMissingAttrs(train, percents);
                System.out.println(train.numInstances());

                QueryManager queryManager = new SimpleQueryManager(train);
                SEUUniformSampling seuMethod = new SEUUniformSampling(trainMissing,
                        queryManager,
                        alpha,
                        batchSize,
                        discretizer,
                        numericAttrsIndexes);

                // Zero step
                int num = seuMethod.getAllQueriesNum() - seuMethod.getPossibleQueriesNum();
                J48 initialCls = seuMethod.makeClassifier();
                double acc = DatasetFactory.calculateAccuracy(initialCls, test);
                if (numToAcc.size() <= cnt) { // todo
                    List<Double> accs = new ArrayList<>();
                    accs.add(acc);
                    numToAcc.add(new Pair<>(num, accs));
                } else {
                    numToAcc.get(cnt).second.add(acc);
                }
                ++cnt;


                List<Pair<List<Pair<Integer, Integer>>, J48>> res = seuMethod.perform(iterationsNumber);
                J48 lastCls = new J48();
                while (res.size() > 0) {
                    num = num + res.get(0).first.size();
                    acc = DatasetFactory.calculateAccuracy(res.get(0).second, test);
                    if (numToAcc.size() <= cnt) { // todo
                        List<Double> accs = new ArrayList<>();
                        accs.add(acc);
                        numToAcc.add(new Pair<>(num, accs));
                    } else {
                        numToAcc.get(cnt).second.add(acc);
                    }
                    ++cnt;

                    lastCls = res.get(0).second; // todo

                    res = seuMethod.perform(iterationsNumber);
                }

                if (j == folds - 1) { // todo
                    J48 cls = lastCls;
                    System.out.println("Super test of last classifier:");
                    System.out.println("Acccuracy = " + DatasetFactory.calculateAccuracy(cls, test));
                    System.out.println("_____");
                }
            }
        }

        int allQueriesNum = (int) ((double) (instances.numInstances() * (instances.numAttributes() - 1)) * ((double) (folds - 1) / folds));
        Map<Integer, Double> numToAccMap = new LinkedHashMap<>();
        for (Pair<Integer, List<Double>> pair : numToAcc) {
            double avgAcc = pair.second.stream().mapToDouble(val -> val).average().getAsDouble();
            numToAccMap.put((int) (Math.round(100.0 * ((double) pair.first / (double) allQueriesNum))), avgAcc);
        }

        return numToAccMap;
    }

    public static Set<Integer> getNumericAttrsIndexes(Instances instances) {
        Set<Integer> res = new HashSet<>();
        for (int i = 0; i < instances.numAttributes(); ++i) {
            if (instances.attribute(i).isNumeric()) {
                res.add(i);
            }
        }
        return res;
    }
}
