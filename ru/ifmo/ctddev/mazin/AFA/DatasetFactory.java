package ru.ifmo.ctddev.mazin.AFA;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DatasetFactory {


    public static Instances reduceInstancesNumber(Instances instances, double percent) {
        ArrayList<List<Integer>> instanceByClass = new ArrayList<>(instances.numClasses());
        for (int i = 0; i < instances.numClasses(); ++i) {
            instanceByClass.add(new ArrayList<>());
        }

        Instances res = new Instances(instances, (int) (instances.numInstances() * percent));
        for (int i = 0; i < instances.numInstances(); ++i) {
            instanceByClass.get((int) instances.instance(i).classValue()).add(i);
        }

        for (int i = 0; i < instanceByClass.size(); ++i) {
            Collections.shuffle(instanceByClass.get(i));
            instanceByClass.set(i, instanceByClass.get(i).subList(0, (int) (instanceByClass.get(i).size() * percent)));
        }

        for (int i = 0; i < instanceByClass.size(); ++i) {
            for (int j : instanceByClass.get(i)) {
                res.add(instances.instance(j));
            }
        }

        return res;
    }

    public static J48 staticMakeClassifier(Instances instances) throws Exception {
        J48 classifier = new J48();
        classifier.setUseLaplace(true); // todo as in paper
        classifier.buildClassifier(instances);
        return classifier;
    }

    /**
     * Make deep copy of 'original' with random (percent * original.numAttributes())
     * attributes for each instance missing.
     *
     * @param original original instances set
     * @param percent percent of all attributes that will be assumed missing.
     *                'percent' value has to belong to interval (0, 1) (exclusively).
     * @return
     */
    public static Instances makeWithMissingAttrsForEachInstUniformly(Instances original, double percent) {
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
     * Make deep copy of 'original' with random (percent * original.numInstances() * original.numAttributes())
     * values in 'original' dataset missing.
     *
     * @param original original instances set
     * @param percent percent of all attributes that will be assumed missing.
     *                'percent' value has to belong to interval (0, 1) (exclusively).
     * @return
     */
    public static Instances makeWithMissingAttrsUniformly(Instances original, double percent) {
        if (percent >= 1.0 || percent <= 0.0) {
            throw new IllegalArgumentException("'percent' value has to belong to interval (0, 1) (exclusively)");
        }
        int n = original.numInstances();
        int m = original.numAttributes();

        Instances resInstances = new Instances(original);
        int classIndex = resInstances.classIndex();
        List<Pair<Integer, Integer>> toBeMissingList = new ArrayList<>(n * m);
        if (classIndex < 0) {
            // no class attribute
            for (int i = 0; i < original.numInstances(); ++i) {
                for (int j = 0; j < original.numAttributes(); ++j) {
                    toBeMissingList.add(new Pair<>(i, j));
                }
            }
        } else {
            // exclude class attribute
            for (int i = 0; i < original.numInstances(); ++i) {
                for (int j = 0; j < original.numAttributes(); ++j) {
                    if (j != classIndex) {
                        toBeMissingList.add(new Pair<>(i, j));
                    }
                }
            }
        }

        int numOfValuesToSetMissing = (int) (toBeMissingList.size() * percent);
        Collections.shuffle(toBeMissingList);
        toBeMissingList = toBeMissingList.subList(0, numOfValuesToSetMissing);
        for (Pair<Integer, Integer> cell : toBeMissingList) {
            resInstances.instance(cell.first).setMissing(cell.second);
        }

        return resInstances;
    }

    public static boolean isInstMisclassified(Instance inst, J48 cls) throws Exception {
        return inst.classValue() != cls.classifyInstance(inst);
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

        int cnt = 0;
        for (int i = 0; i < test.numInstances(); ++i) {
            Instance inst = test.instance(i);
            if (inst.classValue() == (int) cls.classifyInstance(test.instance(i))) {
                ++cnt;
            }
        }

        return (double) cnt / ((double) test.numInstances());
    }

    /**
     * Random active feature-value acquiring
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
    public static Map<Integer, List<Double>> randomAFAGetLerningCurve(Instances instances,
                                                                      int runsNum,
                                                                      int seed,
                                                                      int folds,
                                                                      double percents,
                                                                      int batchSize) throws Exception {
        // todo
        int itersNum = 1;

        int allQueriesNum = (int) ((double) (instances.numInstances() * (instances.numAttributes() - 1)) * ((double) (folds - 1) / folds));
        Map<Integer, List<Double>> numToAccByRuns = new LinkedHashMap<>();
        List<Pair<Integer, List<Double>>> numToAcc = new ArrayList<>();
        for (int i = 0; i < runsNum; ++i) {
            // Randomize data
            Random rand = new Random(seed + i);
            Instances randData = new Instances(instances);   // create copy of original data
            randData.randomize(rand);
            randData.stratify(folds);

            for (int j = 0; j < folds; ++j) {
                Instances train = randData.trainCV(folds, j); // not always same size
                Instances test = randData.testCV(folds, j);

                // Add missing
                Instances trainMissing = DatasetFactory.makeWithMissingAttrsUniformly(train, percents);
                System.out.println(String.format("runNum = %s, foldNum = %s", i, j));

                QueryManager queryManager = new SimpleQueryManager(train);
                RandomAFA afaMethod = new RandomAFA(trainMissing, queryManager, batchSize);

                getLearningCurveHelper(numToAcc, afaMethod, test, itersNum, folds, j, false);
            }

            collectInfoFromRun(numToAcc, numToAccByRuns, allQueriesNum);
            numToAcc.clear();
        }

        return numToAccByRuns;
    }


    /**
     * AfaBanditGetLerningCurve
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
    public static Map<Integer, List<Double>> afaBanditGetLerningCurve(Instances instances,
                                                                int runsNum,
                                                                int seed,
                                                                int folds,
                                                                double percents,
                                                                int batchSize) throws Exception {
        // todo
        int itersNum = 1;

        int allQueriesNum = (int) ((double) (instances.numInstances() * (instances.numAttributes() - 1)) * ((double) (folds - 1) / folds));
        Map<Integer, List<Double>> numToAccByRuns = new LinkedHashMap<>();
        List<Pair<Integer, List<Double>>> numToAcc = new ArrayList<>();
        for (int i = 0; i < runsNum; ++i) {
            // Randomize data
            Random rand = new Random(seed + i);
            Instances randData = new Instances(instances);   // create copy of original data
            randData.randomize(rand);
            randData.stratify(folds);

            for (int j = 0; j < folds; ++j) {
                Instances train = randData.trainCV(folds, j); // not always same size
                Instances test = randData.testCV(folds, j);

                // Add missing
                Instances trainMissing = DatasetFactory.makeWithMissingAttrsUniformly(train, percents);
                System.out.println(String.format("runNum = %s, foldNum = %s", i, j));

                QueryManager queryManager = new SimpleQueryManager(train);
                AFABandit afaMethod = new AFABandit(trainMissing, queryManager, batchSize);

                getLearningCurveHelper(numToAcc, afaMethod, test, itersNum, folds, j, false);
            }

            collectInfoFromRun(numToAcc, numToAccByRuns, allQueriesNum);
            numToAcc.clear();
        }

        return numToAccByRuns;
    }


    /**
     * SeuIniformSamplingGetLerningCurve
     *
     * @param instances
     * @param runsNum
     * @param seed
     * @param folds
     * @param percents
     * @param batchSize
     * @param alpha
     * @return
     * @throws Exception
     */
    public static Map<Integer, List<Double>> seuUniformSamplingGetLerningCurve(Instances instances,
                                                                               int runsNum,
                                                                               int seed,
                                                                               int folds,
                                                                               double percents,
                                                                               int batchSize,
                                                                               int alpha) throws Exception {
        // todo
        int itersNum = 1;
        double nowTime = System.currentTimeMillis();

        Set<Integer> numericAttrsIndexes = getNumericAttrsIndexes(instances);

        Discretize discretizer = new Discretize();
        discretizer.setInputFormat(instances);

        int allQueriesNum = (int) ((double) (instances.numInstances() * (instances.numAttributes() - 1)) * ((double) (folds - 1) / folds));
        Map<Integer, List<Double>> numToAccByRuns = new LinkedHashMap<>();
        List<Pair<Integer, List<Double>>> numToAcc = new ArrayList<>();
        for (int i = 0; i < runsNum; ++i) {
            // Randomize data
            Random rand = new Random(seed + i);
            Instances randData = new Instances(instances);   // create copy of original data
            randData.randomize(rand);
            randData.stratify(folds);

            for (int j = 0; j < folds; ++j) {
                Instances train = randData.trainCV(folds, j); // todo: not always same size
                Instances test = randData.testCV(folds, j);

                // Add missing
                Instances trainMissing = DatasetFactory.makeWithMissingAttrsUniformly(train, percents);
                System.out.println(String.format("runNum = %s, foldNum = %s", i, j));

                QueryManager queryManager = new SimpleQueryManager(train);
                SEUUniformSampling seuMethod = new SEUUniformSampling(trainMissing,
                        queryManager,
                        alpha,
                        batchSize,
                        discretizer,
                        numericAttrsIndexes);

                getLearningCurveHelper(numToAcc, seuMethod, test, itersNum, folds, j, true);

                System.out.println("diff time = " + ((System.currentTimeMillis() - nowTime) / 1000));
                nowTime = System.currentTimeMillis();
            }

            collectInfoFromRun(numToAcc, numToAccByRuns, allQueriesNum);
            numToAcc.clear();
        }

        return numToAccByRuns;
    }

    /**
     * SeuErrorSamplingGetLerningCurve
     *
     * @param instances
     * @param runsNum
     * @param seed
     * @param folds
     * @param percents
     * @param batchSize
     * @param esParam
     * @return
     * @throws Exception
     */
    public static Map<Integer, List<Double>> seuErrorSamplingGetLerningCurve(Instances instances,
                                                                         int runsNum,
                                                                         int seed,
                                                                         int folds,
                                                                         double percents,
                                                                         int batchSize,
                                                                         int esParam) throws Exception {

        int itersNum = 1;
        double nowTime = System.currentTimeMillis();

        Set<Integer> numericAttrsIndexes = DatasetFactory.getNumericAttrsIndexes(instances);

        Discretize discretizer = new Discretize();
        discretizer.setInputFormat(instances);

        int allQueriesNum = (int) ((double) (instances.numInstances() * (instances.numAttributes() - 1)) * ((double) (folds - 1) / folds));
        Map<Integer, List<Double>> numToAccByRuns = new LinkedHashMap<>();
        List<Pair<Integer, List<Double>>> numToAcc = new ArrayList<>();
        for (int i = 0; i < runsNum; ++i) {
            // Randomize data
            Random rand = new Random(seed + i);
            Instances randData = new Instances(instances);   // create copy of original data
            randData.randomize(rand);
            randData.stratify(folds);

            for (int j = 0; j < folds; ++j) {
                Instances train = randData.trainCV(folds, j); // todo: not always same size
                Instances test = randData.testCV(folds, j);

                // Add missing
                Instances trainMissing = DatasetFactory.makeWithMissingAttrsUniformly(train, percents);
                System.out.println(String.format("runNum = %s, foldNum = %s", i, j));

                QueryManager queryManager = new SimpleQueryManager(train);
                SEUErrorSampling seuMethod = new SEUErrorSampling(trainMissing,
                        queryManager,
                        esParam,
                        batchSize,
                        discretizer,
                        numericAttrsIndexes);

                getLearningCurveHelper(numToAcc, seuMethod, test, itersNum, folds, j, true);

                System.out.println("diff time = " + ((System.currentTimeMillis() - nowTime) / 1000));
                nowTime = System.currentTimeMillis();
            }

            collectInfoFromRun(numToAcc, numToAccByRuns, allQueriesNum);
            numToAcc.clear();
        }

        return numToAccByRuns;
    }

    /**
     *
     * @param numToAcc
     * @param seuMethod
     * @param testDataset
     * @param itersNum
     * @param folds
     * @param curFold
     * @param enableLog
     * @throws Exception
     */
    private static void getLearningCurveHelper(List<Pair<Integer, List<Double>>> numToAcc,
                                               AFAMethod seuMethod,
                                               Instances testDataset,
                                               int itersNum,
                                               int folds,
                                               int curFold,
                                               boolean enableLog) throws Exception {
        int cnt = 0; // counter for numToAcc

        // Zero step
        int possibleQueriesNum = seuMethod.getRealPossibleQueries();
        if (enableLog) {
            System.out.print("possibleQueriesNum = " + possibleQueriesNum);
        }

        int num = seuMethod.getAllQueriesNum() - possibleQueriesNum;
        J48 initialCls = seuMethod.makeClassifier();
        double acc = DatasetFactory.calculateAccuracy(initialCls, testDataset);
        if (numToAcc.size() <= cnt) { // todo
            List<Double> accs = new ArrayList<>();
            accs.add(acc);
            numToAcc.add(new Pair<>(num, accs));
        } else {
            numToAcc.get(cnt).second.add(acc);
        }
        ++cnt;


        List<Pair<List<Pair<Integer, Integer>>, J48>> res = seuMethod.perform(itersNum);
        J48 lastCls = new J48();
        while (res.size() > 0) {
            num = num + res.get(0).first.size();
            acc = DatasetFactory.calculateAccuracy(res.get(0).second, testDataset);
            if (numToAcc.size() <= cnt) { // todo
                List<Double> accs = new ArrayList<>();
                accs.add(acc);
                numToAcc.add(new Pair<>(num, accs));
            } else {
                numToAcc.get(cnt).second.add(acc);
            }
            ++cnt;

            lastCls = res.get(0).second; // todo

            possibleQueriesNum -= res.get(0).first.size();
            if (enableLog) {
                System.out.print(" --> " + possibleQueriesNum);
            }
            res = seuMethod.perform(itersNum);
        }

        if (enableLog) {
            System.out.println(" --end--> 0");
            if (curFold == folds - 1) { // todo
                J48 cls = lastCls;
                System.out.println("Super test of last classifier:");
                System.out.println("Acccuracy = " + DatasetFactory.calculateAccuracy(cls, testDataset));
                System.out.println("_____");
            }
        }
    }

    /**
     * todo
     *
     * @param numToAcc
     * @param numToAccByRuns
     * @param allQueriesNum
     */
    private static void collectInfoFromRun(List<Pair<Integer, List<Double>>> numToAcc,
                                           Map<Integer, List<Double>> numToAccByRuns,
                                           int allQueriesNum) {
        for (Pair<Integer, List<Double>> pair : numToAcc) {
            double avgAcc = pair.second.stream().mapToDouble(val -> val).average().getAsDouble();
            int key = (int) (Math.round(100.0 * ((double) pair.first / (double) allQueriesNum)));
            if (numToAccByRuns.containsKey(key)) {
                numToAccByRuns.get(key).add(avgAcc);
            } else {
                List<Double> accuracies = new LinkedList<>();
                accuracies.add(avgAcc);
                numToAccByRuns.put(key, accuracies);
            }
        }
    }


    /**
     * Return indexes of attributes which are numeric
     *
     * @param instances
     * @return
     */
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
