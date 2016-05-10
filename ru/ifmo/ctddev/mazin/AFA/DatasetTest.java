package ru.ifmo.ctddev.mazin.AFA;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.io.*;
import java.text.DecimalFormat;
import java.util.*;

public class DatasetTest {
    private static final String RES_PATH = System.getProperty("user.dir") + "/res/results/";

    private static ArrayList<Dataset> dataSets;

    public static void main(String[] args) {
        DatasetTest test = new DatasetTest();
        test.readDataset();
//        test.testInstancesFromDataset();
        try {
            // test.analyzeDataset();
            for (int i = 2; i < 3; ++i) {
                Instances instances = dataSets.get(i).getInstances();
                String datasetName = dataSets.get(i).getName();
                instances.setClassIndex(instances.numAttributes() - 1);

                Pair<Map<Integer, Double>, String> methodResult = test.massiveTest(instances, datasetName);
                String filename = test.writeMethodResult(methodResult, datasetName);
            }

//            Pair<Map<Integer, Double>, String> methodResultTest = new Pair(test.readMethodResult(filename), "AFABandit");
//            List<Pair<Map<Integer, Double>, String>> toPlot = new ArrayList<>();
//            toPlot.add(methodResult);
//            test.plotMethodResults(toPlot, datasetName);


//            test.seuIniformSamplingTest(instances);
//            test.seuErrorSamplingTest(instances);
        } catch (Exception e) {
            e.printStackTrace();
        }
//
//        try {
//            test.plotTest(carInstances);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//        double percents = 0.5;
//        Instances carInstancesWithMissing = DatasetFactory.getInstancesWithMissingAttrs(carInstances, percents);
//        QueryManager queryManager = new SimpleQueryManager(carInstances);
//        AFABandit afaMethod = new AFABandit(carInstancesWithMissing, queryManager);


        try {
//            Instances instances = dataSets.get(0).getInstances();
//            instances.setClassIndex(instances.numAttributes() - 1);
//            new DatasetTest().J48Test(instances);
//            List<Pair<List<Pair<Integer, nteger>>, J48>> res = afaMethod.perform(5, 10);
//            int x = 0;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void analyzeDataset() throws Exception {
        for (int i = 0; i < dataSets.size(); ++i) {
            Instances instances = dataSets.get(i).getInstances();
            instances.setClassIndex(instances.numAttributes() - 1);
//            instances.setClassIndex(0);
            String datasetName = dataSets.get(i).getName();
            printAccuracyDifference(instances, datasetName);
        }
    }

    public void printAccuracyDifference(Instances instances, String datasetName) throws Exception {
        System.out.println(datasetName + ":");
        System.out.println("Instances number: " + instances.numInstances());
        System.out.println("Attributes number: " + instances.numAttributes());
        System.out.println("Classes number: " + instances.classAttribute().numValues());

        int seed = 137;

        int runsNum = 10;
        int folds = 10;
        double percents = 0.5;

        DecimalFormat df = new DecimalFormat("#.####");

        System.out.println("full instances: ");
        double fullAcc = afaBanditArbitraryTest(instances, seed, runsNum, folds, false, percents);
        fullAcc = Double.valueOf(df.format(fullAcc));
        System.out.println("Averaged accuracy: " + fullAcc);

        System.out.println("instances with missing:");
        double notFullAcc = afaBanditArbitraryTest(instances, seed, runsNum, folds, true, percents);
        notFullAcc = Double.valueOf(df.format(notFullAcc));
        System.out.println("Averaged accuracy: " + notFullAcc);


        double diff = fullAcc - notFullAcc;
        diff = Double.valueOf(df.format(diff));
        System.out.println("Diff: " + diff + "\n");
    }

    public void discretizeTest(Instances instances) throws Exception {
        Discretize filter = new Discretize();
        filter.setInputFormat(instances);

        Instances output;
        output = Filter.useFilter(instances, filter);

        Attribute attr = output.attribute(0);
        double[] cutPoints = filter.getCutPoints(0);
//        Instances instancesEmpty = new Instances(instances, 10);
    }

    public void seuIniformSamplingTest(Instances instances) throws Exception {
        int seed = 137;
        int runsNum = 1;
        int folds = 8;
        double percents = 0.5;
        int batchSize = 30;
        int iterationsNumber = 1;

        int alpha = 20;

        Map<Integer, Double> numToAccMap = DatasetFactory.seuIniformSamplingGetLerningCurve(instances, runsNum, seed, folds, percents, batchSize, alpha);

//        LineChart.run(numToAccMap, "SEU", "% of filled queries", "Accuracy on test set");
    }

    public void seuErrorSamplingTest(Instances instances) throws Exception {
        Set<Integer> numericAttrsIndexes = DatasetFactory.getNumericAttrsIndexes(instances);

        Discretize discretizer = new Discretize();
        discretizer.setInputFormat(instances);
        Instances discInstances = Filter.useFilter(instances, discretizer);

        int seed = 137;
        int runsNum = 1;
        int folds = 6;
        double percents = 0.5;
        int batchSize = 10;
        int iterationsNumber = 1;

        int l = 20;

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
                SEUErrorSampling seuMethod = new SEUErrorSampling(trainMissing,
                        queryManager,
                        l,
                        batchSize,
                        discretizer,
                        numericAttrsIndexes);

                // Zero step
                J48 initialCls = seuMethod.makeClassifier();
                int num = seuMethod.getAllQueriesNum() - seuMethod.getPossibleQueriesNum(seuMethod.getPossibleQueries(initialCls));
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

                    res = seuMethod.perform(iterationsNumber);
                }
            }
        }

        int allQueriesNum = instances.numInstances() * (instances.numAttributes() - 1);
        Map<Integer, Double> numToAccMap = new LinkedHashMap<>();
        for (Pair<Integer, List<Double>> pair : numToAcc) {
            double avgAcc = pair.second.stream().mapToDouble(val -> val).average().getAsDouble();
            numToAccMap.put((int) (Math.round(100.0 * ((double) pair.first / (double) allQueriesNum))), avgAcc);
        }
        int x = 0;

//        LineChart.run(numToAccMap, "SEU", "% of filled queries", "Accuracy on test set");
    }

    public double afaBanditArbitraryTest(Instances instances,
                                      int seed,
                                      int runsNum,
                                      int folds,
                                      boolean isWithMissing,
                                      double percents) throws Exception {
        double acc = 0.0;
        for (int i = 0; i < runsNum; ++i) {
            Random rand = new Random(seed + i);
            Instances randData = new Instances(instances);   // create copy of original data
            randData.randomize(rand);
            randData.stratify(folds);

            for (int j = 0; j < folds; ++j) {
                Instances train = randData.trainCV(folds, j); // todo: not always same size
                Instances test = randData.testCV(folds, j);

                if (isWithMissing) {
                    train = DatasetFactory.getInstancesWithMissingAttrs(train, percents);
                }

                J48 classifier = new J48();
                classifier.setUseLaplace(true);
                classifier.buildClassifier(train);

                acc += DatasetFactory.calculateAccuracy(classifier, test);
            }
        }

        return (acc / (folds * runsNum));
//        System.out.println("Averaged accuracy: " + (acc / (folds * runsNum)));
    }

    public String writeMethodResult(Pair<Map<Integer, Double>, String> methodResult, String datasetName) throws IOException {
        String filename = RES_PATH + datasetName + "_" + methodResult.second + ".csv";
        FileWriter fw = new FileWriter(filename);
        for (Map.Entry<Integer, Double> entry : methodResult.first.entrySet()) {
            fw.write(entry.getKey() + "," + entry.getValue() + "\n");
        }
        fw.close();

        return filename;
    }

    public Map<Integer, Double> readMethodResult(String filename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));

        Map<Integer, Double> res = new LinkedHashMap<>();
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            res.put(Integer.parseInt(parts[0]), Double.parseDouble(parts[1]));
        }
        br.close();

        return res;
    }

    public Pair<Map<Integer, Double>, String> massiveTest(Instances instances, String datasetName) throws Exception {
        int seed = 137;

        int runsNum = 1;
        int folds = 6;
        double percents = 0.5;
        double coef = (folds - 1) / (double) folds;
        int batchSize = (int) (coef * (instances.numInstances()) * (instances.numAttributes() - 1) / 100.0); // todo 1/50 of possible queries

        // afaBandit
//        Map<Integer, Double> numToAccMapBandit = DatasetFactory.afaBanditGetLerningCurve(instances, runsNum, seed, folds, percents, batchSize);
//        return new Pair(numToAccMapBandit, "AFABandit");
        // seu
//        int alpha = (instances.numInstances()) * (instances.numAttributes() - 1) / batchSize; // todo full
        int alpha = 10; // todo
        Map<Integer, Double> numToAccMapSEU = DatasetFactory.seuIniformSamplingGetLerningCurve(instances, runsNum, seed, folds, percents, batchSize, alpha);
        return new Pair(numToAccMapSEU, "SEU-USalpha=" + alpha);

//        Map<Integer, Double> numToAccMapBanditSecond = DatasetFactory.afaBanditGetLerningCurve(instances, runsNum, seed, folds, percents, batchSize);

        // plot
//        List<Pair<Map<Integer, Double>, String>> datasetsToPlot = new ArrayList<>();
//         datasetsToPlot.add(new Pair(numToAccMapBandit, "AFABandit"));
//        datasetsToPlot.add(new Pair(numToAccMapSEU, "SEU"));
//        datasetsToPlot.add(new Pair(numToAccMapBanditSecond, "Second AFABandit__"));

//        plotMethodResults(datasetsToPlot, datasetName);
    }

    public void plotMethodResults(List<Pair<Map<Integer, Double>, String>> methodResults, String datasetName) {
        LineChart.run(methodResults, datasetName, "% of filled queries", "Accuracy on test set");
    }

    public void afaBanditTest(Instances instances) throws Exception {
        int seed = 137;

        int runsNum = 4;
        int folds = 8;
        double percents = 0.5;
        int batchSize = (instances.numInstances()) * (instances.numAttributes() - 1) / 100; // todo 1/50 of possible queries

        Map<Integer, Double> numToAccMap = DatasetFactory.afaBanditGetLerningCurve(instances, runsNum, seed, folds, percents, batchSize);
//        LineChart.run(numToAccMap, "AFABandit", "X", "Y");
    }

    public void J48Test(Instances instances) throws Exception {
        J48 classifier = new J48();
        classifier.buildClassifier(instances);
//        new WekaJ48().vizualize(classifier);
//        System.out.println("Full probs:");
//        for (int i = 10; i < 50; ++i) {
//            double[] probs = classifier.distributionForInstance(instances.instance(i + 13));
//            System.out.println(probs[0] + " " + probs[1] + " " + probs[2] + " " + probs[3]);
//            int x = 0;
//        }


        for (double perc = 0.4; perc < 0.5; perc = perc + 0.1) {
            Instances testInstances = DatasetFactory.getInstancesWithMissingAttrs(instances, perc);
            J48 testCls = new J48();
            testCls.buildClassifier(testInstances);
            new WekaJ48().vizualize(testCls);

//            testCls.setUnpruned(true);
//            testCls.makeClassifier(testInstances);
//            new WekaJ48().vizualize(testCls);
//
//            int x = 0;
        }


//        double percents = 0.5;
//        Instances instancesWithMissing = DatasetFactory.getInstancesWithMissingAttrs(instances, percents);
//        J48 classifierMissing = new J48();
////        classifier.setUseLaplace(true);
////        classifierMissing.setUnpruned(true);
//        classifierMissing.makeClassifier(instancesWithMissing);
//        new WekaJ48().vizualize(classifierMissing);
//        System.out.println("Missing probs:");
//        for (int i = 10; i < 50; ++i) {
//            double[] probs = classifierMissing.distributionForInstance(instances.instance(i + 13));
//            System.out.println(probs[0] + " " + probs[1] + " " + probs[2] + " " + probs[3]);
//        }
    }

    public static class FileFilter {

        public File[] finder( String dirName){
            File dir = new File(dirName);

            return dir.listFiles(new FilenameFilter() {
                public boolean accept(File dir, String filename)
                { return ConverterUtils.DataSource.isArff(filename); }
            } );

        }

    }

    public void readDataset() {
        dataSets = new ArrayList<>();

//        File folder = new File(System.getProperty("user.dir") + "/res/papers_datasets/all_datasets");
        File folder = new File(System.getProperty("user.dir") + "/res/afa_cool_datasets");

        FileFilter filter = new FileFilter();
        File[] listOfFiles = filter.finder(folder.getAbsolutePath());

        System.out.println("Working directory " + folder.getAbsolutePath());
        System.out.println("Arff files found: " + listOfFiles.length);

        for (File file : listOfFiles) {
            if (file.isFile()) {
                if (ConverterUtils.DataSource.isArff(file.getName())) {
                    System.out.println(file.getName());

                    Dataset newDataset = new Dataset(file.getName().substring(0, file.getName().lastIndexOf('.')), file, "classification");
                    dataSets.add(newDataset);
                    try {
//                        manager.addDataset(newDataset);
//                        dataSets.add(newDataset);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                }

            }
        }

    }

    public void testInstancesFromDataset() {
        Instances carInstances = dataSets.get(2).getInstances();
        carInstances.setClassIndex(carInstances.numAttributes() - 1);

        Instances carCopy = new Instances(carInstances);

        Instance inst = carInstances.instance(0);
        inst.setClassMissing();

        Instance copyInst = carCopy.instance(0);

        double nan = Instance.missingValue();

        System.out.println("Missing value: '" + nan + "'");
        System.out.println("Num attributes: '" + carInstances.numAttributes() + "'");
        int x = 0;
    }
}
