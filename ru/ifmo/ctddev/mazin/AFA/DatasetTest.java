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
    private static final String ALL_RUNS = "ALL_RUNS";

    private static final double PERCENTS = 0.5;
    private static ArrayList<Dataset> dataSets;

    public static void main(String[] args) {
        DatasetTest test = new DatasetTest();
        test.readDataset();
//        test.testInstancesFromDataset();
        try {
            int datasetIndex = 3;
            Instances instances = dataSets.get(datasetIndex).getInstances();
            String datasetName = dataSets.get(datasetIndex).getName();
            instances.setClassIndex(instances.numAttributes() - 1);
//            if (!test.coolDiscretizeTest(instances)) {
//                System.out.println("Oh shi...");
//            }
//            test.printAccuracyDifference(instances, datasetName);

//            Instances instances = test.deleteInstancesWithMissing(instances);
//            for (int i = 0; i < dataSets.size(); ++i) {
//                Instances instances = dataSets.get(i).getInstances();
//                String datasetName = dataSets.get(i).getName();
//                instances.setClassIndex(instances.numAttributes() - 1);
//
//                test.printAccuracyDifference(instances, datasetName);
//
////                Pair<Map<Integer, Double>, String> methodResult = test.massiveTest(instances, datasetName);
////                String filename = test.writeMethodResult(methodResult, datasetName);
//            }
	    //
            Pair<Map<Integer, List<Double>>, String> methodResult = test.massiveTest(instances, datasetName);
            String filename = test.writeMethodResult(methodResult, datasetName);
	    //
//            String filename4 = RES_PATH + "remote-tests/" + "car_SEU-ESparam=172-runs=1-folds=10-8597.csv";
//            Map<Integer, Double> part1 = test.readMethodResult(filename4);
//            String filename5 = RES_PATH + "remote-tests/" + "car_SEU-ESparam=172-runs=1-folds=10-first.csv";
//            Map<Integer, Double> part2 = test.readMethodResult(filename5);
//            String filename6 = RES_PATH + "remote-tests/" + "car_SEU-ESparam=172-runs=2-folds=10-1959.csv";
//            Map<Integer, Double> part3 = test.readMethodResult(filename6);
//
//            for (Map.Entry<Integer, Double> entry : part3.entrySet()) {
//                double value = (entry.getValue() + part2.get(entry.getKey()) + part1.get(entry.getKey())) / 3.0;
//                part1.put(entry.getKey(), value);
//            }
//            Pair<Map<Integer, Double>, String> methodResultTest1 = new Pair(part1, "SEU-ES");


//            String filename1 = RES_PATH + "remote-tests/" + "vowel_SEU-ESparam=99-runs=6-folds=10-7652.csv";
//            Pair<Map<Integer, Double>, String> methodResultTest1 = new Pair(test.readMethodResult(filename1), "SEU-ES");
//
//            String filename2 = RES_PATH + "afaBandit_cool_datasets/" + "vowel_AFABandit-runs=10-folds=10.csv";
//            Pair<Map<Integer, Double>, String> methodResultTest2 = new Pair(test.readMethodResult(filename2), "AFABandit");
//            List<Pair<Map<Integer, Double>, String>> toPlot = new ArrayList<>();
//            toPlot.add(methodResultTest1);
//            toPlot.add(methodResultTest2);
//            test.plotMethodResults(toPlot, datasetName);


//            test.seuIniformSamplingTest(instances);
//            test.seuErrorSamplingTest(instances);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public boolean coolDiscretizeTest(Instances instances) throws Exception {
        Instances testInstances = new Instances(instances);

        Discretize discretizer = new Discretize();
        discretizer.setInputFormat(testInstances);
        Instances discInstances = Filter.useFilter(testInstances, discretizer);

        QueryManager queryManager = new SimpleQueryManager(discInstances);
        Instances testMissing = DatasetFactory.makeWithMissingAttrsUniformly(discInstances, PERCENTS);

        for (int i = 0; i < testMissing.numInstances(); ++i) {
            for (int j = 0; j < testMissing.numAttributes(); ++j) {
                if (testMissing.instance(i).isMissing(j)) {
                    SEUErrorSampling.acquireQuery(queryManager, testMissing, i, j);
                }
            }
        }

        for (int i = 0; i < discInstances.numInstances(); ++i) {
            for (int j = 0; j < discInstances.numAttributes(); ++j) {
                if (discInstances.instance(i).value(j) != testMissing.instance(i).value(j)) {
                    System.out.println("instIndex = " + i + ", attrIndex = " + j);
                    return false;
                }
            }
        }
        return true;
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

    public Instances deleteInstancesWithMissing(Instances instances) {
        Instances res = new Instances(instances, instances.numInstances());
        for (int i = 0; i < instances.numInstances(); ++i) {
            Instance inst = instances.instance(i);
            if (!inst.hasMissingValue()) {
                res.add(inst);
            }
        }
        res.setClassIndex(res.numAttributes() - 1);
        return res;
    }

    public void printAccuracyDifference(Instances instances, String datasetName) throws Exception {
        System.out.println(datasetName + ":");
        System.out.println("Instances number: " + instances.numInstances());
        System.out.println("Attributes number: " + instances.numAttributes());
        System.out.println("Classes number: " + instances.classAttribute().numValues());

        int seed = 137;

        int runsNum = 10;
        int folds = 10;
        double percents = PERCENTS;

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
        double percents = PERCENTS;
        int batchSize = 30;
        int iterationsNumber = 1;

        int alpha = 20;

        Map<Integer, List<Double>> numToAccMap = DatasetFactory.seuIniformSamplingGetLerningCurve(instances, runsNum, seed, folds, percents, batchSize, alpha);

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
                    train = DatasetFactory.makeWithMissingAttrsUniformly(train, percents);
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

    public String writeMethodResult(Pair<Map<Integer, List<Double>>, String> methodResult, String datasetName) throws IOException {
        String filename = RES_PATH + datasetName + "_" + methodResult.second + ".csv";
        FileWriter fw = new FileWriter(filename);
        for (Map.Entry<Integer, List<Double>> entry : methodResult.first.entrySet()) {
            StringBuilder sb = new StringBuilder(entry.getKey());
            sb.append(entry.getKey());
            for (Double val : entry.getValue()) {
                sb.append("," + val);
            }
            sb.append("\n");
            fw.write(sb.toString());
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

    public Pair<Map<Integer, List<Double>>, String> massiveTest(Instances instances, String datasetName) throws Exception {
        int seed = 137;

        int runsNum = 6;
        int folds = 10;
        double percents = PERCENTS;
        double coef = (folds - 1) / (double) folds;
        int batchSize = (int) (coef * (instances.numInstances()) * (instances.numAttributes() - 1) / 100.0); // todo 1/50 of possible queries

        Random r = new Random(System.currentTimeMillis());
        int suffixNum = r.nextInt(10000);
        // afaBandit
        /*
        Map<Integer, List<Double>> numToAccMapBandit =
                DatasetFactory.afaBanditGetLerningCurve(instances, runsNum, seed, folds, percents, batchSize);
        return new Pair<>(numToAccMapBandit, String.format("AFABandit-runs=%s-folds=%s-%s-%s", runsNum, folds, ALL_RUNS, suffixNum));
        */
        // seu uniform sampling
        //
        // int alpha = (instances.numInstances()) * (instances.numAttributes() - 1) / batchSize; // todo full
        int alpha = 10; // todo
        Map<Integer, List<Double>> numToAccMapSEU =
                DatasetFactory.seuIniformSamplingGetLerningCurve(instances, runsNum, seed, folds, PERCENTS, batchSize, alpha);
        return new Pair(numToAccMapSEU, String.format("SEU-USalpha=%s-runs=%s-folds=%s-%s-%s", alpha, runsNum, folds, ALL_RUNS, suffixNum));
        //
        // seu error sampling
        /*
        int euParam = instances.numInstances() / 10; // todo
        Map<Integer, List<Double>> numToAccMapSEU =
                DatasetFactory.seuErrorSamplingGetLerningCurve(instances, runsNum, seed, folds, percents, batchSize, euParam);
        return new Pair(numToAccMapSEU, String.format("SEU-ESparam=%s-runs=%s-folds=%s-%s-%s", euParam, runsNum, folds, ALL_RUNS, suffixNum));
        */
    }

    public void plotMethodResults(List<Pair<Map<Integer, Double>, String>> methodResults, String datasetName) {
        LineChart.run(methodResults, datasetName, "% заполненных ячеек", "Точность (%)");
    }

    public void afaBanditTest(Instances instances) throws Exception {
        int seed = 137;

        int runsNum = 4;
        int folds = 8;
        double percents = PERCENTS;
        int batchSize = (instances.numInstances()) * (instances.numAttributes() - 1) / 100; // todo 1/50 of possible queries

        Map<Integer, List<Double>> numToAccMap = DatasetFactory.afaBanditGetLerningCurve(instances, runsNum, seed, folds, percents, batchSize);
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
            Instances testInstances = DatasetFactory.makeWithMissingAttrsForEachInstUniformly(instances, perc);
            J48 testCls = new J48();
            testCls.buildClassifier(testInstances);
            new WekaJ48().vizualize(testCls);

//            testCls.setUnpruned(true);
//            testCls.makeClassifier(testInstances);
//            new WekaJ48().vizualize(testCls);
//
//            int x = 0;
        }

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
        File folder = new File(System.getProperty("user.dir") + "/res/afaBandit_cool_datasets");

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
