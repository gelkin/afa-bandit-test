package ru.ifmo.ctddev.mazin.AFA;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;

public class Dataset {
    private String name;
    private File file;
    private String taskType;

    public Dataset(String name, File file, String taskType){
        this.file = file;
        this.name = name;
        this.taskType = taskType;
    }

    public String getName() {
        return name;
    }

    public Instances getInstances() {
        try {
            ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(file.getPath());

            Instances instances = dataSource.getDataSet();

//            Filter filter = new Normalize();
//            filter.setInputFormat(instances);
//            instances = Filter.useFilter(instances, filter);

            return instances;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }

    public File getFile() {
        return file;
    }

    public String getTaskType() {
        return taskType;
    }
}

