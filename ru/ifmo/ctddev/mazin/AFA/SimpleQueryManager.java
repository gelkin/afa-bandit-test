package ru.ifmo.ctddev.mazin.AFA;

import weka.core.Instances;
import weka.core.Attribute;

public class SimpleQueryManager implements QueryManager {
    private Instances instances;

    public SimpleQueryManager(Instances instances) {
        this.instances = instances;
    }

    @Override
    public double getValue(int instanceIndex, int attributeIndex) {
        return instances.instance(instanceIndex).value(attributeIndex);
    }
}
