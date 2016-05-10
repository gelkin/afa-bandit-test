package ru.ifmo.ctddev.mazin.AFA;

import weka.core.Attribute;

public interface QueryManager {
    double getValue(int instanceIndex, int attributeIndex);
}
