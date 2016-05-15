#!/usr/bin/env zsh

javac -classpath lib/weka-3.7.0.jar:lib/jfreechart-1.0.13.jar:lib/jcommon-1.0.16.jar\
	  -d out\
	  ru/ifmo/ctddev/mazin/AFA/*.java\
