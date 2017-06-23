package meddle;

import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class TrainingData {
    // feature_name:count, for freqency of word in a specific domain
    public Map<String, Integer> wordCount;
    public ArrayList<Map<String, Integer>> trainMatrix;
    public ArrayList<Integer> piiLabels;
    public Instances trainingInstances;

    public MetaEvaluationMeasures metaEvaluationMeasures;

    public TrainingData(){
        wordCount = new HashMap<String, Integer>();
        trainMatrix = new ArrayList<Map<String, Integer>>();
        piiLabels = new ArrayList<Integer>();
        metaEvaluationMeasures = new MetaEvaluationMeasures();
    }

}
