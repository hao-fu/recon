package fu.hao.analysis.flows;

import fu.hao.utils.Log;
import fu.hao.utils.Setting;
import meddle.*;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

import java.io.*;
import java.util.*;

/**
 * Created by hfu on 6/23/2017.
 */
public class ModelTrainer {
    public static final String TAG = "ModelTrainer";

    // Class label
    public final static int LABEL_POSITIVE = 1;
    public final static int LABEL_NEGATIVE = 0;

    public static List<JSONObject> dir2jsons(File jsonDir) {
        List<String> jsonFiles = new ArrayList<>();
        List<JSONObject> jsonObjects = new ArrayList<>();
        String[] dirFiles = jsonDir.list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return (name.endsWith(".json"));
            }
        });
        for (String s : dirFiles) {
            jsonFiles.add(s);
        }

        for (final String fileName : jsonFiles) {
            final long beforeRun = System.nanoTime();
            Log.msg(TAG, "Begin to analyze: " + fileName);
            JSONParser parser = new JSONParser();
            try {
                jsonObjects.add((JSONObject) parser.parse(new FileReader(
                        jsonDir.getAbsolutePath() + File.separator + fileName)));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (ParseException e) {
                e.printStackTrace();
            }
        }

        return jsonObjects;
    }

    /**
     * Given positive lines and negative lines, generate the overall word_count
     * and trainMatrix.
     *

     * @param trainingData
     *            - the original training data object, could be empty or
     *            prefilled with some customized entries
     * @author renjj
     * */
    public static TrainingData genTrainingMatrix(List<JSONObject> flows,
                                                      TrainingData trainingData) {
        ArrayList<Map<String, Integer>> trainMatrix = trainingData.trainMatrix;
        ArrayList<Integer> piiLabels = trainingData.piiLabels;
        Map<String, Integer> word_count = trainingData.wordCount;
        int numOfPossibleFeatures = word_count.size();
        for (JSONObject flow : flows) {
            int label = (int) (long) flow.get(JsonKeyDef.F_KEY_LABEL);
            String line = "";
            // fields: uri, post_body, refererrer, headers+values,
            line += flow.get(JsonKeyDef.F_KEY_URI) + "\t";
            line += flow.get(JsonKeyDef.F_KEY_POST_BODY) + "\t";
            line += flow.get(JsonKeyDef.F_KEY_REFERRER) + "\t";
            JSONObject headers = (JSONObject) flow.get(JsonKeyDef.F_KEY_HEADERS);
            for (Object h : headers.keySet()) {
                line += h + "=" + headers.get(h) + "\t";
            }
            line += flow.get(JsonKeyDef.F_KEY_DOMAIN) + "\t";
            RString sf = new RString();
            sf.breakLineIntoWords(line);
            Map<String, Integer> words = sf.Words;
            for (Map.Entry<String, Integer> entry : words.entrySet()) {
                String word_key = entry.getKey().trim();
                if (word_key.length() == 1) {
                    char c = word_key.toCharArray()[0];
                    if (!Character.isAlphabetic(c) && !Character.isDigit(c))
                        continue;
                }
                if (RString.isStopWord(word_key)
                        || RString.isAllNumeric(word_key))
                    continue;
                if (word_key.length() == 0)
                    continue;

                int frequency = entry.getValue();
                if (word_count.containsKey(word_key))
                    word_count.put(word_key,
                            frequency + word_count.get(word_key));
                else {
                    numOfPossibleFeatures++;
                    word_count.put(word_key, frequency);
                }
            }
            trainMatrix.add(words);
            piiLabels.add(label);
        }
        trainingData.wordCount = word_count;
        trainingData.trainMatrix = trainMatrix;
        trainingData.piiLabels = piiLabels;
        trainingData.metaEvaluationMeasures.numOfPossibleFeatures = numOfPossibleFeatures;
        return trainingData;
    }

    public static Instances populateArff(Map<String, Integer> wordCount,
                                         ArrayList<Map<String, Integer>> trainMatrix,
                                         ArrayList<Integer> PIILabels, int theta) {
//		System.out.println(info);
        // Mapping feature_name_index
        Map<String, Integer> fi = new HashMap<>();
        int index = 0;
        // Populate Features
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int high_freq = trainMatrix.size();

        if (high_freq - theta < 30)
            theta = 0;
        for (Map.Entry<String, Integer> entry : wordCount.entrySet()) {
            // filter low frequency word
            String currentWord = entry.getKey();
            int currentWordFreq = entry.getValue();
            if (currentWordFreq < theta) {
                //if (!SharedMem.wildKeys.get("android").containsKey(currentWord)
                  //      && !SharedMem.wildKeys.get("ios").containsKey(currentWord)
                    //    && !SharedMem.wildKeys.get("windows").containsKey(currentWord))
                    continue;
            }
            Attribute attribute = new Attribute(currentWord);
            attributes.add(attribute);
            fi.put(currentWord, index);
            index++;
        }

        ArrayList<String> classVals = new ArrayList<String>();
        classVals.add("" + LABEL_NEGATIVE);
        classVals.add("" + LABEL_POSITIVE);
        attributes.add(new Attribute("PIILabel", classVals));

        // Populate Data Points
        Iterator<Map<String, Integer>> all = trainMatrix.iterator();
        int count = 0;
        Instances trainingInstances = new Instances("Rel", attributes, 0);
        trainingInstances.setClassIndex(trainingInstances.numAttributes() - 1);
        while (all.hasNext()) {
            Map<String, Integer> dataMap = all.next();
            double[] instanceValue = new double[attributes.size()];
            for (int i = 0; i < attributes.size() - 1; i++) {
                instanceValue[i] = 0;
            }
            int label = PIILabels.get(count);
            instanceValue[attributes.size() - 1] = label;
            for (Map.Entry<String, Integer> entry : dataMap.entrySet()) {
                if (fi.containsKey(entry.getKey())) {
                    int i = fi.get(entry.getKey());
                    int val = entry.getValue();
                    instanceValue[i] = val;
                }
            }
            Instance data = new SparseInstance(1.0, instanceValue);
            trainingInstances.add(data);
            count++;
        }
        // Write into .arff file for persistence
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(
                    Setting.getOutputDirectory() + "test.arff"));
            bw.write(trainingInstances.toString());
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return trainingInstances;
    }


    /**
     * Load all labelled network flows from JSON files and train classifiers for
     * each domain_os.
     *
     * @param classifierName
     *            - support for J48, SGD, ...TODO: LIST ALL THAT SUPPORTED
     */
    public static void train(String classifierName,
                                       String posFlowDirPath,
                                       String negFlowDirPath) {
        File posFlowDir = new File(posFlowDirPath);
        File negFlowDir = new File(negFlowDirPath);

        if (!posFlowDir.isDirectory() || !negFlowDir.isDirectory()) {
            throw new RuntimeException("The postive or negavtive flow dir is incorrect.");
        }

        List<JSONObject> posJsons = dir2jsons(posFlowDir);
        Log.msg(TAG, posJsons.size());
        List<JSONObject> negJsons = dir2jsons(negFlowDir);
        Log.msg(TAG, negJsons.size());

        TrainingData trainingData = new TrainingData();
        trainingData = genTrainingMatrix(posJsons, trainingData);
        trainingData = genTrainingMatrix(negJsons, trainingData);

        Log.warn(TAG, populateArff(trainingData.wordCount, trainingData.trainMatrix,
                trainingData.piiLabels, 0));





    }

    public static void main(String[] args) {
        ModelTrainer.train("", "C:\\Users\\hfu\\PycharmProjects\\TrafficAnalysis\\CTU-13-5\\1",
                "C:\\Users\\hfu\\PycharmProjects\\TrafficAnalysis\\CTU-13-5\\0");
    }


}

