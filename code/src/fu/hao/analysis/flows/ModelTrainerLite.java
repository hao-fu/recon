package fu.hao.analysis.flows;

import fu.hao.utils.Log;
import meddle.*;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import weka.core.*;

import java.io.*;
import java.util.*;

/**
 * Given a set of Strings, return the feature value set
 * Created by hfu on 08/18/2017.
 */
public class ModelTrainerLite {
    public static final String TAG = "ModelTrainerLite";


    public static String removeStopWords(String text) throws Exception {
        String DELIMITERS = "_|\\.|,|\t|/|\\||\\*|!|#|&|\\?|\n|;|\\{|\\}|\\(|\\)| ";
        StringBuilder stringBuilder = new StringBuilder();
        Set<String> words = new HashSet<>();

        for (String word : text.split(DELIMITERS)) {
            words.addAll(Util.wordBreak(word));
        }
        for (String word : words) {
            Analyzer analyzer = new EnglishAnalyzer();
            TokenStream stream = analyzer.tokenStream(null, new StringReader(word));
            stream = new StopFilter(stream, StandardAnalyzer.STOP_WORDS_SET);

            CharTermAttribute cattr = stream.addAttribute(CharTermAttribute.class);
            stream.reset();

            while (stream.incrementToken()) {
                stringBuilder.append(cattr);
                stringBuilder.append(' ');
            }
            stream.end();
            stream.close();
        }
        return stringBuilder.toString();
    }

    /**
     * Given positive lines and negative lines, generate the overall word_count
     * and trainMatrix.
     * @param strings
     * @param trainingData
     *            - the original training data object, could be empty or
     *            prefilled with some customized entries
     * @author renjj, hfu
     * */
    public static TrainingData genTrainingMatrix(List<String> strings,
                                                 TrainingData trainingData) throws Exception {
        ArrayList<Map<String, Integer>> trainMatrix = trainingData.trainMatrix;
        //ArrayList<Integer> piiLabels = trainingData.piiLabels;
        Map<String, Integer> word_count = trainingData.wordCount;
        int numOfPossibleFeatures = word_count.size();
        for (String url : strings) {
            //int label = (int) (long) flow.get(JsonKeyDef.F_KEY_LABEL);
            String line = "";
            // fields: uri, post_body, refererrer, headers+values,
            line += removeStopWords(url);

            //line += flow.get(JsonKeyDef.F_KEY_POST_BODY) + "\t";
            //Log.msg(TAG, flow.get(JsonKeyDef.F_KEY_POST_BODY));
            //line += flow.get(JsonKeyDef.F_KEY_REFERRER) + "\t";
            //Log.msg(TAG, flow.get(JsonKeyDef.F_KEY_REFERRER));
            //JSONObject headers = (JSONObject) flow.get(JsonKeyDef.F_KEY_HEADERS);
            //for (Object h : headers.keySet()) {
            //  line += h + "=" + headers.get(h) + "\t";
            //}

            RString sf = new RString();
            sf.breakLineIntoWords(line);
            Log.msg(TAG, line);
            Log.msg(TAG, removeStopWords(line));

            Map<String, Integer> words = sf.Words;
            for (Map.Entry<String, Integer> entry : words.entrySet()) {
                String word_key = entry.getKey().trim();
                Log.bb(TAG, word_key);
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
            //piiLabels.add(label);
        }
        trainingData.wordCount = word_count;
        trainingData.trainMatrix = trainMatrix;
        // trainingData.piiLabels = piiLabels;
        trainingData.metaEvaluationMeasures.numOfPossibleFeatures = numOfPossibleFeatures;
        return trainingData;
    }

    public static InstancesLite getInstanceValues(Map<String, Integer> wordCount,
                                         ArrayList<Map<String, Integer>> trainMatrix,
                                         int theta) {
        InstancesLite instances = new InstancesLite();
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
                continue;
            }
            Attribute attribute = new Attribute(currentWord);
            attributes.add(attribute);
            fi.put(currentWord, index);
            index++;
        }
        instances.getFeatures().putAll(fi);

        // Populate Data Points
        Iterator<Map<String, Integer>> all = trainMatrix.iterator();
        while (all.hasNext()) {
            Map<String, Integer> dataMap = all.next();
            double[] instanceValue = new double[attributes.size()];
            for (int i = 0; i < attributes.size() - 1; i++) {
                instanceValue[i] = 0;
            }

            for (Map.Entry<String, Integer> entry : dataMap.entrySet()) {
                if (fi.containsKey(entry.getKey())) {
                    int i = fi.get(entry.getKey());
                    int val = entry.getValue();
                    instanceValue[i] = val;
                }
            }
            //Log.msg(TAG, instanceValue);
            instances.getValues().add(instanceValue);
        }

        return instances;
    }

    public static InstancesLite str2FeatureValues(List<String> strings) throws Exception {
        TrainingData trainingData = new TrainingData();
        trainingData = genTrainingMatrix(strings, trainingData);
        return  getInstanceValues(trainingData.wordCount, trainingData.trainMatrix, 0);
    }

    public static void main(String[] args) throws Exception {
        final long beforeRun = System.nanoTime();
        List<String> urls = new ArrayList<>();
        urls.add("/action/account/getinfo?app_id=393f0e2a54c8f7f9b9ab2c71b61b11bd&udid=351565054929465&imsi=310410");
        urls.add("https://www.zhihu.com/explore");
        InstancesLite instances = ModelTrainerLite.str2FeatureValues(urls);
        Log.msg(TAG, instances.getFeatures());
        for (double[] value : instances.getValues()) {
            StringBuilder stringBuilder = new StringBuilder();
            for (double subValue : value) {
                stringBuilder.append(subValue + ",");
            }
            Log.msg(TAG, stringBuilder.toString());
        }
        Log.msg(TAG, "Time to generate feature values: " + (System.nanoTime() - beforeRun) / 1E9 + " seconds");
    }


}

