package fu.hao.analysis.flows;

import cc.mallet.util.CommandOption;
import fu.hao.utils.Log;
import fu.hao.utils.WekaUtils;
import javafx.util.Pair;
import meddle.*;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.AttributeFactory;
import org.apache.lucene.util.Version;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import weka.core.*;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

import static fu.hao.utils.WekaUtils.buildClassifier;
import static fu.hao.utils.WekaUtils.getClassifier;

/**
 * Created by hfu on 6/23/2017.
 */
public class ModelTrainer {
    public static final String TAG = "ModelTrainer";

    // Class label
    public final static int LABEL_POSITIVE = 1;
    public final static int LABEL_NEGATIVE = 0;

    public static int MANY = 5000;

    public static List<JSONObject> dir2jsons(File jsonDir) {
        List<String> jsonFiles = new ArrayList<>();
        List<JSONObject> jsonObjects = new ArrayList<>();
        String[] dirFiles = jsonDir.list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return (name.endsWith(".json"));
            }
        });
        if (dirFiles != null) {
            for (String s : dirFiles) {
                jsonFiles.add(s);
            }
        }

        for (final String fileName : jsonFiles) {
            Log.bb(TAG, "Begin to analyze: " + fileName);
            JSONParser parser = new JSONParser();
            try {
                jsonObjects.add((JSONObject) parser.parse(new FileReader(
                        jsonDir.getAbsolutePath() + File.separator + fileName)));
            } catch (IOException | ParseException e) {
                e.printStackTrace();
            }
        }

        return jsonObjects;
    }

    /**
     * Give the number of positive samples and the number of negative samples,
     * find out the balancing values for both parties.
     */
    public static Pair<String, Float> balanceClassSamples(float numPos, float numNeg) {
        // FIXME the class numbers are reversed
        if (numPos > numNeg) {
            return new Pair<>("1", numPos / numNeg * 100 - 100);
        } else {
            return new Pair<>("0", numNeg / numPos * 100 - 100);
        }
    }

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
     *

     * @param trainingData
     *            - the original training data object, could be empty or
     *            prefilled with some customized entries
     * @author renjj
     * */
    public static TrainingData genTrainingMatrix(List<JSONObject> flows,
                                                      TrainingData trainingData) throws Exception {
        ArrayList<Map<String, Integer>> trainMatrix = trainingData.trainMatrix;
        ArrayList<Integer> piiLabels = trainingData.piiLabels;
        Map<String, Integer> word_count = trainingData.wordCount;
        int numOfPossibleFeatures = word_count.size();
        for (JSONObject flow : flows) {
            int label = (int) (long) flow.get(JsonKeyDef.F_KEY_LABEL);
            String line = "";
            // fields: uri, post_body, refererrer, headers+values,
            line += removeStopWords((String) flow.get(JsonKeyDef.F_KEY_DOMAIN)) + " ";
            line += removeStopWords((String) flow.get(JsonKeyDef.F_KEY_URI)) + " ";

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
            Log.bb(TAG, line);
            Log.bb(TAG, removeStopWords(line));

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
            //Log.msg(TAG, instanceValue);
            //Instance data = new DenseInstance(1.0, instanceValue);
            Instance data = new SparseInstance(1.0, instanceValue);
            trainingInstances.add(data);
            count++;
        }

        return trainingInstances;
    }

    public static Instances createArff(String posFlowDirPath,
                                  String negFlowDirPath) throws Exception {
        File posFlowDir = new File(posFlowDirPath);
        File negFlowDir = new File(negFlowDirPath);

        if (!posFlowDir.isDirectory() || !negFlowDir.isDirectory()) {
            Log.warn(TAG,"The postive or negavtive flow dir is incorrect.");
        }

        if (posFlowDir.getParentFile() == null || !posFlowDir.getParentFile().equals(negFlowDir.getParentFile())) {
            Log.warn(TAG, "The postive or negavtive flow dir is inconsistent");
        }

        List<JSONObject> posJsons = dir2jsons(posFlowDir);
        Log.msg(TAG, posJsons.size());
        List<JSONObject> negJsons = dir2jsons(negFlowDir);
        Log.msg(TAG, negJsons.size());

        TrainingData trainingData = new TrainingData();
        trainingData = genTrainingMatrix(posJsons, trainingData);
        trainingData = genTrainingMatrix(negJsons, trainingData);

        Instances instances =  populateArff(trainingData.wordCount, trainingData.trainMatrix,
                trainingData.piiLabels, 0);
        Pair<String, Float> balance = balanceClassSamples(posJsons.size(), negJsons.size());
        boolean test = true;
        if (!test && balance.getValue() > 25) {
            Log.msg(TAG, "Oversampling: " + balance);
            instances = WekaUtils.overSampling(instances, balance.getKey(), balance.getValue().intValue());
        }
        String timeStamp = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date());
        WekaUtils.write2Arff(instances, negFlowDir.getParentFile().getAbsolutePath() + File.separator +
                negFlowDir.getParentFile().getName()+ "_" + timeStamp + ".arff");
        return instances;
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
                                       String negFlowDirPath) throws Exception {
        Instances instances = createArff(posFlowDirPath, negFlowDirPath);
        buildClassifier(getClassifier(classifierName), instances, "CTU-13");
    }

    public static void main(String[] args) throws Exception {
        final long beforeRun = System.nanoTime();
        ModelTrainer.createArff("C:\\Users\\hfu\\Documents\\flows\\CTU-13\\CTU-13-1\\1",
                "C:\\Users\\hfu\\Documents\\flows\\CTU-13\\CTU-13-1\\0");
        Log.msg(TAG, "Time to generate arff: " + (System.nanoTime() - beforeRun) / 1E9 + " seconds");
    }


}

