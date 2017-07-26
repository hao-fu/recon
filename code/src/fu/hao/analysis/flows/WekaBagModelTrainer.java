package fu.hao.analysis.flows;

import fu.hao.utils.LabelledDoc;
import fu.hao.utils.Log;
import fu.hao.utils.WekaUtils;
import javafx.util.Pair;
import meddle.JsonKeyDef;
import meddle.RString;
import meddle.TrainingData;
import org.json.simple.JSONObject;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.*;

import static fu.hao.utils.WekaUtils.getWordFilter;

/**
 * Created by hfu on 7/21/2017.
 */
public class WekaBagModelTrainer {
    public static void genDocsHelper(List<JSONObject> flows, List<LabelledDoc> docs) throws Exception {
        for (JSONObject flow : flows) {
            int label = (int) (long) flow.get(JsonKeyDef.F_KEY_LABEL);
            String line = "";
            // fields: uri, post_body, refererrer, headers+values,
            line += WekaUtils.removeStopWords((String) flow.get(JsonKeyDef.F_KEY_DOMAIN)) + " ";
            line += WekaUtils.removeStopWords((String) flow.get(JsonKeyDef.F_KEY_URI)) + " ";

            RString sf = new RString();
            sf.breakLineIntoWords(line);

            Map<String, Integer> words = sf.Words;
            StringBuilder doc = new StringBuilder();
            for (Map.Entry<String, Integer> entry : words.entrySet()) {
                String word_key = entry.getKey().trim();
                doc.append(word_key);
                doc.append(" ");
            }
            docs.add(new LabelledDoc(Integer.toString(label), doc.toString()));
        }
    }

    /**
     * Given positive lines and negative lines, generate the overall word_count
     * and trainMatrix.
     *
     * @author renjj
     * */
    public static List<LabelledDoc> genDocs(List<JSONObject> posflows, List<JSONObject> negFlows) throws Exception {
        List<LabelledDoc> docs = new ArrayList<>();
        genDocsHelper(posflows, docs);
        genDocsHelper(negFlows, docs);
        return docs;
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

        List<JSONObject> posJsons = WekaUtils.dir2jsons(posFlowDir);
        Log.msg(TAG, posJsons.size());
        List<JSONObject> negJsons = WekaUtils.dir2jsons(negFlowDir);
        Log.msg(TAG, negJsons.size());

        List<String> labels = new ArrayList<>();
        labels.add("0");
        labels.add("1");
        List<LabelledDoc> docs = genDocs(posJsons, negJsons);
        WekaUtils.preProcessingDocs(docs);
        Instances instances = WekaUtils.docs2Instances(docs, labels);
        StringToWordVector stringToWordVector = getWordFilter(instances, false);
        stringToWordVector.setWordsToKeep(100000);
        stringToWordVector.setLowerCaseTokens(true);
        stringToWordVector.setMinTermFreq(1);
        //stringToWordVector.setDoNotOperateOnPerClassBasis(false);
        Log.msg(TAG, "keeping " + stringToWordVector.getWordsToKeep());
        instances = Filter.useFilter(instances, stringToWordVector);

        Log.msg(TAG, "Attributes: " + instances.numAttributes());

        WekaUtils.featureFilterByPrefix(instances, docs, labels);
        String timeStamp = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss").format(new Date());
        WekaUtils.createArff(instances, negFlowDir.getParentFile().getAbsolutePath() + File.separator +
                negFlowDir.getParentFile().getName()+ "_" + timeStamp + ".arff");
        return instances;
    }

    public static final String TAG = "WekaBagModelTrainer";
    public static void main(String[] args) throws Exception {
        final long beforeRun = System.nanoTime();
        WekaBagModelTrainer.createArff( "C:\\Users\\hfu\\Documents\\flows\\CTU-13\\CTU-13-1\\1",
                "C:\\Users\\hfu\\Documents\\flows\\CTU-13\\CTU-13-1\\0");
        Log.msg(TAG, "Time to generate arff: " + (System.nanoTime() - beforeRun) / 1E9 + " seconds");
    }
}
