package meddle;

import org.json.simple.JSONObject;

/**
 * Intermediate and final results during training a classifier for the domain,os.
 * */
public class MetaEvaluationMeasures {
    public double falsePositiveRate;
    public double falseNegativeRate;
    public double trainingTime;
    public double populatingTime;

    public int numTotal;
    public int numPositive;
    public int numNegative;
    public int numOfPossibleFeatures;

    public double AUC;
    public double fMeasure;
    public double numInstance;
    public int numCorrectlyClassified;
    public double accuracy; // = NumCorrectlyClassified / NumTotal
    public double[][] confusionMatrix;
    public double TP;
    public double TN;
    public double FP;
    public double FN;
    public Info info;

    public String recordForInitialTrain() {
        String str = "";
        str += this.info.domain + "\t";
        str += this.info.OS + "\t";
        str += String.format("%.4f", this.accuracy) + "\t";
        str += String.format("%.4f", this.falsePositiveRate) + "\t";
        str += String.format("%.4f", this.falseNegativeRate) + "\t";
        str += String.format("%.4f", this.AUC) + "\t";
        str += String.format("%.4f", this.trainingTime) + "\t";

        str += this.numPositive + "\t";
        str += this.numNegative + "\t";
        str += this.numTotal + "\t";

        str += this.info.initNumPos + "\t"; // # positive samples initially
        str += this.info.initNumNeg + "\t";
        str += this.info.initNumTotal + "\t";

        return str;
    }

    @SuppressWarnings("unchecked")
    public String recordJSONFormat() {
        String str = "";
        JSONObject obj = new JSONObject();
        obj.put("domain", info.domain);
        obj.put("os", info.OS);
        obj.put("domain_os", info.domainOS);
        obj.put("json_file", info.fileNameRelative);
        obj.put("accuracy", this.accuracy);
        obj.put("fpr", falsePositiveRate);
        obj.put("fnr", falseNegativeRate);
        obj.put("auc", AUC);
        obj.put("traing_time", trainingTime);
        obj.put("populating_time", populatingTime);

        obj.put("init_num_pos", info.initNumPos);
        obj.put("init_num_neg", info.initNumNeg);
        obj.put("init_num_total", info.initNumTotal);

        obj.put("num_pos", this.numPositive);
        obj.put("num_neg", this.numNegative);
        obj.put("num_total", this.numTotal);

        str = obj.toJSONString() + "";
        System.out.println(str);
        return str;
    }



}
