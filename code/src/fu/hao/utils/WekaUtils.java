package fu.hao.utils;

import weka.attributeSelection.*;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.KStar;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.*;
import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.*;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.pmml.Array;
import weka.core.stemmers.SnowballStemmer;
//import weka.core.stopwords.WordsFromFile;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import weka.classifiers.Evaluation;

import javax.xml.parsers.FactoryConfigurationError;

/**
 * Description:
 *
 * @author Hao Fu(haofu@ucdavis.edu)
 * @since 3/1/2017
 */
public class WekaUtils {
    private class LabelledDoc {
        private String label;
        private String doc = null;

        LabelledDoc(String label, String doc) {
            this.label = label;
            doc = doc.replace(",", "");
            this.doc = doc;
        }

        public String getLabel() {
            return label;
        }

        public String getDoc() {
            return doc;
        }
    }

    public static Instances loadArff(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static Classifier getPureClassifier() {
        return new RandomForest();
    }

    public static FilteredClassifier buildClassifier(Instances data, boolean rmFirst) throws Exception {
        FilteredClassifier fc = new FilteredClassifier();
        if (rmFirst) {
            // filter
            Remove rm = new Remove();
            rm.setAttributeIndices("1");  // remove 1st attribute
            fc.setFilter(rm);
        }
        // classifier
        // J48 j48 = new J48();
        // j48.setUnpruned(true);        // using an unpruned J48

        //RandomForest classifier = new RandomForest();
        HoeffdingTree classifier = new HoeffdingTree();
        System.err.println("Parameters");
        for (int i = 0; i < classifier.getOptions().length; i++) {
            System.err.println(classifier.getOptions()[i]);
        }
        // meta-classifier
        fc.setClassifier(classifier); //j48);
        // train and make predictions
        fc.buildClassifier(data);

        weka.core.SerializationHelper.write("weka.model", fc);

        return fc;
    }


    private static AttributeSelection getAttributeSelector(
            Instances trainingData, int number) throws Exception {
        AttributeSelection selector = new AttributeSelection();
        InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(Math.min(number, trainingData.numAttributes() - 1));
        selector.setEvaluator(evaluator);
        selector.setSearch(ranker);
        selector.SelectAttributes(trainingData);
        return selector;
    }

    public static Classifier buildClassifier(Classifier classifier, Instances data, String modelName) throws Exception {
        classifier.buildClassifier(data);

        weka.core.SerializationHelper.write(modelName + ".model", classifier);

        return classifier;
    }


    public static Classifier buildClassifier(Instances data, String modelName,
                                             boolean updateable) throws Exception {


        // classifier
        // J48 j48 = new J48();
        // j48.setUnpruned(true);        // using an unpruned J48

        //RandomForest classifier = new RandomForest();
        //NaiveBayes classifier = new NaiveBayes();
        //HoeffdingTree classifier = new HoeffdingTree();
        Classifier classifier;
        if (updateable) {
            SGD sgd = new SGD();
            sgd.setLossFunction(new SelectedTag(SGD.LOGLOSS, SGD.TAGS_SELECTION));
            //sgd.setLossFunction(new SelectedTag(SGD.HINGE, SGD.TAGS_SELECTION));
            classifier = sgd; //new KStar(); //SGD(); //IBk(); //; //NaiveBayesMultinomial(); //LWL();//;//; //SGD(); //HoeffdingTree();
            //classifier = new HoeffdingTree();//NaiveBayesMultinomialUpdateable();
        } else {
            classifier = new SMO();
        }


        // System.err.println("Parameters");
        //for (int i = 0; i < classifier.getOptions().length; i++) {
        //  System.err.println(classifier.getOptions()[i]);
        //}
        // train and make predictions
        classifier.buildClassifier(data);

        weka.core.SerializationHelper.write(modelName + ".model", classifier);

        return classifier;
    }

    public static StringToWordVector getWordFilter(Instances input, boolean useIdf) throws Exception {
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(input);
        filter.setWordsToKeep(10000);
        if (useIdf) {
            filter.setIDFTransform(true);
        }
        //filter.setTFTransform(true);
        //filter.setLowerCaseTokens(true);
        //filter.setOutputWordCounts(false);

        //WordsFromFile stopwords = new WordsFromFile();
        //stopwords.setStopwords(new File("data/stopwords.txt"));
        //filter.setStopwordsHandler(stopwords);
        //SnowballStemmer stemmer = new SnowballStemmer();
        //filter.setStemmer(stemmer);

        return filter;
    }

    /**
     * Method: docs2Instances
     * Description:
     *
     * @param docs   Documents
     * @param labels Pre-defined labels
     * @return weka.core.Instances
     * @author Hao Fu(haofu AT ucdavis.edu)
     * @since 3/5/2017 2:24 PM
     */
    public static Instances docs2Instances(List<LabelledDoc> docs, List<String> labels) throws FileNotFoundException {
        ArrayList<Attribute> atts = new ArrayList<>();
        ArrayList<String> classVal = new ArrayList<>();
        for (String label : labels) {
            classVal.add(label);
        }


        Attribute attribute1 = new Attribute("text", (ArrayList<String>) null);
        Attribute attribute2 = new Attribute("text_label", classVal); // Do not use common words.txt for this attribute

        atts.add(attribute1);
        atts.add(attribute2);

        //build training data
        Instances data = new Instances("docs", atts, 1);
        DenseInstance instance;

        for (LabelledDoc labelledDoc : docs) {
            instance = new DenseInstance(2);
            instance.setValue((Attribute) atts.get(0), labelledDoc.getDoc());
            instance.setValue((Attribute) atts.get(1), labelledDoc.getLabel());
            data.add(instance);
        }
        data.setClassIndex(1);

        return data;
    }


    public static Instance doc2Instance(LabelledDoc doc, List<String> labels) throws Exception {
        ArrayList<LabelledDoc> docs = new ArrayList<>();
        docs.add(doc);

        return docs2Instances(docs, labels).get(0);
    }

    public static Instance genInstanceForUpdateable(LabelledDoc doc, List<String> labels, StringToWordVector stringToWordVector) throws Exception {
        ArrayList<LabelledDoc> docs = new ArrayList<>();
        docs.add(doc);
        return Filter.useFilter(docs2Instances(docs, labels), stringToWordVector).get(0);

    }

    public static Instances docs2Instances(List<String> docs) {
        ArrayList<Attribute> atts = new ArrayList<>();

        Attribute attribute1 = new Attribute("text", (ArrayList<String>) null);
        //Attribute attribute2 = new Attribute("text_label", classVal); // Do not use common words.txt for this attribute

        atts.add(attribute1);
        //atts.add(attribute2);

        //build training data
        Instances data = new Instances("docs", atts, 1);
        DenseInstance instance;

        for (String doc : docs) {
            instance = new DenseInstance(2);
            instance.setValue((Attribute) atts.get(0), doc);
//            instance.setValue((Attribute)atts.get(1), "?");
            data.add(instance);
        }
        //data.setClassIndex(data.numAttributes() - 1);

        return data;
    }

    public static String predict(String doc, StringToWordVector stringToWordVector, Classifier classifier, Attribute classAttribute)
            throws Exception {
        List<String> docs = new ArrayList<>();
        docs.add(doc);
        return predict(docs, stringToWordVector, classifier, classAttribute).get(0);
    }

    public static List<String> predict(List<String> docs, StringToWordVector stringToWordVector,
                                       Classifier classifier, Attribute classAttribute) throws Exception {
        Instances unlabelledInstances = docs2Instances(docs);
        unlabelledInstances = Filter.useFilter(unlabelledInstances, stringToWordVector);
        List<String> results = new ArrayList<>();
        for (Instance instance : unlabelledInstances) {
            Double clsLabel = classifier.classifyInstance(instance);

            if (classAttribute != null && classAttribute.numValues() > 0) {
                results.add(classAttribute.value(clsLabel.intValue()));
                System.out.println("Predicted: " + classAttribute.value(clsLabel.intValue()) + ", " + clsLabel);
            } else {
                results.add(clsLabel.toString());
                System.out.println("Predicted: " + clsLabel);
            }

            //get the predicted probabilities
            double[] prediction = classifier.distributionForInstance(instance);

            //output predictions
            for (int i = 0; i < prediction.length; i++) {
                System.out.println("Probability of class " + i +
                        classAttribute.value(i) +
                        " : " + Double.toString(prediction[i]));
            }

        }

        return results;
    }

    public static Instances createArff(Instances data, String filePath) throws Exception {
        //System.out.println("--------------------------------------------------");
        System.out.println("Create ARFF file:" + filePath);
        //System.out.println(data.toString());
        //System.out.println("--------------------------------------------------");
        //System.out.println(data.numAttributes());
        /*
        PrintWriter out = new PrintWriter("data.arff");
        out.print(data.toString());
        out.close();*/
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(filePath));
        //saver.setDestination(new File(filePath));   // **not** necessary in 3.5.4 and later
        saver.writeBatch();
        return data;
    }

    public static String readFile(String path, Charset encoding)
            throws IOException {
        byte[] encoded = Files.readAllBytes(Paths.get(path));
        return new String(encoded, encoding);
    }

    public List<LabelledDoc> getDocs(String docsDirPath) throws IOException {
        List<LabelledDoc> res = new ArrayList<>();
        File[] files = new File(docsDirPath).listFiles();
        List<File> allFiles = new ArrayList<>();
        showFiles(files, allFiles);

        for (File file : allFiles) {
            if (file.getName().endsWith("txt")) {
                if (file.getPath().contains("labelled_T")) {
                    res.add(new LabelledDoc("T", readFile(file.getPath(), StandardCharsets.UTF_8)));
                } else if ((file.getPath().contains("labelled_D"))) {
                    res.add(new LabelledDoc("D", readFile(file.getPath(), StandardCharsets.UTF_8)));
                } else if ((file.getPath().contains("labelled_F"))) {
                    res.add(new LabelledDoc("F", readFile(file.getPath(), StandardCharsets.UTF_8)));
                }
            }
        }

        return res;
    }

    public List<LabelledDoc> getUserDocs(String docsDirPath) throws IOException {
        List<LabelledDoc> res = new ArrayList<>();
        File[] files = new File(docsDirPath).listFiles();
        List<File> allFiles = new ArrayList<>();
        showFiles(files, allFiles);

        for (File file : allFiles) {
            if (file.getName().endsWith("txt")) {
                if (file.getPath().contains("Allow")) {
                    res.add(new LabelledDoc("T", readFile(file.getPath(), StandardCharsets.UTF_8)));
                } else if ((file.getPath().contains("Deny"))) {
                    res.add(new LabelledDoc("F", readFile(file.getPath(), StandardCharsets.UTF_8)));
                }
            }
        }

        return res;
    }

    public static void showFiles(File[] files, Collection<File> allFiles) {
        for (File file : files) {
            System.out.println(file.getName());
            if (file.isDirectory()) {
                showFiles(file.listFiles(), allFiles);
            } else {
                allFiles.add(file);
            }
        }
    }

    /**
     * Method: crossValidation
     * Description: Note that classifier should not be pre-trained
     *
     * @param data
     * @param classifier
     * @return void
     * @throw
     * @author Hao Fu(haofu AT ucdavis.edu)
     * @since 3/4/2017 4:13 PM
     */
    public static double crossValidation(Instances data, Classifier classifier, int fold) throws Exception {
        Evaluation eval = new Evaluation(data);
        System.out.println(eval.getHeader().numAttributes());
        eval.crossValidateModel(classifier, data, fold, new Random(10));
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
        return eval.weightedFMeasure();
    }

    public static String fixEncoding(String latin1) {
        try {
            byte[] bytes = latin1.getBytes("ISO-8859-1");
            if (!validUTF8(bytes))
                return latin1;
            return new String(bytes, "UTF-8");
        } catch (UnsupportedEncodingException e) {
            // Impossible, throw unchecked
            throw new IllegalStateException("No Latin1 or UTF-8: " + e.getMessage());
        }

    }

    public static boolean validUTF8(byte[] input) {
        int i = 0;
        // Check for BOM
        if (input.length >= 3 && (input[0] & 0xFF) == 0xEF
                && (input[1] & 0xFF) == 0xBB & (input[2] & 0xFF) == 0xBF) {
            i = 3;
        }

        int end;
        for (int j = input.length; i < j; ++i) {
            int octet = input[i];
            if ((octet & 0x80) == 0) {
                continue; // ASCII
            }

            // Check for UTF-8 leading byte
            if ((octet & 0xE0) == 0xC0) {
                end = i + 1;
            } else if ((octet & 0xF0) == 0xE0) {
                end = i + 2;
            } else if ((octet & 0xF8) == 0xF0) {
                end = i + 3;
            } else {
                // Java only supports BMP so 3 is max
                return false;
            }

            while (i < end) {
                i++;
                try {
                    octet = input[i];
                } catch (Exception e) {
                    e.printStackTrace();
                    return false;
                }
                if ((octet & 0xC0) != 0x80) {
                    // Not a valid trailing byte
                    return false;
                }
            }
        }
        return true;
    }

    public static void write2Arff(Instances instances, String outFilePath) {
        // Write into .arff file for persistence
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(
                    outFilePath));
            bw.write(instances.toString());
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void save2Arff(Instances instances, String fileName) throws IOException {
        // Save instances to arff
        instances.renameAttribute(0, "class");
        for (int i = 1; i < instances.numAttributes(); i++) {
            String name = fixEncoding(instances.attribute(i).name());
            try {
                instances.renameAttribute(i, name);
            } catch (IllegalArgumentException e) {
                instances.renameAttribute(i, "_" + name);
            }
        }
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        File dataFile = new File(fileName + ".arff");
        saver.setFile(dataFile);
        // saver.setDestination(dataFile);   // **not** necessary in 3.5.4 and later
        saver.writeBatch();
        for (Instance instance : instances) {
            instance.classAttribute();
            System.out.println(instance);
        }
    }

    public static FilteredClassifier loadClassifier(InputStream fileInputStream) throws Exception {
        FilteredClassifier filteredClassifier = null;

        filteredClassifier = (FilteredClassifier)
                weka.core.SerializationHelper.read(fileInputStream);

        return filteredClassifier;
    }

    public static Classifier loadClassifier(File file) throws Exception {
        FileInputStream fileInputStream = new FileInputStream(file);
        return (Classifier) SerializationHelper.read(fileInputStream);
    }

    public static StringToWordVector loadStr2WordVec(File file) throws Exception {
        FileInputStream fileInputStream = new FileInputStream(file);
        return loadStr2WordVec(fileInputStream);
    }

    public static StringToWordVector loadStr2WordVec(InputStream fileInputStream) throws Exception {
        return (StringToWordVector) SerializationHelper.read(fileInputStream);
    }

    public static Pair<Instances, Instances> splitInstances(Instances instances, float percent) {
        instances.randomize(new java.util.Random(0));
        int trainSize = (int) Math.round(instances.numInstances() * percent
                / 100);
        int testSize = instances.numInstances() - trainSize;
        Instances train = new Instances(instances, 0, trainSize);
        Instances test = new Instances(instances, trainSize, testSize);

        Pair<Instances, Instances> res = new Pair<>(train, test);
        return res;
    }

    public static Instances overSampling(Instances instances, String classValue, float percent) throws Exception {
        SMOTE smote = new SMOTE();
        smote.setClassValue(classValue);
        smote.setInputFormat(instances); // Instances instances;
        smote.setPercentage(percent);
        return Filter.useFilter(instances, smote);
    }

    public static Instances readArff(String filePath, int classIndex) throws Exception {
        DataSource source = new DataSource(filePath);
        Instances data = source.getDataSet();
        data.setClassIndex(classIndex);
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        //if (data.classIndex() == -1) {
        //data.setClassIndex(data.numAttributes() - 1);
        //}

        return data;
    }

    public static void eval1() throws Exception {
        boolean user = false;
        List<String> labels = new ArrayList<>();
        String type;
        WekaUtils wekaUtils = new WekaUtils();
        String mark = "Location";//RECORD_AUDI";//Camera"; //READ_PHONE_STATE";//SEND_SMS";//BLUETOOTH";//Location";//";//NFC"; //";//"; Camera"; //"; //"; //Location"; //; ";//Location"; //"; //"; //SEND_SMS";
        String pyWorkLoc = "D:\\workspace\\COSPOS_MINING";
        boolean smote = false;
        boolean attriSel = true;
        int attNum = 500;
        String smoteClass = "1";
        int smotePercent = 230;
        if (mark.startsWith("Camera")) {
            smote = true;
            smoteClass = "2";
            smotePercent = 130;
        } else if (mark.startsWith("REA")) {
            smote = true;
            smoteClass = "1";
            smotePercent = 230;
            attriSel = true;
            attNum = 100;
        } else if (mark.startsWith("SEN")) {
            smote = true;
            smoteClass = "1";
            smotePercent = 830;
            attriSel = true;
        } else if (mark.startsWith("Location")) {
            //smote = true;
            //smoteClass = "1";
            //smotePercent = 130;
            attriSel = false;
        } else if (mark.startsWith("REC")) {
            //smote = true;
            //smoteClass = "1";
            //smotePercent = 80;
            //attriSel = false;
            attNum = 500;
        }


        List<List<LabelledDoc>> docsResutls = new ArrayList<>();
        if (!user) {
            type = "full"; //Location"; //READ_PHONE_STATE";
            //Instances ata = WekaUtils.loadArff();
            //FilteredClassifier filteredClassifier = WekaUtils.buildClassifier(data);
            //System.out.println(filteredClassifier.getBatchSize());

            docsResutls.add(wekaUtils.getDocs(pyWorkLoc + "\\output\\gnd\\comp\\" + mark + "\\"
                    + type));// D:\\workspace\\COSPOS_MINING\\output\\gnd\\" + PERMISSION); //Location");
            labels.add("T");
            //labels.add("D");
            labels.add("F");
        } else {
            type = "users"; //READ_PHONE_STATE";
            for (int i = 0; i < 10; i++) {
                int user_num = i;
                mark = Integer.toString(user_num);
                docsResutls.add(wekaUtils.getUserDocs(pyWorkLoc + "\\output\\gnd\\" + type + "\\" + user_num)); //Location");
            }
            labels = new ArrayList<>();
            labels.add("T");
            labels.add("F");
        }

        Map<Integer, Double> fmeasures = new HashMap<>();
        for (int i = 0; i < docsResutls.size(); i++) {
            List<LabelledDoc> labelledDocs = docsResutls.get(i);
            Instances instances = docs2Instances(labelledDocs, labels);
            if (instances.numInstances() < 10) {
                continue;
            }
            //for (Instance instance : instances) {
            // System.out.println(instance.classAttribute());
            // System.out.println(instance);
            // }

            StringToWordVector stringToWordVector = getWordFilter(instances, false);

            instances = Filter.useFilter(instances, stringToWordVector);
            AttributeSelection attributeSelection = null;

            if (attriSel) {
                attributeSelection = getAttributeSelector(instances, attNum);
                instances = attributeSelection.reduceDimensionality(instances);
            }

            createArff(instances, type + "_" + mark + ".arff");
        /*PrintWriter out = new PrintWriter(PERMISSION + "_" + mark + ".arff");
        out.print(instances.toString());
        out.close();*/
            weka.core.SerializationHelper.write(type + "_" + mark + ".filter", stringToWordVector);
            if (!user && smote) {
                instances = WekaUtils.overSampling(instances, smoteClass, smotePercent); //250);
                System.out.println(instances.numInstances());
                //instances = WekaUtils.overSampling(instances, "2", 150);
                //System.out.println(instances.numInstances());

                WekaUtils.createArff(instances, type + "_" + mark + "_smote.arff");
            }


            // Evaluate classifier and print some statistics
            Classifier classifier = buildClassifier(instances, type, true);

            try {
                fmeasures.put(i, crossValidation(instances, classifier, 5));
            } catch (Exception e) {
                e.printStackTrace();
            }

            boolean prediction = false;

            if (prediction) {
                List<LabelledDoc> labelledTestDocs = wekaUtils.getDocs("data/test");
                Instances testInstances = docs2Instances(labelledTestDocs, labels);

                testInstances = Filter.useFilter(testInstances, stringToWordVector);
                if (attriSel) {
                    testInstances = attributeSelection.reduceDimensionality(testInstances);
                }
                // Evaluate classifier and print some statistics
                Evaluation eval = new Evaluation(instances);
                eval.evaluateModel(classifier, testInstances);
                System.out.println(eval.toSummaryString("\nResults\n======\n", false));
                System.out.println(eval.toClassDetailsString());
                System.out.println(eval.toMatrixString());

                List<String> unlabelledDocs = new ArrayList<>();
                unlabelledDocs.add("xx haha lulu");
                predict(unlabelledDocs, stringToWordVector, classifier, instances.classAttribute());
            }
            // save2Arff(instances, "data_bag");
            // save2Arff(testInstances, "test_bag");
        }


    }


    public static void main(String[] args) throws Exception {
        //eval1();

        UpdateableClassifier classifier = (UpdateableClassifier) loadClassifier(new File("full.model"));
        StringToWordVector stringToWordVector = loadStr2WordVec(new File("full_location.filter"));
        Instances instances = loadArff("full_Location.arff");
        WekaUtils wekaUtils = new WekaUtils();

        List<String> labels = new ArrayList<>();
        labels.add("T");
        labels.add("F");
        List<List<LabelledDoc>> docsResutls = new ArrayList<>();
        String mark = null;
        String pyWorkLoc = "D:\\workspace\\COSPOS_MINING";
        String type = "users";

        for (int i = 0; i < 26; i++) {
            int user_num = i;
            mark = Integer.toString(user_num);
            docsResutls.add(wekaUtils.getUserDocs(pyWorkLoc + "\\output\\gnd\\" + type + "\\" + user_num)); //Location");
        }

        boolean update = true;
        double partition = 0.67;
        int weight = 1;
        Map<Integer, List<Double>> measures = new HashMap<>();

        for (int i = 0; i < docsResutls.size(); i++) {
            List<LabelledDoc> labelledDocs = docsResutls.get(i);
            if (labelledDocs.size() < 10) {
                continue;
            }

            float split = labelledDocs.size() * (float) partition; //splitD.intValue();

            Set<LabelledDoc> Tset = new HashSet<>();
            Set<LabelledDoc> Fset = new HashSet<>();
            List<LabelledDoc> trainSet = new ArrayList<>();
            List<LabelledDoc> testSet = new ArrayList<>();
            int Tcount = 0;
            int Fcount = 0;

            for (LabelledDoc labelledDoc : labelledDocs) {
                if (labelledDoc.getLabel().equals("T")) {
                    Tset.add(labelledDoc);
                } else {
                    Fset.add(labelledDoc);
                }
            }

            float Tsize = Tset.size() * (float) partition;
            float Fsize = Fset.size() * (float) partition;
            for (LabelledDoc labelledDoc : labelledDocs) {
                if (labelledDoc.getLabel().equals("T")) {
                    Tset.add(labelledDoc);
                    Tcount++;
                    if (Tcount < Tsize) {
                        trainSet.add(labelledDoc);
                    }
                } else {
                    Fset.add(labelledDoc);
                    Fcount++;
                    if (Fcount < Fsize) {
                        trainSet.add(labelledDoc);
                    }
                }
            }

            for (LabelledDoc labelledDoc : labelledDocs) {
                if (!trainSet.contains(labelledDoc)) {
                    if (trainSet.size() < split) {
                        trainSet.add(labelledDoc);
                    } else {
                        testSet.add(labelledDoc);
                    }
                }
            }


            if (update) {
                classifier = (UpdateableClassifier) loadClassifier(new File("full.model"));
                for (LabelledDoc labelledDoc : trainSet) {
                    Instance instance = genInstanceForUpdateable(labelledDoc, labels, stringToWordVector);

                    for (int k = 0; k < weight; k++) {
                        classifier.updateClassifier(instance);
                    }
                }
            }

            int wrong = 0;

            Instances testInstances = docs2Instances(testSet, labels);
            testInstances = Filter.useFilter(testInstances, stringToWordVector);
            Evaluation eval = new Evaluation(testInstances);
            eval.evaluateModel((Classifier) classifier, testInstances);
            measures.put(i, new ArrayList<Double>());
            measures.get(i).add(eval.weightedPrecision());
            measures.get(i).add(eval.weightedRecall());
            measures.get(i).add(eval.weightedFMeasure());
            System.out.println("-------------------");
            System.out.println("user: " + i);
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
            boolean showDetail = false;
            if (showDetail) {
                for (LabelledDoc labelledDoc : testSet) {
                    System.out.println(labelledDoc.getLabel() + ": " + labelledDoc.getDoc());
                    String result = predict(labelledDoc.getDoc(), stringToWordVector, (Classifier) classifier, instances.classAttribute());
                    System.out.println(result);
                    if (result.equals("1.0")) {
                        result = "F";
                    } else {
                        result = "T";
                    }

                    if (!labelledDoc.getLabel().equals(result)) {
                        wrong++;
                        //System.out.println(labelledDoc.getLabel() + ": " + labelledDoc.getDoc());
                        //System.out.println(result);
                    }
                }

                System.out.println(((float) (testSet.size() - wrong)) / testSet.size());
            }


            System.out.println(labelledDocs.size() + "," + Tset.size() + "," + Fset.size() + "," + testSet.size() + ", " + trainSet.size());
        }

        double[] f1s = new double[measures.size()];
        int j = 0;
        for (Integer index : measures.keySet()) {
            f1s[j] = measures.get(index).get(2);
            j++;
        }

        Arrays.sort(f1s);
        double median = 0.0;
        if (f1s.length % 2 == 0)
            median = ((double) f1s[f1s.length / 2] + (double) f1s[f1s.length / 2 - 1]) / 2;
        else
            median = (double) f1s[f1s.length / 2];


        PrintWriter pw = new PrintWriter(new File("results.csv"));
        StringBuilder sb = new StringBuilder();
        sb.append("id");
        sb.append(',');
        sb.append("precision");
        sb.append(',');
        sb.append("recall");
        sb.append(",");
        sb.append("f1");
        sb.append('\n');

        for (Integer index : measures.keySet()) {
            sb.append(Integer.toString(index));
            for (Double res : measures.get(index)) {
                sb.append(',');
                sb.append(Double.toString(res));
            }
            sb.append('\n');
        }
        pw.write(sb.toString());
        pw.close();
        System.out.println("done!" + median);


        boolean prediction = false;
        if (prediction) {


            List<String> unlabelledDocs = new ArrayList<>();
            unlabelledDocs.add("weather");
            unlabelledDocs.add("map");
            predict(unlabelledDocs, stringToWordVector, (Classifier) classifier, instances.classAttribute());
        }
        // save2Arff(instances, "data_bag");
        // save2Arff(testInstances, "test_bag");


    }

    /**
     * Given the classifierName, return a classifier
     *
     * @param classifierName
     *            e.g. J48, Bagging etc.
     */
    public static Classifier getClassifier(String classifierName) {
        Classifier classifier = null;
        if (classifierName.equals("J48") || classifierName.equals("j48")) {
            J48 j48 = new J48();
            j48.setUnpruned(true);
            classifier = j48;
        } else if (classifierName.equals("AdaBoostM1")) {
            AdaBoostM1 adm = new AdaBoostM1();
            adm.setNumIterations(10);
            J48 j48 = new J48();
            adm.setClassifier(j48);
            classifier = adm;
        } else if (classifierName.equals("Bagging")) {
            Bagging bagging = new Bagging();
            bagging.setNumIterations(10);
            J48 j48 = new J48();
            bagging.setClassifier(j48);
            classifier = bagging;
        } else if (classifierName.equals("Stacking")) {
            Stacking stacking = new Stacking();
            stacking.setMetaClassifier(new Logistic());
            Classifier cc[] = new Classifier[2];
            cc[0] = new J48();
            cc[1] = new IBk();
            stacking.setClassifiers(cc);
            classifier = stacking;
        } else if (classifierName.equals("AdditiveRegression")) {
            AdditiveRegression ar = new AdditiveRegression();
            ar.setClassifier(new J48());
            classifier = ar;
        } else if (classifierName.equals("LogitBoost")) {
            LogitBoost lb = new LogitBoost();
            lb.setClassifier(new J48());
            classifier = lb;
        }
        return classifier;
    }

}

