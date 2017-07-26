package fu.hao.utils;

/**
 * Created by hfu on 7/21/2017.
 */
public class LabelledDoc {
    private String label;
    private String doc = null;

    public LabelledDoc(String label, String doc) {
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

    public void setDoc(String doc) {
        this.doc = doc;
    }

    public void setLabel(String label) {
        this.label = label;
    }
}
