package fu.hao.analysis.flows;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by hfu on 8/18/2017.
 */
class InstancesLite {
    private Map<String, Integer> features; // <feature, index>
    private List<double[]> values;

    InstancesLite() {
        features = new HashMap<>();
        values = new ArrayList<>();
    }

    public Map<String, Integer> getFeatures() {
        return features;
    }

    public void setFeatures(Map<String, Integer> features) {
        this.features = features;
    }

    public List<double[]> getValues() {
        return values;
    }

    public void setValues(List<double[]> values) {
        this.values = values;
    }
}
