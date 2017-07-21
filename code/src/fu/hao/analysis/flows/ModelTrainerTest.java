package fu.hao.analysis.flows;


import fu.hao.utils.Log;
import org.junit.runner.RunWith;

/**
 * Created by hfu on 7/21/2017.
 */
public class ModelTrainerTest {
    @org.junit.Test
    public void balanceClassSamples() throws Exception {
    }

    @org.junit.Test
    public void genTrainingMatrix() throws Exception {
    }

    @org.junit.Test
    public void train() throws Exception {
        ModelTrainer.train("j48", "C:\\Users\\hfu\\Documents\\flows\\CTU-13\\CTU-13-1\\1",
                "C:\\Users\\hfu\\Documents\\flows\\CTU-13\\CTU-13-1\\0");
    }



}
