package fu.hao.analysis.flows;

import fu.hao.utils.Log;

import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.*;

/**
 * Created by hfu on 7/26/2017.
 */
public class WekaBagModelTrainerTest {
    @Test
    public void main() throws Exception {
        final long beforeRun = System.nanoTime();
        WekaBagModelTrainer.createArff( "C:\\Users\\hfu\\IdeaProjects\\recon\\test\\1",
                    "C:\\Users\\hfu\\IdeaProjects\\recon\\test\\0");
    }



}
