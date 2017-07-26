package fu.hao.utils;


import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.*;

/**
 * Created by hfu on 7/25/2017.
 */
public class WekaUtilsTest {
    @Test
    public void filterNumber() throws Exception {
        System.out.println(WekaUtils.filterNumber("12ab"));
    }

    @Test
    public void samePrefix() throws Exception {
        System.out.println(WekaUtils.samePrefix("chfnb9nlxboaxnolxnoyxzhcjgaqajkdagaeiilfxilixcgauofcaaq9ncwgohbfiabkhedljngaeeb9ghjaeqb9qhmaq",
                "chfnb9nlxboaxnolxnoyxzhcjgaqajkdagaeiilfxilixcgauofcaaq9ncwgohbfiabkhedljngaeeb9ghjaeqb9qhmaq9mdbi"));
    }


}
