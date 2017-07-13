package fu.hao.utils;


public final class Pair<T1, T2> {
    private T1 key;
    private T2 value;

    public Pair(T1 first, T2 second) {
        this.key = first;
        this.value = second;
    }

    public void setKey(T1 first) {
        this.key = first;
    }

    public void setValue(T2 second) {
        this.value = second;
    }

    public T1 getKey() {
        return key;
    }

    public T2 getValue() {
        return value;
    }
}
