import java.util.*;

public class ranodm{
    public static void main(String[] args) {
        String str = "abc";
        List<String> list = new ArrayList<>();
        char[] arr = str.toCharArray();

        solve(arr, list, 0);
        System.out.println(list);
    }

    public static String  solve(String str , String temp , List<String> list, int idx) {
        if (idx == str.length()) {
            list.add(temp);
            return;
        }
        for (int i = idx; i < str.length(); i++) {
            solve(str , temp + solve(str , temp , list , i +1 ) ,  list , i + 1) ;
        }
    }

    // private static void swap(char[] arr, int i, int j) {
    //     char temp = arr[i];
    //     arr[i] = arr[j];
    //     arr[j] = temp;
    // }
}
