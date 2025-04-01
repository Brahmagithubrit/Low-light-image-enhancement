class solution {
    public static void main(String args[ ]) {
        solve (4);
    }

    public static void solve(int n) {

    }

    public static void rec(int row, List<Character> columns, List<List<String>> res, int n) {
        //base case
        if (row == n) {
            // reach to end 
            res.add(constructBoard(columns));
            return;
        }
        for (int col = 0; col < n; col++) {
            if (isSafe(col , row , columns)) {
                columns.add("Q");
                rec(row + 1, columns, res, n);
                columns.remove(columns.size() - 1);
                //backtracking 
            }
        }
    }

    public static boolean isSafe(int col, int Currrow, List<Character> columns) {
        for (int row = 0; row < columns.size(); row++) {
            int currCol = columns.get(row);
            if (col == currCol || Math.abs(currCol - col) == Math.abs(Currrow - row))
                return false;
        }
        return true;
    }

    public static void constructBoard(List<Character> columns) {
        
    }
}